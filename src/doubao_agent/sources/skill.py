from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

from ..skills import Skill, SkillManager
from ..types import Risk, SourceKind, ToolCall, ToolResult, ToolSpec

_MAX_OUTPUT = 8000


def _risk(value: str) -> Risk:
    try:
        return Risk(value)
    except ValueError:
        return Risk.REVERSIBLE


class SkillSource:
    """Skill 通道：可执行技能注册为工具，知识技能按需阅读。"""

    name = "skill"

    def __init__(self, manager: SkillManager, cancel_event: asyncio.Event | None = None) -> None:
        self.manager = manager
        self.cancel_event = cancel_event

    async def list_tools(self) -> list[ToolSpec]:
        specs = [
            ToolSpec(
                name="read",
                description="阅读指定技能的完整说明与资源清单（渐进式加载）。按技能说明执行前先调用它。",
                input_schema={
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "技能名"}},
                    "required": ["name"],
                },
                kind=SourceKind.SKILL,
                risk=Risk.READ,
                source=self.name,
            )
        ]
        for skill in self.manager.list():
            if not skill.executable:
                continue
            specs.append(
                ToolSpec(
                    name=skill.name,
                    description=f"[技能] {skill.description}",
                    input_schema=skill.input_schema or {"type": "object", "properties": {}},
                    kind=SourceKind.SKILL,
                    risk=_risk(skill.risk),
                    tags=skill.tags,
                    idempotent=False,
                    source=self.name,
                )
            )
        return specs

    async def call_tool(self, call: ToolCall) -> ToolResult:
        if call.name == "read":
            name = str((call.arguments or {}).get("name", ""))
            skill = self.manager.get(name)
            if skill is None:
                return ToolResult(call_id=call.call_id, ok=False, error=f"unknown skill: {name}")
            self.manager._meter(skill.name, "view")
            resources = "\n".join(f"- {item}" for item in skill.resources()) or "-（无）"
            content = (
                f"# {skill.name} v{skill.version}\n\n"
                f"{skill.body}\n\n"
                f"## 资源\n{resources}\n\n"
                f"## 入口\n{skill.entry or '（知识型技能，无脚本入口）'}"
            )
            return ToolResult(call_id=call.call_id, ok=True, content=content)

        skill = self.manager.get(call.name)
        if skill is None or not skill.executable:
            return ToolResult(call_id=call.call_id, ok=False, error=f"unknown skill tool: {call.name}")
        result = await self._run(skill, call)
        self.manager._meter(skill.name, "use")
        return result

    @staticmethod
    async def _kill_tree(process: asyncio.subprocess.Process) -> None:
        """回收整个进程组，避免技能脚本拉起的子进程变成孤儿。"""
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        try:
            await process.wait()
        except ProcessLookupError:
            pass

    async def _run(self, skill: Skill, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        root = skill.directory.resolve()
        entry_path = (root / Path(skill.entry or "")).resolve()
        if not entry_path.is_relative_to(root) or not entry_path.is_file():
            return ToolResult(
                call_id=call.call_id, ok=False, error=f"skill entry not found: {skill.entry}"
            )

        argv = (
            [sys.executable, str(entry_path)]
            if skill.runtime == "python"
            else ["/bin/bash", str(entry_path)]
        )
        env = dict(os.environ)
        env["SKILL_NAME"] = skill.name
        env["SKILL_DIR"] = str(skill.directory)

        # 会话级取消优先于源级取消，互不干扰（与脚本通道同口径）。
        cancel_event = call.cancel_event or self.cancel_event
        if cancel_event is not None and cancel_event.is_set():
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                error="cancelled",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(skill.directory),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        payload = json.dumps(call.arguments or {}, ensure_ascii=False).encode("utf-8")
        communicate = asyncio.ensure_future(process.communicate(payload))
        cancel_wait: asyncio.Future | None = None
        try:
            if cancel_event is None:
                stdout, stderr = await asyncio.wait_for(communicate, timeout=skill.timeout)
            else:
                cancel_wait = asyncio.ensure_future(cancel_event.wait())
                done, _ = await asyncio.wait(
                    {communicate, cancel_wait},
                    timeout=skill.timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise asyncio.TimeoutError
                if cancel_wait in done and not communicate.done():
                    await self._kill_tree(process)
                    communicate.cancel()
                    return ToolResult(
                        call_id=call.call_id,
                        ok=False,
                        error="cancelled",
                        elapsed_ms=int((time.perf_counter() - started) * 1000),
                    )
                stdout, stderr = communicate.result()
        except asyncio.TimeoutError:
            await self._kill_tree(process)
            communicate.cancel()
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                error=f"timeout after {skill.timeout:g}s",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        finally:
            if cancel_wait is not None:
                cancel_wait.cancel()

        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        parts = []
        if out:
            parts.append(f"[stdout]\n{out.rstrip()}")
        if err:
            parts.append(f"[stderr]\n{err.rstrip()}")
        content = "\n".join(parts) or "(no output)"
        if len(content) > _MAX_OUTPUT:
            content = content[:_MAX_OUTPUT] + f"\n...[truncated {len(content) - _MAX_OUTPUT} chars]"

        code = process.returncode or 0
        return ToolResult(
            call_id=call.call_id,
            ok=code == 0,
            content=content,
            error=None if code == 0 else f"exit code {code}",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            meta={"exit_code": code},
        )

    async def health(self) -> tuple[bool, str]:
        skills = self.manager.list()
        executable = sum(1 for skill in skills if skill.executable)
        return True, f"{len(skills)} skills ({executable} executable) @ {self.manager.root}"

    async def aclose(self) -> None:
        return None
