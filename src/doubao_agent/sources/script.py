from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from ..types import Risk, SourceKind, ToolCall, ToolResult, ToolSpec


def _set_cpu_limit(seconds: float):
    """Return a preexec_fn that caps CPU time in the child (0 disables)."""
    if seconds <= 0:
        return None

    def _limit() -> None:
        import resource

        try:
            resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
        except (ValueError, OSError):
            pass

    return _limit

_MAX_OUTPUT = 8000

_SCHEMAS: dict[str, dict[str, Any]] = {
    "run_python": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "完整可执行的 Python 代码，用 print 输出关键结果"},
            "cwd": {"type": "string", "description": "工作目录，默认云电脑工作区"},
            "timeout": {"type": "number", "description": "超时秒数，默认 120"},
        },
        "required": ["code"],
    },
    "run_shell": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "bash 命令（非交互）"},
            "cwd": {"type": "string", "description": "工作目录，默认云电脑工作区"},
            "timeout": {"type": "number", "description": "超时秒数，默认 120"},
        },
        "required": ["command"],
    },
}


class ScriptSource:
    """Code-as-action: one script execution replaces many model round-trips."""

    kind = SourceKind.BUILTIN

    def __init__(
        self,
        workspace: str | Path = "/workspace",
        risk: Risk = Risk.REVERSIBLE,
        default_timeout: float = 120.0,
        max_cpu_seconds: float = 0.0,
        cancel_event: asyncio.Event | None = None,
        name: str = "script",
    ) -> None:
        self.workspace = Path(workspace)
        self.risk = risk
        self.name = name
        self.default_timeout = default_timeout
        self.max_cpu_seconds = max_cpu_seconds
        self.cancel_event = cancel_event
        self._processes: set[asyncio.subprocess.Process] = set()
        self._specs = [
            ToolSpec(
                name="run_python",
                description=(
                    "执行一段 Python 代码，一次完成多步操作（读写文件、调用库、数据处理、批量请求）。"
                    "优先用它替代多次零散工具调用以降低成本。代码必须非交互，结果用 print 输出。"
                ),
                input_schema=_SCHEMAS["run_python"],
                kind=SourceKind.BUILTIN,
                risk=risk,
                source=name,
                idempotent=False,
            ),
            ToolSpec(
                name="run_shell",
                description=(
                    "执行非交互 bash 命令，一次完成多步系统操作。"
                    "优先用它替代多次零散工具调用以降低成本。"
                ),
                input_schema=_SCHEMAS["run_shell"],
                kind=SourceKind.BUILTIN,
                risk=risk,
                source=name,
                idempotent=False,
            ),
        ]

    async def list_tools(self) -> list[ToolSpec]:
        return list(self._specs)

    def _truncate(self, text: str) -> str:
        if len(text) <= _MAX_OUTPUT:
            return text
        return text[:_MAX_OUTPUT] + f"\n...[truncated {len(text) - _MAX_OUTPUT} chars]"

    def _resolve_cwd(self, raw: Any) -> Path:
        """把 cwd 固定在工作区内，阻断参数越界到工作区之外。"""
        root = self.workspace.resolve()
        target = Path(str(raw or root)).expanduser()
        if not target.is_absolute():
            target = root / target
        target = target.resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"cwd outside workspace: {target}")
        return target

    @staticmethod
    async def _kill_tree(process: asyncio.subprocess.Process) -> None:
        """回收整个进程组，避免 bash 拉起的子进程变成孤儿。"""
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

    async def call_tool(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        args = call.arguments or {}
        try:
            cwd = str(self._resolve_cwd(args.get("cwd")))
        except ValueError as exc:
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                error=str(exc),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        timeout = float(args.get("timeout") or self.default_timeout)

        if call.name == "run_python":
            code = str(args.get("code", ""))
            if not code.strip():
                return ToolResult(call_id=call.call_id, ok=False, error="code is required")
            argv = [sys.executable, "-c", code]
        elif call.name == "run_shell":
            command = str(args.get("command", ""))
            if not command.strip():
                return ToolResult(call_id=call.call_id, ok=False, error="command is required")
            # 非登录 shell：不加载 profile，避免别名 / PATH 被环境注入。
            argv = ["/bin/bash", "-c", command]
        else:
            return ToolResult(call_id=call.call_id, ok=False, error=f"unknown script tool: {call.name}")

        # 优先使用本次调用携带的取消信号，实现会话级取消互不干扰。
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
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_set_cpu_limit(self.max_cpu_seconds),
                start_new_session=True,
            )
            self._processes.add(process)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        communicate = asyncio.ensure_future(process.communicate())
        cancel_wait: asyncio.Future | None = None
        try:
            if cancel_event is None:
                stdout, stderr = await asyncio.wait_for(communicate, timeout=timeout)
            else:
                cancel_wait = asyncio.ensure_future(cancel_event.wait())
                done, _ = await asyncio.wait(
                    {communicate, cancel_wait}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
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
                error=f"timeout after {timeout:g}s",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        finally:
            if cancel_wait is not None:
                cancel_wait.cancel()
            self._processes.discard(process)

        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        parts = []
        if out:
            parts.append(f"[stdout]\n{out.rstrip()}")
        if err:
            parts.append(f"[stderr]\n{err.rstrip()}")
        content = self._truncate("\n".join(parts) or "(no output)")
        code = process.returncode
        if code is None:
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                content=content,
                error="process exited without an exit code",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                meta={"exit_code": None},
            )
        return ToolResult(
            call_id=call.call_id,
            ok=code == 0,
            content=content,
            error=None if code == 0 else f"exit code {code}",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            meta={"exit_code": code},
        )

    async def cancel(self) -> None:
        """Kill every subprocess this channel is currently running."""
        for process in list(self._processes):
            await self._kill_tree(process)

    async def health(self) -> tuple[bool, str]:
        return True, f"python {sys.version.split()[0]} · cwd {self.workspace}"

    async def aclose(self) -> None:
        await self.cancel()
