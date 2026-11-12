from __future__ import annotations

import asyncio
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
        self._process: asyncio.subprocess.Process | None = None
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

    async def call_tool(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        args = call.arguments or {}
        cwd = str(args.get("cwd") or self.workspace)
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
            argv = ["/bin/bash", "-lc", command]
        else:
            return ToolResult(call_id=call.call_id, ok=False, error=f"unknown script tool: {call.name}")

        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_set_cpu_limit(self.max_cpu_seconds),
            )
            self._process = process
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                error=f"timeout after {timeout:g}s",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        finally:
            self._process = None

        if self.cancel_event is not None and self.cancel_event.is_set():
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                error="cancelled",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        parts = []
        if out:
            parts.append(f"[stdout]\n{out.rstrip()}")
        if err:
            parts.append(f"[stderr]\n{err.rstrip()}")
        content = self._truncate("\n".join(parts) or "(no output)")
        code = process.returncode or 0
        return ToolResult(
            call_id=call.call_id,
            ok=code == 0,
            content=content,
            error=None if code == 0 else f"exit code {code}",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            meta={"exit_code": code},
        )

    async def cancel(self) -> None:
        """Kill the currently running subprocess, if any."""
        process = self._process
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()

    async def health(self) -> tuple[bool, str]:
        return True, f"python {sys.version.split()[0]} · cwd {self.workspace}"

    async def aclose(self) -> None:
        await self.cancel()
