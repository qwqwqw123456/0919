"""Expose the unified capability registry as a standard MCP stdio server.

External MCP clients (including `traecli mcp add`) launch this process and see
the same tools the built-in agent uses. Calls still pass through the configured
permission policy: in the default `guarded` mode, L2/L3 tools are denied when no
interactive approver is available, which is the safe default for a headless
stdio server.
"""

from __future__ import annotations

import hashlib
import re
import sys
import uuid
from contextlib import redirect_stdout
from typing import Any

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

from . import __version__
from .config import Config
from .core.audit import AuditLog
from .runtime import Runtime
from .types import Risk, SourceKind, ToolCall, ToolSpec

MCP_SESSION = "mcp"
_MAX_TEXT = 20000
_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _external_name(name: str) -> str:
    """Map registry names (source.tool) to MCP-safe names (source__tool)."""
    cleaned = _NAME_RE.sub("_", name.replace(".", "__")).strip("_") or "tool"
    if len(cleaned) > 64:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[:55]}_{digest}"
    return cleaned


def _build_index(specs: list[ToolSpec]) -> dict[str, ToolSpec]:
    index: dict[str, ToolSpec] = {}
    for spec in specs:
        base = _external_name(spec.name)
        candidate = base
        duplicate = 2
        while candidate in index:
            candidate = f"{base[:60]}_{duplicate}"
            duplicate += 1
        index[candidate] = spec
    return index


def _resolve(index: dict[str, ToolSpec], name: str) -> ToolSpec | None:
    if name in index:
        return index[name]
    return next((spec for spec in index.values() if spec.name == name), None)


def _input_schema(spec: ToolSpec) -> dict[str, Any]:
    schema = spec.input_schema
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}
    if schema.get("type") == "object":
        return schema
    return {"type": "object", "properties": {"value": schema}}


def _annotations(spec: ToolSpec) -> ToolAnnotations:
    return ToolAnnotations(
        readOnlyHint=spec.risk is Risk.READ,
        destructiveHint=spec.risk in (Risk.IRREVERSIBLE, Risk.SENIOR),
        idempotentHint=spec.idempotent,
        openWorldHint=spec.kind in (SourceKind.API, SourceKind.BROWSER, SourceKind.MCP),
    )


def _error(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], isError=True)


def _audit(audit: AuditLog, event: str, **payload: Any) -> None:
    try:
        audit.append(MCP_SESSION, event, **payload)
    except OSError as exc:  # audit must never break a tool call
        print(f"[mcp-serve] audit write failed: {exc}", file=sys.stderr)


def build_server(runtime: Runtime) -> Server:
    """Build an MCP server backed by one already-initialized runtime."""
    server = Server(
        "doubao-agent",
        version=__version__,
        instructions=(
            "Doubao Agent unified capability registry. Tool names are MCP-safe "
            "(source__tool); the original registry name is shown as the tool title. "
            "Write and destructive calls follow permission_mode and auto_approve."
        ),
    )
    index = _build_index(runtime.registry.list_tools())
    policy = runtime.agent.policy
    audit = runtime.agent.audit

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=external,
                title=spec.name,
                description=spec.description,
                inputSchema=_input_schema(spec),
                annotations=_annotations(spec),
            )
            for external, spec in sorted(index.items())
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> CallToolResult:
        spec = _resolve(index, name)
        if spec is None:
            return _error(f"unknown tool: {name}")
        call = ToolCall(
            name=spec.name,
            arguments=dict(arguments) if isinstance(arguments, dict) else {},
            call_id=uuid.uuid4().hex,
        )
        try:
            # Sources occasionally print; stdout is the MCP transport and must stay clean.
            with redirect_stdout(sys.stderr):
                decision = await policy.authorize(spec, call)
                if not decision.allowed:
                    _audit(audit, "tool_denied", tool=spec.name, reason=decision.reason)
                    return _error(f"denied: {decision.reason}")
                result = await runtime.registry.call(call)
        except Exception as exc:  # noqa: BLE001 - surface as a tool error, keep the server alive
            _audit(audit, "tool_error", tool=spec.name, error=f"{type(exc).__name__}: {exc}")
            return _error(f"{type(exc).__name__}: {exc}")

        _audit(
            audit,
            "tool_call",
            tool=spec.name,
            ok=result.ok,
            elapsed_ms=result.elapsed_ms,
            error=result.error,
        )
        return CallToolResult(
            content=[TextContent(type="text", text=result.as_text(limit=_MAX_TEXT))],
            isError=not result.ok,
        )

    return server


async def serve_stdio(cfg: Config) -> int:
    """Run the registry as a stdio MCP server until the client disconnects."""
    try:
        # Runtime.build prints host-discovery notices to stdout; redirect them
        # before stdio_server binds to the real stdout.
        with redirect_stdout(sys.stderr):
            runtime = await Runtime.build(cfg, approver=None)
    except Exception as exc:  # noqa: BLE001 - report startup failure on stderr
        print(f"[mcp-serve] 启动失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        server = build_server(runtime)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    except Exception as exc:  # noqa: BLE001 - a broken pipe ends the session
        print(f"[mcp-serve] 会话结束：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        with redirect_stdout(sys.stderr):
            await runtime.aclose()
    return 0
