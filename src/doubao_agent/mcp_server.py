from __future__ import annotations

from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .registry import CapabilityRegistry
from .types import Risk, ToolCall
from . import __version__

# 对外 MCP 默认只暴露只读工具：外部 client 绕过本地 Agent 的审批链，
# 可逆/不可逆/系统级动作不应对外开放。
_READONLY = {Risk.READ}
_ALL = {Risk.READ, Risk.REVERSIBLE, Risk.IRREVERSIBLE, Risk.SENIOR}

_SERVER_NAME = "doubao-agent"


def _exposed_risks(expose: str) -> set[Risk]:
    return _ALL if expose == "all" else _READONLY


def _tool_from_spec(spec: Any) -> types.Tool:
    schema = spec.input_schema or {"type": "object", "properties": {}}
    return types.Tool(
        name=spec.name,
        description=spec.description,
        inputSchema=schema,
    )


def build_server(registry: CapabilityRegistry, expose: str = "read") -> Server[Any]:
    """把统一注册表包装成标准 MCP server（stdio）。

    expose="read"（默认）只暴露 L0 只读工具；expose="all" 放开全部风险等级，
    仅限本机受控场景，外部桥接不建议开启。
    """
    risks = _exposed_risks(expose)
    server: Server[Any] = Server(_SERVER_NAME, version=__version__)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            _tool_from_spec(spec)
            for spec in registry.list_tools()
            if spec.risk in risks
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        spec = registry.get(name)
        if spec is None:
            return _error_result(f"unknown tool: {name}")
        # list 过滤之外的二次拦截，防止按别名直调未暴露的高风险工具。
        if spec.risk not in risks:
            return _error_result(f"tool not exposed over MCP: {name}")

        result = await registry.call(ToolCall(name=name, arguments=arguments or {}))
        if not result.ok:
            return _error_result(result.error or "tool failed")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.as_text())]
        )

    return server


def _error_result(message: str) -> types.CallToolResult:
    return types.CallToolResult(
        isError=True,
        content=[types.TextContent(type="text", text=message)],
    )


async def serve(registry: CapabilityRegistry, expose: str = "read") -> None:
    """以 stdio 传输运行 MCP server，阻塞直到连接关闭。"""
    server = build_server(registry, expose)
    init_options = server.create_initialization_options()
    async with stdio_server() as (read, write):
        await server.run(read, write, init_options)
