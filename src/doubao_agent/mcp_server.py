from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Route

from .registry import CapabilityRegistry
from .types import Risk, ToolCall
from . import __version__

# 对外 MCP 默认只暴露只读工具：外部 client 绕过本地 Agent 的审批链，
# 可逆/不可逆/系统级动作不应对外开放。
_READONLY = {Risk.READ}
_ALL = {Risk.READ, Risk.REVERSIBLE, Risk.IRREVERSIBLE, Risk.SENIOR}

_SERVER_NAME = "doubao-agent"
_DEFAULT_HTTP_HOST = "127.0.0.1"
_DEFAULT_HTTP_PORT = 8766
_DEFAULT_HTTP_PATH = "/mcp"


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


def build_http_app(server: Server[Any], path: str = _DEFAULT_HTTP_PATH) -> Starlette:
    """把已构造的 MCP server 包装成 Streamable HTTP ASGI 应用。"""
    manager = StreamableHTTPSessionManager(server)

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        async with manager.run():
            yield

    class _MCPASGI:
        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            await manager.handle_request(scope, receive, send)

    return Starlette(routes=[Route(path, endpoint=_MCPASGI())], lifespan=lifespan)


async def serve_http(
    registry: CapabilityRegistry,
    expose: str = "read",
    host: str = _DEFAULT_HTTP_HOST,
    port: int = _DEFAULT_HTTP_PORT,
    path: str = _DEFAULT_HTTP_PATH,
) -> None:
    """以 Streamable HTTP 传输运行 MCP server，阻塞直到服务关闭。

    默认只绑本机回环地址；对外网暴露需显式改 host，并确认已做访问控制。
    """
    server = build_server(registry, expose)
    app = build_http_app(server, path=path)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    await uvicorn.Server(config).serve()
