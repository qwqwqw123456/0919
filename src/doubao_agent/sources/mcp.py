from __future__ import annotations

import asyncio
import os
import socket
import time
from contextlib import AsyncExitStack
from typing import Any
from urllib.parse import urlparse

import httpx

from ..config import MCPServerConfig
from ..types import SourceKind, ToolCall, ToolResult, ToolSpec
from ..utils import dumps, infer_risk, jsonable
from .base import SourceError

# 云电脑的能力入口是标准 MCP（平台侧 8091 端口），启动时可自动探测。
DEFAULT_HOST_MCP_URL = "http://127.0.0.1:8091/mcp"


def _tcp_reachable(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def probe_http_mcp(
    url: str = DEFAULT_HOST_MCP_URL,
    timeout: float = 3.0,
    client: httpx.AsyncClient | None = None,
) -> bool:
    """探测标准 MCP 端点：发一次 JSON-RPC initialize，拿到 result 才算可用。

    探测失败只表示当前不可达，不代表配置有误；调用方应跳过而不是报错。
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "doubao-agent", "version": "0.2.0"},
        },
    }
    headers = {"Accept": "application/json, text/event-stream"}
    try:
        if client is not None:
            response = await client.post(url, json=payload, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=timeout, trust_env=True) as owner:
                response = await owner.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception:  # noqa: BLE001 - 探测是 best-effort
        return False
    return isinstance(data, dict) and isinstance(data.get("result"), dict)


class MCPSource:
    """MCP over stdio, SSE or Streamable HTTP. Tools are discovered, never hardcoded."""

    kind = SourceKind.MCP

    def __init__(self, cfg: MCPServerConfig) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self._stack: AsyncExitStack | None = None
        self._session: Any = None
        self._failed: str | None = None

    async def _ensure(self) -> Any:
        if self._session is not None:
            return self._session
        if self._failed is not None:
            raise SourceError(self._failed)

        await self._precheck()
        try:
            session = await self._connect()
        except BaseException as exc:  # noqa: BLE001 - 含 anyio 取消域异常，一律判不可用
            self._failed = f"MCP {self.name} 初始化失败：{type(exc).__name__}: {exc}"
            await self._discard()
            raise SourceError(self._failed) from exc
        self._session = session
        return session

    async def _precheck(self) -> None:
        """http/sse 先做一次 TCP 连通性探测，避免在端口未开放时触发客户端清理异常。"""
        transport = self.cfg.transport.lower()
        if transport not in ("http", "streamable_http", "streamable-http", "sse"):
            return
        if not self.cfg.url:
            return
        parsed = urlparse(self.cfg.url)
        host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return
        reachable = await asyncio.to_thread(_tcp_reachable, host, port, 1.0)
        if not reachable:
            raise SourceError(f"MCP {self.name} 不可达：{host}:{port}")

    async def _discard(self) -> None:
        stack, self._stack = self._stack, None
        self._session = None
        if stack is None:
            return
        try:
            await stack.aclose()
        except BaseException:  # noqa: BLE001 - 清理阶段的取消域报错一律吞掉
            pass

    async def _connect(self) -> Any:
        try:
            from mcp import ClientSession
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SourceError("mcp SDK is not installed") from exc

        self._stack = AsyncExitStack()
        transport = self.cfg.transport.lower()

        if transport == "stdio":
            if not self.cfg.command:
                raise SourceError(f"MCP server '{self.name}' has no command")
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=self.cfg.command,
                args=self.cfg.args,
                env={**os.environ, **self.cfg.env} if self.cfg.env else None,
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        elif transport == "sse":
            from mcp.client.sse import sse_client

            if not self.cfg.url:
                raise SourceError(f"MCP server '{self.name}' has no url")
            read, write = await self._stack.enter_async_context(sse_client(self.cfg.url))
        elif transport in ("http", "streamable_http", "streamable-http"):
            from mcp.client.streamable_http import streamablehttp_client

            if not self.cfg.url:
                raise SourceError(f"MCP server '{self.name}' has no url")
            read, write, *_ = await self._stack.enter_async_context(
                streamablehttp_client(self.cfg.url)
            )
        else:
            raise SourceError(f"unknown MCP transport: {transport}")

        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def list_tools(self) -> list[ToolSpec]:
        session = await self._ensure()
        result = await session.list_tools()
        specs: list[ToolSpec] = []
        for tool in getattr(result, "tools", []) or []:
            description = getattr(tool, "description", "") or ""
            specs.append(
                ToolSpec(
                    name=tool.name,
                    description=description,
                    input_schema=getattr(tool, "inputSchema", None) or {},
                    kind=SourceKind.MCP,
                    risk=infer_risk(tool.name, description),
                    source=self.name,
                )
            )
        return specs

    async def call_tool(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        try:
            session = await self._ensure()
            result = await session.call_tool(call.name, call.arguments)
        except Exception as exc:  # noqa: BLE001 - surface as a tool error
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        parts: list[str] = []
        for item in getattr(result, "content", []) or []:
            if getattr(item, "type", None) == "text":
                parts.append(item.text)
            else:
                parts.append(dumps(jsonable(item)))
        content = "\n".join(parts)
        ok = not bool(getattr(result, "isError", False))
        return ToolResult(
            call_id=call.call_id,
            ok=ok,
            content=content,
            error=None if ok else content or "tool reported an error",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    async def health(self) -> tuple[bool, str]:
        try:
            tools = await self.list_tools()
            return True, f"{len(tools)} tools"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    async def aclose(self) -> None:
        await self._discard()
