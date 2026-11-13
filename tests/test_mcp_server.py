from __future__ import annotations

import asyncio
import socket

import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import create_connected_server_and_client_session

from conftest import FakeSource, _registry

from doubao_agent.mcp_server import build_http_app, build_server
from doubao_agent.types import Risk


async def test_readonly_tool_is_listed_and_callable() -> None:
    registry = await _registry(FakeSource(risk=Risk.READ))
    server = build_server(registry, expose="read")

    async with create_connected_server_and_client_session(server, raise_exceptions=True) as session:
        tools = await session.list_tools()
        names = [t.name for t in tools.tools]
        assert names == ["fake.ping"]

        result = await session.call_tool("fake.ping", {})
        assert result.isError is False
        assert result.content[0].text == "pong"


async def test_write_risk_tool_is_hidden_by_default() -> None:
    registry = await _registry(FakeSource(risk=Risk.REVERSIBLE))
    server = build_server(registry, expose="read")

    async with create_connected_server_and_client_session(server, raise_exceptions=True) as session:
        tools = await session.list_tools()
        assert [t.name for t in tools.tools] == []

        # 直调未暴露工具：服务端二次拦截，返回 isError 而不是执行。
        result = await session.call_tool("fake.ping", {})
        assert result.isError is True
        assert "not exposed" in result.content[0].text


async def test_expose_all_reveals_write_risk_tool() -> None:
    registry = await _registry(FakeSource(risk=Risk.REVERSIBLE))
    server = build_server(registry, expose="all")

    async with create_connected_server_and_client_session(server, raise_exceptions=True) as session:
        tools = await session.list_tools()
        assert [t.name for t in tools.tools] == ["fake.ping"]

        result = await session.call_tool("fake.ping", {})
        assert result.isError is False
        assert result.content[0].text == "pong"


async def test_unknown_tool_reports_error() -> None:
    registry = await _registry(FakeSource(risk=Risk.READ))
    server = build_server(registry, expose="read")

    async with create_connected_server_and_client_session(server, raise_exceptions=True) as session:
        result = await session.call_tool("no.such.tool", {})
        assert result.isError is True
        assert "unknown tool" in result.content[0].text


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


async def _start_http(registry, expose: str = "read"):
    server = build_server(registry, expose=expose)
    app = build_http_app(server)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=_free_port(), log_level="warning"
    )
    uv = uvicorn.Server(config)
    task = asyncio.create_task(uv.serve())
    for _ in range(60):
        if uv.started:
            break
        await asyncio.sleep(0.05)
    url = f"http://127.0.0.1:{config.port}/mcp"
    return uv, task, url


async def test_http_readonly_tool_is_listed_and_callable() -> None:
    registry = await _registry(FakeSource(risk=Risk.READ))
    uv, task, url = await _start_http(registry)
    try:
        async with streamable_http_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert [t.name for t in tools.tools] == ["fake.ping"]

                result = await session.call_tool("fake.ping", {})
                assert result.isError is False
                assert result.content[0].text == "pong"
    finally:
        uv.should_exit = True
        await task


async def test_http_write_risk_tool_is_hidden_by_default() -> None:
    registry = await _registry(FakeSource(risk=Risk.REVERSIBLE))
    uv, task, url = await _start_http(registry)
    try:
        async with streamable_http_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert [t.name for t in tools.tools] == []

                result = await session.call_tool("fake.ping", {})
                assert result.isError is True
                assert "not exposed" in result.content[0].text
    finally:
        uv.should_exit = True
        await task
