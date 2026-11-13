from __future__ import annotations

from mcp.shared.memory import create_connected_server_and_client_session

from conftest import FakeSource, _registry

from doubao_agent.mcp_server import build_server
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
