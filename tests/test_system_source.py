from __future__ import annotations

from doubao_agent.sources.system import SystemSource
from doubao_agent.types import Risk, ToolCall


async def test_system_status_reports_readonly_metrics(tmp_path) -> None:
    source = SystemSource(tmp_path)
    specs = await source.list_tools()
    assert len(specs) == 2
    status = next(s for s in specs if s.name == "status")
    assert status.risk is Risk.READ and status.idempotent is True

    result = await source.call_tool(ToolCall(name="status", call_id="c1"))
    assert result.ok
    assert "CPU 核数" in result.content
    assert "RSS" in result.content
    assert "PID" in result.content
    assert result.meta["metrics"]["cpu_count"] is not None
    assert result.meta["metrics"]["pid"] > 0


async def test_net_check_reports_unreachable(tmp_path) -> None:
    source = SystemSource(tmp_path)
    result = await source.call_tool(
        ToolCall(name="net_check", arguments={"host": "127.0.0.1", "port": 1, "timeout": 1.0}, call_id="c1")
    )
    assert result.ok
    probe = result.meta["probe"]
    assert probe["reachable"] is False
    assert probe["host"] == "127.0.0.1" and probe["port"] == 1


async def test_net_check_rejects_bad_port(tmp_path) -> None:
    source = SystemSource(tmp_path)
    result = await source.call_tool(
        ToolCall(name="net_check", arguments={"host": "127.0.0.1", "port": "abc"}, call_id="c1")
    )
    assert not result.ok and "port" in result.error


async def test_system_status_unknown_tool_fails(tmp_path) -> None:
    source = SystemSource(tmp_path)
    result = await source.call_tool(ToolCall(name="nope", call_id="c1"))
    assert not result.ok and "unknown system tool" in result.error
