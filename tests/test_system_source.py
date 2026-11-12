from __future__ import annotations

from doubao_agent.sources.system import SystemSource
from doubao_agent.types import Risk, ToolCall


async def test_system_status_reports_readonly_metrics(tmp_path) -> None:
    source = SystemSource(tmp_path)
    specs = await source.list_tools()
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "status"
    assert spec.risk is Risk.READ and spec.idempotent is True

    result = await source.call_tool(ToolCall(name="status", call_id="c1"))
    assert result.ok
    assert "CPU 核数" in result.content
    assert "RSS" in result.content
    assert result.meta["metrics"]["cpu_count"] is not None


async def test_system_status_unknown_tool_fails(tmp_path) -> None:
    source = SystemSource(tmp_path)
    result = await source.call_tool(ToolCall(name="nope", call_id="c1"))
    assert not result.ok and "unknown system tool" in result.error
