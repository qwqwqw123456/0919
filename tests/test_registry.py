from __future__ import annotations

from conftest import FakeSource, _registry

from doubao_agent.types import ToolCall


async def test_registry_registers_and_resolves_by_alias() -> None:
    registry = await _registry(FakeSource())
    names = [spec.name for spec in registry.list_tools()]
    assert names == ["fake.ping"]
    assert registry.get("ping") is not None

    result = await registry.call(ToolCall(name="ping", call_id="c1"))
    assert result.ok and result.content == "pong"
