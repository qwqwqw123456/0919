from __future__ import annotations

from conftest import FakeSource, _registry

from doubao_agent.registry import CapabilityRegistry
from doubao_agent.types import Risk, ToolCall


async def test_registry_registers_and_resolves_by_alias() -> None:
    registry = await _registry(FakeSource())
    names = [spec.name for spec in registry.list_tools()]
    assert names == ["fake.ping"]
    assert registry.get("ping") is not None

    result = await registry.call(ToolCall(name="ping", call_id="c1"))
    assert result.ok and result.content == "pong"


async def test_readonly_idempotent_result_is_cached() -> None:
    source = FakeSource()
    registry = await _registry(source)

    first = await registry.call(ToolCall(name="ping", arguments={"q": "a"}, call_id="c1"))
    second = await registry.call(ToolCall(name="ping", arguments={"q": "a"}, call_id="c2"))

    assert first.ok and second.ok
    assert len(source.calls) == 1
    assert second.meta.get("cached") is True


async def test_cache_keyed_by_arguments() -> None:
    source = FakeSource()
    registry = await _registry(source)

    await registry.call(ToolCall(name="ping", arguments={"q": "a"}, call_id="c1"))
    await registry.call(ToolCall(name="ping", arguments={"q": "b"}, call_id="c2"))

    assert len(source.calls) == 2


async def test_write_risk_tool_is_not_cached() -> None:
    source = FakeSource(risk=Risk.REVERSIBLE)
    registry = await _registry(source)

    await registry.call(ToolCall(name="ping", arguments={"q": "a"}, call_id="c1"))
    await registry.call(ToolCall(name="ping", arguments={"q": "a"}, call_id="c2"))

    assert len(source.calls) == 2


async def test_cache_disabled_when_ttl_zero() -> None:
    source = FakeSource()
    registry = CapabilityRegistry(cache_ttl=0.0)
    registry.add_source(source)
    await registry.refresh()

    await registry.call(ToolCall(name="ping", arguments={"q": "a"}, call_id="c1"))
    await registry.call(ToolCall(name="ping", arguments={"q": "a"}, call_id="c2"))

    assert len(source.calls) == 2
