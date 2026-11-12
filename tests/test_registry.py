from __future__ import annotations

import asyncio

from conftest import FakeSource, _registry

from doubao_agent.registry import CapabilityRegistry
from doubao_agent.types import Risk, ToolCall, ToolSpec, ToolResult


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


class _ConcurrencySource:
    name = "conc"

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def list_tools(self):
        return [
            ToolSpec(
                name="ping",
                description="x",
                input_schema={"type": "object", "properties": {}},
                risk=Risk.READ,
                source=self.name,
            )
        ]

    async def call_tool(self, call: ToolCall) -> ToolResult:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return ToolResult(call_id=call.call_id, ok=True, content="x")


async def _run_concurrent(registry: CapabilityRegistry, n: int) -> None:
    await asyncio.gather(
        *[registry.call(ToolCall(name="ping", call_id=f"c{i}")) for i in range(n)]
    )


async def test_channel_concurrency_limits_inflight() -> None:
    source = _ConcurrencySource()
    registry = CapabilityRegistry(channel_concurrency=1, cache_ttl=0.0)
    registry.add_source(source)
    await registry.refresh()

    results = await asyncio.gather(
        *[registry.call(ToolCall(name="ping", call_id=f"c{i}")) for i in range(5)]
    )
    assert all(r.ok for r in results)
    assert source.peak == 1


async def test_channel_concurrency_zero_is_unlimited() -> None:
    source = _ConcurrencySource()
    registry = CapabilityRegistry(channel_concurrency=0, cache_ttl=0.0)
    registry.add_source(source)
    await registry.refresh()

    await _run_concurrent(registry, 5)
    assert source.peak == 5
