from __future__ import annotations

from doubao_agent.registry import CapabilityRegistry
from doubao_agent.types import Risk, ToolCall, ToolResult, ToolSpec


class FakeSource:
    name = "fake"

    def __init__(self, risk: Risk = Risk.READ) -> None:
        self.risk = risk
        self.calls: list[ToolCall] = []

    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="ping",
                description="ping the fake channel",
                input_schema={"type": "object", "properties": {}},
                risk=self.risk,
                source=self.name,
            )
        ]

    async def call_tool(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return ToolResult(call_id=call.call_id, ok=True, content="pong")

    async def health(self) -> tuple[bool, str]:
        return True, "ok"

    async def aclose(self) -> None:
        return None


class FakeBridge:
    def __init__(self, capabilities: list[dict]) -> None:
        self.capabilities = capabilities
        self.calls: list[tuple[str, dict]] = []

    async def list_capabilities(self) -> list[dict]:
        return self.capabilities

    async def invoke(self, capability: str, payload: dict):
        self.calls.append((capability, payload))
        return {"ok": True, "capability": capability}

    async def aclose(self) -> None:
        return None


async def _registry(source: FakeSource) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.add_source(source)
    await registry.refresh()
    return registry
