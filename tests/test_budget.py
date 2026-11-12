from __future__ import annotations

from conftest import _registry

from doubao_agent.core.agent import Agent
from doubao_agent.core.audit import AuditLog
from doubao_agent.core.memory import MemoryStore
from doubao_agent.core.permissions import PermissionPolicy
from doubao_agent.core.planner import Plan
from doubao_agent.core.session import SessionStore
from doubao_agent.core.usage import UsageMeter
from doubao_agent.providers.mock import MockProvider
from doubao_agent.types import Risk, ToolCall, ToolResult, ToolSpec


class FlakySource:
    """参数缺 text 时失败，补齐后成功；用于验证失败修正。"""

    name = "flaky"

    def __init__(self) -> None:
        self.arguments: list[dict] = []

    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="write",
                description="写入文本",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                risk=Risk.READ,
                source=self.name,
            )
        ]

    async def call_tool(self, call: ToolCall) -> ToolResult:
        self.arguments.append(dict(call.arguments or {}))
        text = str((call.arguments or {}).get("text") or "")
        if not text:
            return ToolResult(call_id=call.call_id, ok=False, error="text is required")
        return ToolResult(call_id=call.call_id, ok=True, content=f"wrote {text}")

    async def health(self) -> tuple[bool, str]:
        return True, "ok"

    async def aclose(self) -> None:
        return None


class ScriptedRepairProvider:
    """第一次返回修正后的工具调用，之后给最终答复。"""

    def __init__(self, tool: str, arguments: dict) -> None:
        self.tool = tool
        self.arguments = arguments
        self.calls = 0

    async def complete(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": None,
                "tool_calls": [{"id": "r1", "name": self.tool, "arguments": self.arguments}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        return {"content": "完成", "tool_calls": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    async def aclose(self) -> None:
        return None


def test_usage_meter_budget() -> None:
    meter = UsageMeter(max_calls=2)
    assert meter.exceeded() is None
    meter.record({"prompt_tokens": 1, "completion_tokens": 1})
    assert meter.exceeded() is None
    meter.record({"prompt_tokens": 1, "completion_tokens": 1})
    assert "调用预算" in (meter.exceeded() or "")

    tokens = UsageMeter(max_tokens=100)
    tokens.record({"prompt_tokens": 60, "completion_tokens": 50})
    assert "token 预算" in (tokens.exceeded() or "")

    unlimited = UsageMeter()
    unlimited.record({"prompt_tokens": 10**6, "completion_tokens": 10**6})
    assert unlimited.exceeded() is None


async def _agent(tmp_path, registry, provider, usage=None, repair_rounds=1) -> Agent:
    db = tmp_path / "agent.db"
    return Agent(
        registry=registry,
        provider=provider,
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(db),
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(db),
        max_steps=5,
        usage=usage or UsageMeter(),
        repair_rounds=repair_rounds,
    )


async def test_agent_stops_when_call_budget_exhausted(tmp_path) -> None:
    from conftest import FakeSource

    registry = await _registry(FakeSource())
    call = {"id": "c1", "name": "fake.ping", "arguments": {}}
    provider = MockProvider(
        script=[
            {
                "content": None,
                "tool_calls": [call],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            }
        ]
        * 3
    )
    agent = await _agent(tmp_path, registry, provider, usage=UsageMeter(max_calls=1))
    out = await agent.run("反复调用工具")

    assert out["status"] == "incomplete"
    assert "调用预算已用尽" in out["answer"]
    assert out["usage"]["calls"] == 1


async def test_planned_step_repairs_arguments_once(tmp_path) -> None:
    registry = await _registry(FlakySource())
    provider = ScriptedRepairProvider("flaky.write", {"text": "修正后"})
    agent = await _agent(tmp_path, registry, provider)
    plan = Plan(mode="steps", steps=[{"tool": "flaky.write", "arguments": {}}])

    out = await agent._execute_plan("写一段文本", plan, None)

    assert out["status"] == "completed"
    assert "wrote 修正后" in out["answer"]
    assert provider.calls == 1
    assert agent.usage.calls == 1


async def test_planned_step_repair_respects_round_limit(tmp_path) -> None:
    registry = await _registry(FlakySource())
    provider = ScriptedRepairProvider("flaky.write", {})
    agent = await _agent(tmp_path, registry, provider, repair_rounds=0)
    plan = Plan(mode="steps", steps=[{"tool": "flaky.write", "arguments": {}}])

    out = await agent._execute_plan("写一段文本", plan, None)

    assert out["status"] == "incomplete"
    assert provider.calls == 0
