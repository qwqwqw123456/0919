from __future__ import annotations

import json

from conftest import FakeSource, _registry

from doubao_agent.core.agent import Agent
from doubao_agent.core.audit import AuditLog
from doubao_agent.core.memory import MemoryStore
from doubao_agent.core.permissions import PermissionPolicy
from doubao_agent.core.session import SessionStore
from doubao_agent.providers.mock import MockProvider
from doubao_agent.registry import CapabilityRegistry


class PlanProvider:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def complete(self, messages, tools):
        self.calls += 1
        return {"content": self.content, "tool_calls": []}

    async def aclose(self) -> None:
        return None


def test_parse_plan_accepts_fenced_json_and_rejects_noise() -> None:
    from doubao_agent.core.planner import parse_plan

    plan = parse_plan('```json\n{"mode":"tools","steps":[{"tool":"a.b","arguments":{"x":1}}]}\n```')
    assert plan is not None
    assert plan.mode == "tools"
    assert plan.steps[0]["tool"] == "a.b"

    assert parse_plan("抱歉，我无法完成") is None
    assert parse_plan('{"mode":"script"}') is None


async def test_decompose_falls_back_to_single_task_on_provider_error() -> None:
    from doubao_agent.core.coordinator import Coordinator

    class BrokenProvider:
        async def complete(self, messages, tools):
            raise TimeoutError("decompose timeout")

    coordinator = Coordinator(lambda: None, BrokenProvider())
    assert await coordinator.decompose("一个复杂任务") == ["一个复杂任务"]


async def test_entry_planning_executes_script_with_single_model_call(tmp_path) -> None:
    from doubao_agent.core.planner import Planner
    from doubao_agent.sources.script import ScriptSource

    registry = CapabilityRegistry()
    registry.add_source(ScriptSource(tmp_path))
    await registry.refresh()

    provider = PlanProvider(
        json.dumps({"mode": "script", "language": "python", "code": "print(6 * 7)"})
    )
    db = tmp_path / "agent.db"
    agent = Agent(
        registry=registry,
        provider=provider,
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(db),
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(db),
        max_steps=4,
        planner=Planner(provider),
    )

    result = await agent.run("算一下 6 乘 7")

    assert result["status"] == "completed"
    assert "42" in result["answer"]
    assert result["planned"] is True
    assert provider.calls == 1


async def test_invalid_plan_falls_back_to_normal_loop(tmp_path) -> None:
    from doubao_agent.core.planner import Planner

    registry = await _registry(FakeSource())
    provider = MockProvider(
        script=[
            {"content": "我先想一想（不是 JSON）", "tool_calls": []},
            {"content": "完成", "tool_calls": []},
        ]
    )
    db = tmp_path / "agent.db"
    agent = Agent(
        registry=registry,
        provider=provider,
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(db),
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(db),
        max_steps=4,
        planner=Planner(provider),
    )

    result = await agent.run("随便看看")
    assert result["status"] == "completed"
    assert result.get("planned") is None
