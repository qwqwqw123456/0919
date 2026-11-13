from __future__ import annotations

import json
import time

import pytest
from conftest import FakeSource, _registry

from doubao_agent.core.agent import Agent
from doubao_agent.core.audit import AuditLog
from doubao_agent.core.coordinator import Coordinator
from doubao_agent.core.knowledge import KnowledgeStore, route_domain
from doubao_agent.core.limits import RateLimiter
from doubao_agent.core.memory import MemoryStore
from doubao_agent.core.permissions import PermissionPolicy
from doubao_agent.core.journal import ToolJournal
from doubao_agent.core.session import SessionStore
from doubao_agent.core.telemetry import Telemetry
from doubao_agent.providers.mock import MockProvider
from doubao_agent.registry import CapabilityRegistry
from doubao_agent.types import Risk, ToolCall, ToolResult


def _agent(tmp_path, registry, provider, **kwargs) -> Agent:
    db = tmp_path / "agent.db"
    return Agent(
        registry=registry,
        provider=provider,
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(db),
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(db),
        max_steps=6,
        **kwargs,
    )


async def test_journal_replays_repeatable_tool_on_resume(tmp_path) -> None:
    source = FakeSource()
    registry = await _registry(source)
    db = tmp_path / "agent.db"
    agent = _agent(tmp_path, registry, MockProvider(), journal=ToolJournal(db))
    agent._allow_replay = True  # 模拟从检查点恢复的会话

    first = await agent._execute("fake.ping", {"x": 1}, "c1", "s1", step=1)
    second = await agent._execute("fake.ping", {"x": 1}, "c2", "s1", step=1)

    assert first.ok and second.ok
    assert len(source.calls) == 1
    assert second.meta.get("replayed") is True
    events = AuditLog(tmp_path / "runs").read("s1")
    assert any(event["event"] == "tool_replay" for event in events)


async def test_journal_never_replays_irreversible_tool(tmp_path) -> None:
    source = FakeSource(risk=Risk.IRREVERSIBLE)
    registry = await _registry(source)
    db = tmp_path / "agent.db"
    agent = _agent(tmp_path, registry, MockProvider(), journal=ToolJournal(db))
    agent._allow_replay = True

    await agent._execute("fake.ping", {}, "c1", "s1")
    await agent._execute("fake.ping", {}, "c2", "s1")

    assert len(source.calls) == 2


async def test_repeated_call_within_single_run_is_not_replayed(tmp_path) -> None:
    source = FakeSource()
    registry = await _registry(source)
    provider = MockProvider(
        script=[
            {
                "content": None,
                "tool_calls": [
                    {"id": "c1", "name": "fake.ping", "arguments": {"x": 1}},
                    {"id": "c2", "name": "fake.ping", "arguments": {"x": 1}},
                ],
            },
            {"content": "完成", "tool_calls": []},
        ]
    )
    agent = _agent(tmp_path, registry, provider, journal=ToolJournal(tmp_path / "agent.db"))

    await agent.run("同一次运行内重复调用")

    assert len(source.calls) == 2


class FlakySource(FakeSource):
    """第一次连接失败，第二次成功；用于验证策略化重试。"""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    async def call_tool(self, call: ToolCall) -> ToolResult:
        self.attempts += 1
        self.calls.append(call)
        if self.attempts == 1:
            return ToolResult(call_id=call.call_id, ok=False, error="connection reset by peer")
        return ToolResult(call_id=call.call_id, ok=True, content="pong")


async def test_transient_failure_is_retried(tmp_path) -> None:
    source = FlakySource()
    registry = await _registry(source)
    provider = MockProvider(
        script=[
            {"content": None, "tool_calls": [{"id": "c1", "name": "fake.ping", "arguments": {}}]},
            {"content": "恢复", "tool_calls": []},
        ]
    )
    agent = _agent(tmp_path, registry, provider, tool_retries=1)

    result = await agent.run("先失败后成功")

    assert result["status"] == "completed"
    assert source.attempts == 2
    events = AuditLog(tmp_path / "runs").read(result["session_id"])
    assert any(event["event"] == "tool_retry" for event in events)


def test_rate_limiter_blocks_per_source_and_recovers(tmp_path) -> None:
    now = [0.0]
    limiter = RateLimiter({"fake": 2}, window=60.0, clock=lambda: now[0])
    registry = CapabilityRegistry(limiter)

    assert limiter.allow("fake") and limiter.allow("fake")
    assert not limiter.allow("fake")

    now[0] = 61.0
    assert limiter.allow("fake")


async def test_registry_applies_rate_limit(tmp_path) -> None:
    source = FakeSource()
    registry = CapabilityRegistry(RateLimiter({"fake": 1, "default": 100}))
    registry.add_source(source)
    await registry.refresh()

    first = await registry.call(ToolCall(name="fake.ping", arguments={}, call_id="1"))
    second = await registry.call(ToolCall(name="fake.ping", arguments={}, call_id="2"))

    assert first.ok
    assert not second.ok and "rate limited" in (second.error or "")


def test_telemetry_records_and_aggregates(tmp_path) -> None:
    telemetry = Telemetry(tmp_path / "agent.db")
    telemetry.record("s1", "model.complete", kind="model", duration_ms=120)
    telemetry.record("s1", "fake.ping", kind="tool", ok=False, duration_ms=30)

    summary = telemetry.summary("s1")

    assert summary["spans"] == 2
    assert summary["failed"] == 1
    assert summary["total_ms"] == 150
    assert summary["tools"] == {"fake.ping": 1}
    assert summary["slowest"][0]["name"] == "model.complete"
    assert telemetry.summary()["sessions"] == 1


def test_memory_decay_and_feedback(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "agent.db")
    now = time.time()
    fact_id = memory.remember("用户在做知识库项目", kind="fact")
    pref_id = memory.remember("回答要简洁", kind="preference")

    memory._conn.execute(
        "UPDATE facts SET used_at = ? WHERE id IN (?, ?)",
        (now - 30 * 86400, fact_id, pref_id),
    )
    memory._conn.commit()

    changed = memory.decay(half_life_days=30.0, now=now)

    assert changed == 1
    weights = {item["id"]: item["weight"] for item in memory.recent(10)}
    assert weights[fact_id] == pytest.approx(0.5, abs=0.05)
    assert weights[pref_id] == pytest.approx(1.0)

    def weight_of(target: int) -> float:
        return next(item["weight"] for item in memory.recent(10) if item["id"] == target)

    memory.feedback([fact_id], ok=True)
    assert weight_of(fact_id) >= 0.7
    boosted = weight_of(fact_id)
    memory.feedback([fact_id], ok=False)
    assert weight_of(fact_id) < boosted


def test_knowledge_auto_routes_to_domain(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "agent.db")
    store.add_text("补贴申报", "最新补贴政策与申报流程", domain="07")
    store.add_text("定价策略", "商业定价与投资回报", domain="03")

    assert route_domain("最新补贴政策有哪些") == "07"
    hits = store.search("最新补贴政策", domain="auto")

    assert hits and all(hit["domain"] == "07" for hit in hits)


class StubAgent:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def run(self, task, session_id=None, on_event=None, images=None):
        return {"status": "completed", "answer": self.answer, "session_id": "s1"}


class ReviewProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return {"content": '["子任务A", "子任务B"]', "tool_calls": []}
        payload = {
            "answer": "合并后的最终结论",
            "conflicts": ["A 与 B 口径不一致"],
            "verdict": "needs_attention",
        }
        return {"content": json.dumps(payload, ensure_ascii=False), "tool_calls": []}

    async def aclose(self) -> None:
        return None


async def test_coordinator_review_merges_results() -> None:
    answers = iter(["A 结果", "B 结果"])
    provider = ReviewProvider()
    coordinator = Coordinator(lambda: StubAgent(next(answers)), provider)

    result = await coordinator.run_parallel("写一份对比", review=True)

    assert provider.calls == 2
    assert result["review"]["verdict"] == "needs_attention"
    assert result["review"]["conflicts"] == ["A 与 B 口径不一致"]
    assert result["answer"] == "合并后的最终结论"
    assert result["status"] == "incomplete"
    assert all(item["role"] == "executor" for item in result["subtasks"])
    assert "review" in result["blackboard"]


async def test_coordinator_without_review_skips_reviewer() -> None:
    answers = iter(["A 结果", "B 结果"])
    provider = ReviewProvider()
    coordinator = Coordinator(lambda: StubAgent(next(answers)), provider)

    result = await coordinator.run_parallel("写一份对比")

    assert provider.calls == 1
    assert result["review"] is None
    assert result["status"] == "completed"


def test_audit_export_verifies_then_writes(tmp_path) -> None:
    log = AuditLog(tmp_path / "runs")
    log.append("s1", "start", task="demo")
    log.append("s1", "finish", answer="ok")

    target, count = log.export("s1", tmp_path / "export.jsonl")

    assert count == 2
    assert len(target.read_text("utf-8").strip().splitlines()) == 2

    original = (tmp_path / "runs" / "s1" / "audit.jsonl").read_text("utf-8")
    (tmp_path / "runs" / "s1" / "audit.jsonl").write_text(
        original.replace('"start"', '"tampered"'), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        log.export("s1", tmp_path / "export2.jsonl")
