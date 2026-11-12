from __future__ import annotations

from conftest import FakeSource, _registry

from doubao_agent.core.agent import Agent
from doubao_agent.core.audit import AuditLog
from doubao_agent.core.memory import MemoryStore
from doubao_agent.core.permissions import PermissionPolicy
from doubao_agent.core.recipe import RecipeStore
from doubao_agent.core.session import SessionStore
from doubao_agent.providers.mock import MockProvider
from doubao_agent.registry import CapabilityRegistry
from doubao_agent.types import ToolCall, ToolResult


class ExplodingProvider:
    async def complete(self, messages, tools):
        raise AssertionError("replay must not call the model")

    async def aclose(self) -> None:
        return None


class RecordingProvider:
    def __init__(self) -> None:
        self.messages: list[dict] | None = None

    async def complete(self, messages, tools):
        self.messages = messages
        return {"content": "好", "tool_calls": []}

    async def aclose(self) -> None:
        return None


async def test_agent_runs_tool_then_finishes(tmp_path) -> None:
    registry = await _registry(FakeSource())
    provider = MockProvider(
        script=[
            {"content": None, "tool_calls": [{"id": "c1", "name": "fake.ping", "arguments": {}}]},
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
    )

    result = await agent.run("ping 一下")
    assert result["status"] == "completed"
    assert result["answer"] == "完成"

    events = AuditLog(tmp_path / "runs").read(result["session_id"])
    assert any(e["event"] == "tool_call" and e["ok"] for e in events)
    assert any(e["event"] == "finish" for e in events)


async def test_recipe_recorded_and_replayed_without_model(tmp_path) -> None:
    source = FakeSource()
    registry = await _registry(source)
    db = tmp_path / "agent.db"
    recipes = RecipeStore(db)
    provider = MockProvider(
        script=[
            {"content": None, "tool_calls": [{"id": "c1", "name": "fake.ping", "arguments": {"x": 1}}]},
            {"content": "完成", "tool_calls": []},
        ]
    )
    agent = Agent(
        registry=registry,
        provider=provider,
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(db),
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(db),
        max_steps=4,
        recipes=recipes,
    )

    result = await agent.run("执行 ping")
    assert result["recipe_id"] is not None
    assert len(recipes.list()) == 1

    calls_before = len(source.calls)
    replayer = Agent(
        registry=registry,
        provider=ExplodingProvider(),
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(db),
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(db),
        max_steps=1,
        recipes=recipes,
    )
    replay = await replayer.replay(result["recipe_id"])

    assert replay["status"] == "completed"
    assert len(source.calls) == calls_before + 1
    assert source.calls[-1].arguments == {"x": 1}


async def test_memory_dedupes_and_reinforces(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "m.db")
    first = memory.remember("用户偏好：回答用表格", kind="preference")
    second = memory.remember("用户偏好：回答用表格", kind="preference")

    assert first == second
    rows = memory.recent(5)
    assert len(rows) == 1
    assert rows[0]["weight"] > 1.0


def test_extract_preferences_keeps_longest_only() -> None:
    from doubao_agent.core.memory import extract_preferences

    found = extract_preferences("记住：以后都用表格；另外我更喜欢简洁")
    assert any("表格" in item for item in found)
    assert any("简洁" in item for item in found)
    assert "表格" not in found


async def test_agent_consolidates_memory_after_run(tmp_path) -> None:
    registry = await _registry(FakeSource())
    db = tmp_path / "agent.db"
    memory = MemoryStore(db)
    recipes = RecipeStore(db)
    provider = MockProvider(
        script=[
            {"content": None, "tool_calls": [{"id": "c1", "name": "fake.ping", "arguments": {}}]},
            {"content": "已经完成", "tool_calls": []},
        ]
    )
    agent = Agent(
        registry=registry,
        provider=provider,
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(db),
        audit=AuditLog(tmp_path / "runs"),
        memory=memory,
        max_steps=4,
        recipes=recipes,
    )

    result = await agent.run("记住：以后都用表格；帮我 ping 一下")
    assert result["status"] == "completed"

    kinds = {row["kind"] for row in memory.recent(20)}
    assert {"preference", "procedure", "experience"} <= kinds

    events = AuditLog(tmp_path / "runs").read(result["session_id"])
    assert any(event["event"] == "memory_write" for event in events)


async def test_preferences_are_injected_into_system_prompt(tmp_path) -> None:
    db = tmp_path / "agent.db"
    memory = MemoryStore(db)
    memory.remember("回答尽量用表格", kind="preference")
    provider = RecordingProvider()
    agent = Agent(
        registry=CapabilityRegistry(),
        provider=provider,
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(db),
        audit=AuditLog(tmp_path / "runs"),
        memory=memory,
        max_steps=2,
    )

    await agent.run("帮我整理数据")

    assert provider.messages is not None
    system = provider.messages[0]["content"]
    assert "用户长期偏好" in system
    assert "回答尽量用表格" in system


def test_trim_messages_keeps_system_and_user_boundary() -> None:
    from doubao_agent.core.agent import _trim_messages

    messages = [{"role": "system", "content": "sys"}]
    for index in range(10):
        messages.append({"role": "user", "content": f"u{index}"})
        messages.append({"role": "assistant", "content": f"a{index}"})
        messages.append({"role": "tool", "tool_call_id": f"t{index}", "content": "r"})

    trimmed = _trim_messages(messages, 8)

    assert trimmed[0]["content"] == "sys"
    assert len(trimmed) <= 8
    assert trimmed[1]["role"] == "user"


async def test_usage_counts_model_calls(tmp_path) -> None:
    from doubao_agent.core.usage import UsageMeter

    registry = await _registry(FakeSource())
    provider = MockProvider([{"content": "完成", "tool_calls": []}])
    agent = Agent(
        registry=registry,
        provider=provider,
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(tmp_path / "agent.db"),
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(tmp_path / "agent.db"),
        max_steps=2,
        usage=UsageMeter(),
    )

    result = await agent.run("算一下")

    assert result["usage"]["calls"] == 1


async def test_tool_output_is_marked_untrusted(tmp_path) -> None:
    class RecordingProvider:
        def __init__(self) -> None:
            self.messages: list[list[dict]] = []

        async def complete(self, messages, tools):
            self.messages.append([dict(message) for message in messages])
            if len(self.messages) == 1:
                return {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "name": "fake.ping", "arguments": {}}
                    ],
                }
            return {"content": "完成", "tool_calls": []}

        async def aclose(self) -> None:
            return None

    registry = await _registry(FakeSource())
    provider = RecordingProvider()
    agent = Agent(
        registry=registry,
        provider=provider,
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(tmp_path / "agent.db"),
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(tmp_path / "agent.db"),
        max_steps=3,
    )

    await agent.run("调用一次工具")

    tool_messages = [
        message
        for batch in provider.messages
        for message in batch
        if message.get("role") == "tool"
    ]
    assert tool_messages
    assert tool_messages[0]["content"].startswith("【工具输出")


async def test_resume_reuses_history_without_duplicating_task(tmp_path) -> None:
    registry = await _registry(FakeSource())
    provider = MockProvider(
        [
            {"content": "第一轮完成", "tool_calls": []},
            {"content": "第二轮完成", "tool_calls": []},
        ]
    )
    sessions = SessionStore(tmp_path / "agent.db")
    agent = Agent(
        registry=registry,
        provider=provider,
        policy=PermissionPolicy("auto"),
        sessions=sessions,
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(tmp_path / "agent.db"),
        max_steps=3,
    )

    first = await agent.run("原始任务")
    second = await agent.run("另一个任务", first["session_id"])

    rows = sessions.messages(first["session_id"])
    assert sum(1 for row in rows if row.get("role") == "user") == 1
    assert second["answer"] == "第二轮完成"


async def test_agent_cancellation_short_circuits(tmp_path) -> None:
    import asyncio

    registry = await _registry(FakeSource())
    agent = Agent(
        registry=registry,
        provider=ExplodingProvider(),
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(tmp_path / "agent.db"),
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(tmp_path / "agent.db"),
        max_steps=3,
        cancel_event=asyncio.Event(),
    )

    agent.cancel()
    result = await agent.run("随便做什么")

    assert result["status"] == "cancelled"
    assert result["answer"] == "任务已被取消"


async def test_tool_artifacts_are_registered(tmp_path) -> None:
    class WritingSource(FakeSource):
        session_id = ""

        async def call_tool(self, call: ToolCall) -> ToolResult:
            target = tmp_path / "runs" / self.session_id / "report.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("ok", encoding="utf-8")
            return ToolResult(call_id=call.call_id, ok=True, content="written")

    source = WritingSource()
    registry = await _registry(source)
    provider = MockProvider(
        script=[
            {"content": None, "tool_calls": [{"id": "c1", "name": "fake.ping", "arguments": {}}]},
            {"content": "完成", "tool_calls": []},
        ]
    )
    db = tmp_path / "agent.db"
    sessions = SessionStore(db)
    agent = Agent(
        registry=registry,
        provider=provider,
        policy=PermissionPolicy("auto"),
        sessions=sessions,
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(db),
        max_steps=4,
    )

    def on_event(event: dict) -> None:
        if event.get("type") == "start":
            source.session_id = event["session"]

    result = await agent.run("写报告", on_event=on_event)
    assert result["status"] == "completed"

    paths = [row["path"] for row in sessions.artifacts(result["session_id"])]
    assert str(tmp_path / "runs" / result["session_id"] / "report.txt") in paths
    assert all(not path.endswith("audit.jsonl") for path in paths)


async def test_artifact_scan_is_incremental_and_deduped(tmp_path) -> None:
    registry = await _registry(FakeSource())
    db = tmp_path / "agent.db"
    sessions = SessionStore(db)
    agent = Agent(
        registry=registry,
        provider=MockProvider(),
        policy=PermissionPolicy("auto"),
        sessions=sessions,
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(db),
        max_steps=4,
    )

    session_id = "s-artifacts"
    root = tmp_path / "runs" / session_id
    (root / "nested").mkdir(parents=True)
    (root / "first.txt").write_text("1", encoding="utf-8")
    (root / "audit.jsonl").write_text("{}", encoding="utf-8")

    agent._register_artifacts(session_id)
    agent._register_artifacts(session_id)
    paths = [row["path"] for row in sessions.artifacts(session_id)]
    assert str(root / "first.txt") in paths
    assert len(paths) == 1, "同一产物不应重复登记"

    # 目录 mtime 未变时走缓存；新增嵌套文件后仍能被发现且不重复
    (root / "nested" / "second.txt").write_text("2", encoding="utf-8")
    agent._register_artifacts(session_id)
    agent._register_artifacts(session_id)
    paths = [row["path"] for row in sessions.artifacts(session_id)]
    assert str(root / "nested" / "second.txt") in paths
    assert len(paths) == 2
