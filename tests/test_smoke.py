from __future__ import annotations

import json

from doubao_agent.core.agent import Agent
from doubao_agent.core.audit import AuditLog
from doubao_agent.core.memory import MemoryStore
from doubao_agent.core.permissions import PermissionPolicy
from doubao_agent.core.recipe import RecipeStore
from doubao_agent.core.session import SessionStore
from doubao_agent.providers.mock import MockProvider
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


async def _registry(source: FakeSource) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.add_source(source)
    await registry.refresh()
    return registry


async def test_registry_registers_and_resolves_by_alias() -> None:
    registry = await _registry(FakeSource())
    names = [spec.name for spec in registry.list_tools()]
    assert names == ["fake.ping"]
    assert registry.get("ping") is not None

    result = await registry.call(ToolCall(name="ping", call_id="c1"))
    assert result.ok and result.content == "pong"


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


async def test_guarded_mode_denies_irreversible_without_approver() -> None:
    policy = PermissionPolicy("guarded")
    spec = ToolSpec(name="crm.delete", description="delete a record", risk=Risk.IRREVERSIBLE)
    decision = await policy.authorize(spec, ToolCall(name="crm.delete"))
    assert decision.needs_approval is True
    assert decision.allowed is False


async def test_guarded_mode_allows_read() -> None:
    policy = PermissionPolicy("guarded")
    spec = ToolSpec(name="crm.list", description="list records", risk=Risk.READ)
    decision = await policy.authorize(spec, ToolCall(name="crm.list"))
    assert decision.allowed is True


async def test_script_mode_runs_python_in_one_call(tmp_path) -> None:
    from doubao_agent.sources.script import ScriptSource

    source = ScriptSource(tmp_path)
    specs = await source.list_tools()
    assert {spec.name for spec in specs} == {"run_python", "run_shell"}

    result = await source.call_tool(
        ToolCall(
            name="run_python",
            arguments={"code": "from pathlib import Path\nPath('out.txt').write_text('hi')\nprint((2 + 3) * 7)"},
            call_id="s1",
        )
    )
    assert result.ok
    assert "35" in result.content
    assert (tmp_path / "out.txt").read_text() == "hi"


async def test_script_mode_reports_nonzero_exit(tmp_path) -> None:
    from doubao_agent.sources.script import ScriptSource

    source = ScriptSource(tmp_path)
    result = await source.call_tool(
        ToolCall(name="run_shell", arguments={"command": "exit 3"}, call_id="s2")
    )
    assert result.ok is False
    assert "exit code 3" in (result.error or "")


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


async def test_native_source_discovers_every_capability() -> None:
    from doubao_agent.sources.native import NativeSource

    bridge = FakeBridge(
        [
            {"name": "clipboard.read", "description": "读取剪贴板"},
            {"name": "notify.send", "description": "发送系统通知",
             "input_schema": {"type": "object", "properties": {"text": {"type": "string"}},
                              "required": ["text"]}},
            {"name": "media.camera", "description": "调用摄像头", "risk": "L2"},
        ]
    )
    source = NativeSource(bridge)
    specs = await source.list_tools()

    assert {spec.name for spec in specs} == {"clipboard.read", "notify.send", "media.camera"}
    camera = next(spec for spec in specs if spec.name == "media.camera")
    assert camera.risk is Risk.IRREVERSIBLE

    result = await source.call_tool(
        ToolCall(name="notify.send", arguments={"text": "hi"}, call_id="n1")
    )
    assert result.ok
    assert bridge.calls == [("notify.send", {"text": "hi"})]


async def test_native_source_falls_back_when_manifest_empty() -> None:
    from doubao_agent.sources.native import NativeSource

    source = NativeSource(FakeBridge([]))
    names = {spec.name for spec in await source.list_tools()}
    assert "browser.open" in names
    assert "desktop.screenshot" in names


async def test_native_tools_keep_their_namespace() -> None:
    from doubao_agent.sources.native import NativeSource

    registry = CapabilityRegistry()
    registry.add_source(NativeSource(FakeBridge([{"name": "notify.send", "description": "通知"}])))
    await registry.refresh()
    assert [spec.name for spec in registry.list_tools()] == ["notify.send"]


class ExplodingProvider:
    async def complete(self, messages, tools):
        raise AssertionError("replay must not call the model")

    async def aclose(self) -> None:
        return None


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


class RecordingProvider:
    def __init__(self) -> None:
        self.messages: list[dict] | None = None

    async def complete(self, messages, tools):
        self.messages = messages
        return {"content": "好", "tool_calls": []}

    async def aclose(self) -> None:
        return None


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


def _write_skill(root, name: str, meta: str, body: str) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(f"---\n{meta}\n---\n{body}", encoding="utf-8")


def test_parse_skill_reads_frontmatter(tmp_path) -> None:
    from doubao_agent.skills import parse_skill

    _write_skill(
        tmp_path,
        "report",
        "name: report\ndescription: 生成周报\ntags: [report, weekly]\nversion: 1.2\nrisk: L1",
        "# 周报\n先收集数据，再输出表格。",
    )
    skill = parse_skill(tmp_path / "report" / "SKILL.md")

    assert skill is not None
    assert skill.name == "report"
    assert skill.tags == ("report", "weekly")
    assert skill.version == "1.2"
    assert skill.executable is False
    assert "先收集数据" in skill.body


def test_skill_without_frontmatter_is_ignored(tmp_path) -> None:
    from doubao_agent.skills import parse_skill

    directory = tmp_path / "plain"
    directory.mkdir()
    (directory / "SKILL.md").write_text("# 没有元数据", encoding="utf-8")
    assert parse_skill(directory / "SKILL.md") is None


async def test_executable_skill_registers_and_runs(tmp_path) -> None:
    from doubao_agent.skills import SkillManager
    from doubao_agent.sources.skill import SkillSource

    _write_skill(
        tmp_path,
        "hello",
        "name: hello\ndescription: 生成问候\nentry: scripts/main.py\nruntime: python\n"
        "input_schema:\n  type: object\n  properties:\n    name: {type: string}\n"
        "  required: [name]",
        "# 问候技能\n按名字生成一句问候。",
    )
    scripts = tmp_path / "hello" / "scripts"
    scripts.mkdir()
    (scripts / "main.py").write_text(
        "import json, sys\n"
        "args = json.load(sys.stdin)\n"
        "print(f\"hello {args.get('name', 'world')}\")\n",
        encoding="utf-8",
    )

    manager = SkillManager(tmp_path)
    assert manager.refresh() == 1

    registry = CapabilityRegistry()
    registry.add_source(SkillSource(manager))
    await registry.refresh()

    names = {spec.name for spec in registry.list_tools()}
    assert {"skill.read", "skill.hello"} <= names

    result = await registry.call(
        ToolCall(name="skill.hello", arguments={"name": "豆包"}, call_id="s1")
    )
    assert result.ok
    assert "hello 豆包" in result.content


async def test_skill_read_returns_body_and_resources(tmp_path) -> None:
    from doubao_agent.skills import SkillManager
    from doubao_agent.sources.skill import SkillSource

    _write_skill(tmp_path, "report", "name: report\ndescription: 生成周报", "先收集数据。")
    (tmp_path / "report" / "template.md").write_text("模板", encoding="utf-8")

    manager = SkillManager(tmp_path)
    manager.refresh()
    source = SkillSource(manager)

    result = await source.call_tool(
        ToolCall(name="read", arguments={"name": "report"}, call_id="r1")
    )
    assert result.ok
    assert "先收集数据" in result.content
    assert "template.md" in result.content


async def test_knowledge_skill_is_injected_into_planner_prompt(tmp_path) -> None:
    from doubao_agent.core.planner import Planner
    from doubao_agent.skills import SkillManager

    _write_skill(
        tmp_path,
        "weekly",
        "name: weekly\ndescription: 周报写法\ntags: [周报]",
        "先收集本周数据，再按部门输出表格。",
    )
    manager = SkillManager(tmp_path)
    manager.refresh()

    class CapturingProvider:
        def __init__(self) -> None:
            self.messages: list[list[dict]] = []

        async def complete(self, messages, tools):
            self.messages.append(messages)
            return {"content": json.dumps({"mode": "answer", "answer": "ok"}), "tool_calls": []}

        async def aclose(self) -> None:
            return None

    provider = CapturingProvider()
    planner = Planner(provider, manager)
    plan = await planner.plan("帮我写周报", [], "")

    assert plan is not None and plan.mode == "answer"
    system = provider.messages[0][0]["content"]
    assert "周报写法" in system
    assert "先收集本周数据" in system


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


def test_cron_next_run_matches_next_midnight() -> None:
    from datetime import datetime

    from doubao_agent.core.scheduler import next_run, parse_cron

    spec = parse_cron("*/5 * * * *")
    assert spec is not None and spec["minute"] == {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}

    after = datetime(2026, 11, 12, 15, 30).timestamp()
    expected = datetime(2026, 11, 13, 0, 0).timestamp()
    assert next_run("0 0 * * *", after=after) == expected
    assert next_run("not a cron", after=after) is None


def test_scheduler_add_list_due_remove(tmp_path) -> None:
    import time

    from doubao_agent.core.scheduler import ScheduleStore

    store = ScheduleStore(tmp_path / "s.db")
    schedule_id = store.add("每日", "0 0 * * *", "生成日报")

    assert schedule_id > 0
    rows = store.list()
    assert rows and rows[0]["next_run"] is not None
    # 未来某个时刻必然到期
    due = store.due(now=time.time() + 10**9)
    assert any(row["id"] == schedule_id for row in due)

    store.set_enabled(schedule_id, False)
    assert store.due(now=time.time() + 10**9) == []

    assert store.remove(schedule_id) is True
    assert store.get(schedule_id) is None

    try:
        store.add("bad", "61 * * * *", "x")
        raise AssertionError("invalid cron should raise")
    except ValueError:
        pass


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


async def test_script_reports_cancelled_when_event_set(tmp_path) -> None:
    import asyncio

    from doubao_agent.sources.script import ScriptSource

    event = asyncio.Event()
    event.set()
    source = ScriptSource(tmp_path, cancel_event=event)

    result = await source.call_tool(
        ToolCall(name="run_python", arguments={"code": "print(1)"}, call_id="s1")
    )

    assert result.ok is False
    assert result.error == "cancelled"


def test_knowledge_chunking_and_search(tmp_path) -> None:
    from doubao_agent.core.knowledge import KnowledgeStore

    store = KnowledgeStore(tmp_path / "k.db")
    doc_id = store.add_text(
        "报销制度",
        "公司报销周期为每月 25 日。\n\n差旅费需提前申请，住宿标准每晚不超过 500 元。\n\n餐饮补贴每人每天 80 元。",
    )

    assert doc_id > 0
    hits = store.search("报销周期是什么时候", limit=3)
    assert hits and hits[0]["doc_id"] == doc_id
    assert "25" in hits[0]["text"]

    # 语义近似：没有完全相同的词，bigram 重叠也能命中相关块
    hits2 = store.search("出差住宿标准")
    assert any("住宿" in hit["text"] for hit in hits2)

    assert store.stats()["documents"] == 1
    assert store.remove(doc_id) is True
    assert store.list_documents() == []


async def test_knowledge_source_tools(tmp_path) -> None:
    from doubao_agent.core.knowledge import KnowledgeStore
    from doubao_agent.sources.knowledge import KnowledgeSource

    store = KnowledgeStore(tmp_path / "k.db")
    source = KnowledgeSource(store)

    specs = await source.list_tools()
    assert {"search", "add"} <= {spec.name for spec in specs}

    add = await source.call_tool(
        ToolCall(name="add", arguments={"title": "T", "text": "重要结论：A 等于 B"}, call_id="a")
    )
    assert add.ok is True

    search = await source.call_tool(
        ToolCall(name="search", arguments={"query": "A 和 B 的关系"}, call_id="s")
    )
    assert search.ok is True and "A" in search.content


def test_host_provider_normalizes_shapes() -> None:
    from doubao_agent.providers.host import HostProvider

    assert HostProvider._normalize("你好") == {"content": "你好", "tool_calls": []}

    simple = HostProvider._normalize(
        {"content": "hi", "tool_calls": [{"id": "a", "name": "ping", "arguments": {}}]}
    )
    assert simple["content"] == "hi"
    assert simple["tool_calls"][0]["name"] == "ping"

    openai_style = HostProvider._normalize(
        {
            "choices": [
                {
                    "message": {
                        "content": "x",
                        "tool_calls": [
                            {"id": "b", "function": {"name": "go", "arguments": '{"n": 1}'}}
                        ],
                    }
                }
            ]
        }
    )
    assert openai_style["content"] == "x"
    assert openai_style["tool_calls"][0]["name"] == "go"
    assert openai_style["tool_calls"][0]["arguments"] == {"n": 1}


async def test_host_provider_completes_via_bridge() -> None:
    from doubao_agent.providers.host import HostProvider

    class FakeBridge:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def invoke(self, capability: str, payload: dict) -> dict:
            self.calls.append((capability, payload))
            return {"content": "pong", "tool_calls": []}

        async def list_capabilities(self) -> list:
            return []

        async def aclose(self) -> None:
            return None

    bridge = FakeBridge()
    provider = HostProvider(bridge, capability="model.complete", model="doubao")
    out = await provider.complete([{"role": "user", "content": "hi"}], [])

    assert out["content"] == "pong"
    capability, payload = bridge.calls[0]
    assert capability == "model.complete"
    assert payload["model"] == "doubao"
    assert payload["messages"][0]["content"] == "hi"


def test_build_provider_host_branch() -> None:
    from doubao_agent.providers import build_provider
    from doubao_agent.providers.host import HostProvider
    from doubao_agent.providers.mock import MockProvider

    class Dummy:
        pass

    provider = build_provider(
        {"provider": "host", "capability": "model.complete", "model": "x"}, bridge=Dummy()
    )
    assert isinstance(provider, HostProvider)

    # 宿主桥缺失时退回 mock，避免运行时崩溃
    fallback = build_provider({"provider": "host"}, bridge=None)
    assert isinstance(fallback, MockProvider)


def test_parse_string_list_shapes() -> None:
    from doubao_agent.core.coordinator import _parse_string_list

    assert _parse_string_list('["a", "b"]') == ["a", "b"]
    assert _parse_string_list('```json\n["x", "y"]\n```') == ["x", "y"]
    assert _parse_string_list('{"subtasks": ["p", "q"]}') == ["p", "q"]
    # 兜底：非 JSON 时按行提取双引号字符串
    assert _parse_string_list('任务1："整理数据"\n任务2："写报告"') == ["整理数据", "写报告"]
    assert _parse_string_list("") == []


async def test_coordinator_decomposes_task() -> None:
    from doubao_agent.core.coordinator import Coordinator

    class FakeProvider:
        async def complete(self, messages, tools):
            return {"content": '["a", "b", "c"]', "tool_calls": []}

    coordinator = Coordinator(lambda: None, FakeProvider())
    assert await coordinator.decompose("任意任务") == ["a", "b", "c"]


async def test_coordinator_runs_subtasks_in_parallel() -> None:
    import asyncio

    from doubao_agent.core.coordinator import Coordinator

    class FakeAgent:
        active = 0
        max_active = 0

        async def run(self, task, on_event=None):
            FakeAgent.active += 1
            FakeAgent.max_active = max(FakeAgent.max_active, FakeAgent.active)
            await asyncio.sleep(0.02)
            FakeAgent.active -= 1
            return {"status": "completed", "answer": f"done:{task}", "session_id": task}

    spawned: list[FakeAgent] = []

    def spawn() -> FakeAgent:
        agent = FakeAgent()
        spawned.append(agent)
        return agent

    class FakeProvider:
        async def complete(self, messages, tools):
            return {"content": "", "tool_calls": []}

    coordinator = Coordinator(spawn, FakeProvider())
    result = await coordinator.run_parallel("主任务", subtasks=["子1", "子2", "子3"])

    assert result["status"] == "completed"
    assert result["completed"] == 3
    assert len(spawned) == 3
    assert "done:子1" in result["answer"]
    assert "done:子3" in result["answer"]
    # 三个子任务确实并发执行，而非串行
    assert FakeAgent.max_active == 3



async def test_policy_rule_preauthorizes_without_approver() -> None:
    policy = PermissionPolicy(
        "guarded",
        rules=[
            {
                "tool": "send_message",
                "arguments": {"target_id": "self"},
                "risk": "L2",
                "reason": "仅本人",
            }
        ],
    )
    spec = ToolSpec(name="send_message", description="send", risk=Risk.IRREVERSIBLE)

    hit = await policy.authorize(
        spec, ToolCall(name="send_message", arguments={"target_id": "self"})
    )
    assert hit.allowed is True
    assert hit.needs_approval is False
    assert "仅本人" in hit.reason

    miss = await policy.authorize(
        spec, ToolCall(name="send_message", arguments={"target_id": "other"})
    )
    assert miss.allowed is False


async def test_policy_rule_respects_risk_ceiling() -> None:
    policy = PermissionPolicy("guarded", rules=[{"tool": "*", "risk": "L1"}])
    spec = ToolSpec(name="crm.delete", description="delete", risk=Risk.SENIOR)
    decision = await policy.authorize(spec, ToolCall(name="crm.delete"))
    assert decision.allowed is False


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


def test_example_config_enables_policy_and_planning() -> None:
    from pathlib import Path

    from doubao_agent.config import load_config

    # agent.yaml 是本地部署文件（gitignore），测试以仓库内的模板为准。
    cfg = load_config(Path(__file__).resolve().parents[1] / "agent.example.yaml")
    assert cfg.planning_enabled is True
    assert cfg.auto_approve and cfg.auto_approve[0]["tool"] == "send_message"


async def test_runtime_reload_skills_picks_up_new_skill(tmp_path) -> None:
    from doubao_agent.config import Config
    from doubao_agent.runtime import Runtime

    cfg = Config(
        workspace=tmp_path,
        runs_dir=tmp_path / "runs",
        db_path=tmp_path / "agent.db",
        skills_dir=tmp_path / "skills",
        scheduler_enabled=False,
        planning_enabled=False,
        model={"provider": "mock"},
    )
    runtime = await Runtime.build(cfg)
    try:
        assert runtime.registry.get("skill.read") is None

        skill_dir = tmp_path / "skills" / "demo"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo\ndescription: 演示技能\n---\n\n正文", "utf-8"
        )

        assert await runtime.reload_skills() == 1
        assert runtime.registry.get("skill.read") is not None
    finally:
        await runtime.aclose()


async def test_policy_rule_matches_source_prefixed_tool_name() -> None:
    policy = PermissionPolicy(
        "guarded",
        rules=[{"tool": "send_message", "arguments": {"target_id": "self"}, "risk": "L2"}],
    )
    spec = ToolSpec(name="native.send_message", description="send", risk=Risk.IRREVERSIBLE)
    decision = await policy.authorize(
        spec, ToolCall(name="native.send_message", arguments={"target_id": "self"})
    )
    assert decision.allowed is True
    assert decision.needs_approval is False


async def test_script_cancel_is_per_call(tmp_path) -> None:
    import asyncio

    from doubao_agent.sources.script import ScriptSource

    source = ScriptSource(tmp_path)
    slow_event = asyncio.Event()

    slow = asyncio.create_task(
        source.call_tool(
            ToolCall(
                name="run_python",
                arguments={"code": "import time; time.sleep(10)"},
                call_id="slow",
                cancel_event=slow_event,
            )
        )
    )
    await asyncio.sleep(0.3)

    fast = await source.call_tool(
        ToolCall(
            name="run_python",
            arguments={"code": "print(42)"},
            call_id="fast",
            cancel_event=asyncio.Event(),
        )
    )
    assert fast.ok is True
    assert "42" in fast.content

    slow_event.set()
    cancelled = await slow
    assert cancelled.ok is False
    assert cancelled.error == "cancelled"


async def test_cancel_session_is_scoped_to_target(tmp_path) -> None:
    from doubao_agent.config import Config
    from doubao_agent.runtime import Runtime

    cfg = Config(
        workspace=tmp_path,
        runs_dir=tmp_path / "runs",
        db_path=tmp_path / "agent.db",
        skills_dir=tmp_path / "skills",
        scheduler_enabled=False,
        model={"provider": "mock"},
    )
    runtime = await Runtime.build(cfg)
    try:
        result = await runtime.start_run("随便聊聊")
        assert result["status"] == "completed"
        assert runtime.running_sessions() == []

        sid_a = runtime.sessions.create("A")
        sid_b = runtime.sessions.create("B")
        agent_a = runtime.spawn_agent()
        agent_b = runtime.spawn_agent()
        runtime._runs[sid_a] = agent_a
        runtime._runs[sid_b] = agent_b

        assert runtime.cancel_session(sid_a) is True
        assert agent_a.cancel_event.is_set() is True
        assert agent_b.cancel_event.is_set() is False
        assert runtime.cancel_session("missing") is False

        assert runtime.cancel_session(sid_b) is True
        assert agent_b.cancel_event.is_set() is True
    finally:
        await runtime.aclose()
