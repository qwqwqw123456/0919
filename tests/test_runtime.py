from __future__ import annotations


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


def test_gap_config_fields_parse(tmp_path) -> None:
    from doubao_agent.config import load_config

    cfg_file = tmp_path / "agent.yaml"
    cfg_file.write_text(
        """
workspace: /tmp/ws
tool_channel_concurrency: 5
context_model_summary: true
model_fallback:
  provider: openai
  base_url: https://backup.example/v1
  model: backup-model
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.tool_channel_concurrency == 5
    assert cfg.context_model_summary is True
    assert cfg.model_fallback["model"] == "backup-model"
