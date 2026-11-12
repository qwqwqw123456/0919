from __future__ import annotations

import asyncio
import os
from pathlib import Path

from conftest import _registry

from doubao_agent.core.agent import Agent, build_user_content
from doubao_agent.core.audit import AuditLog
from doubao_agent.core.memory import MemoryStore
from doubao_agent.core.permissions import PermissionPolicy
from doubao_agent.core.planner import Plan
from doubao_agent.core.session import SessionStore
from doubao_agent.core.usage import UsageMeter
from doubao_agent.providers.mock import MockProvider
from doubao_agent.server.app import _run_scheduler
from doubao_agent.skills import SkillManager
from doubao_agent.types import Risk, ToolCall, ToolResult, ToolSpec


class FlakySource:
    name = "flaky"

    def __init__(self) -> None:
        self.calls = 0

    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="write",
                description="写入",
                input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                risk=Risk.READ,
                source=self.name,
            )
        ]

    async def call_tool(self, call: ToolCall) -> ToolResult:
        self.calls += 1
        return ToolResult(call_id=call.call_id, ok=True, content="ok")

    async def health(self) -> tuple[bool, str]:
        return True, "ok"

    async def aclose(self) -> None:
        return None


async def _agent(tmp_path, registry, repair_rounds=0) -> Agent:
    db = tmp_path / "agent.db"
    return Agent(
        registry=registry,
        provider=MockProvider(script=[]),
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(db),
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(db),
        max_steps=4,
        usage=UsageMeter(),
        repair_rounds=repair_rounds,
    )


async def test_dag_skips_downstream_when_upstream_fails(tmp_path) -> None:
    source = FlakySource()
    registry = await _registry(source)
    agent = await _agent(tmp_path, registry)
    plan = Plan(
        mode="tools",
        steps=[
            {"id": "a", "tool": "missing.tool", "arguments": {}},
            {"id": "b", "tool": "flaky.write", "arguments": {"text": "x"}, "depends_on": ["a"]},
        ],
    )

    out = await agent._execute_plan("依赖任务", plan, None)

    assert out["status"] == "incomplete"
    assert "跳过" in out["answer"]
    assert source.calls == 0  # 下游没有执行


async def test_build_user_content_multipart(tmp_path) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    parts = build_user_content("看图", [str(image)])

    assert parts[0] == {"type": "text", "text": "看图"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert build_user_content("纯文本", None) == "纯文本"


def test_skill_install_and_uninstall(tmp_path) -> None:
    src = tmp_path / "src-skill"
    src.mkdir()
    (src / "SKILL.md").write_text(
        "---\nname: demo\ndescription: 演示\nversion: 0.1.0\nrisk: L0\n---\n\n正文",
        encoding="utf-8",
    )
    manager = SkillManager(tmp_path / "skills")

    skill = manager.install(src)
    assert skill.name == "demo"
    assert (tmp_path / "skills" / "demo" / "SKILL.md").is_file()

    assert manager.uninstall("demo") is True
    assert manager.uninstall("demo") is False
    assert not (tmp_path / "skills" / "demo").exists()


def test_managed_mcp_merges_from_local_file(tmp_path, monkeypatch) -> None:
    from doubao_agent.config import load_config

    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / ".agent" / "agent.db"))
    monkeypatch.delenv("DOUBAO_AGENT_CONFIG", raising=False)
    managed = tmp_path / ".agent" / "mcp.local.yaml"
    managed.parent.mkdir(parents=True, exist_ok=True)
    managed.write_text(
        "- name: host\n  transport: http\n  url: http://127.0.0.1:8091/mcp\n", encoding="utf-8"
    )

    cfg = load_config(tmp_path / "agent.yaml")
    names = {server.name for server in cfg.mcp_servers}
    assert "host" in names


def test_env_local_injects_missing_vars(tmp_path, monkeypatch) -> None:
    from doubao_agent.config import load_config

    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("AGENT_TEST_ENV_LOCAL", raising=False)
    (tmp_path / ".env.local").write_text(
        "AGENT_TEST_ENV_LOCAL=from_env_local\n# 注释行\n\n", encoding="utf-8"
    )

    load_config(tmp_path / "agent.yaml")
    assert os.environ["AGENT_TEST_ENV_LOCAL"] == "from_env_local"


def test_env_local_does_not_override_existing(tmp_path, monkeypatch) -> None:
    from doubao_agent.config import load_config

    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("AGENT_TEST_ENV_LOCAL", "already_set")
    (tmp_path / ".env.local").write_text(
        'AGENT_TEST_ENV_LOCAL="from_env_local"\n', encoding="utf-8"
    )

    load_config(tmp_path / "agent.yaml")
    assert os.environ["AGENT_TEST_ENV_LOCAL"] == "already_set"


async def test_scheduler_dispatches_concurrently() -> None:
    state = {"active": 0, "max_active": 0, "started": 0}

    class FakeScheduler:
        def __init__(self) -> None:
            self.polls = 0

        def due(self):
            self.polls += 1
            if self.polls == 1:
                return [
                    {"id": "1", "name": "a", "task": "t1"},
                    {"id": "2", "name": "b", "task": "t2"},
                ]
            return []

        def mark_ran(self, schedule_id: str) -> None:
            return None

    class FakeRuntime:
        def __init__(self) -> None:
            self.config = type("C", (), {"scheduler_interval": 0.02})()
            self.scheduler = FakeScheduler()

        async def start_run(self, task):
            state["active"] += 1
            state["started"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            await asyncio.sleep(0.05)
            state["active"] -= 1
            return {"status": "completed"}

    class FakeBus:
        def publish(self, event) -> None:
            return None

    task = asyncio.create_task(_run_scheduler(FakeRuntime(), FakeBus()))
    await asyncio.sleep(0.12)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert state["started"] == 2
    assert state["max_active"] == 2  # 两个调度并行执行，而不是串行
    assert state["active"] == 0
