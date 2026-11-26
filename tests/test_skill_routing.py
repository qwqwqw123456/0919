from __future__ import annotations

from conftest import FakeSource, _registry

from doubao_agent.core.agent import Agent
from doubao_agent.core.audit import AuditLog
from doubao_agent.core.memory import MemoryStore
from doubao_agent.core.permissions import PermissionPolicy
from doubao_agent.core.session import SessionStore
from doubao_agent.providers.mock import MockProvider
from doubao_agent.skills import SkillManager


def _write_skill(root, name: str, description: str) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n正文示例\n", encoding="utf-8"
    )


async def _agent_with_skills(tmp_path, skills: SkillManager) -> Agent:
    registry = await _registry(FakeSource())
    return Agent(
        registry=registry,
        provider=MockProvider(script=[{"content": "完成", "tool_calls": []}]),
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(tmp_path / "agent.db"),
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(tmp_path / "agent.db"),
        max_steps=4,
        skills=skills,
    )


async def test_skill_hint_injected_when_matched(tmp_path) -> None:
    root = tmp_path / "skills"
    _write_skill(root, "weekly-report", "写周报时按五段式结构汇总进展与风险")
    _write_skill(root, "unrelated", "完全无关的发票开具说明")
    manager = SkillManager(root)
    manager.refresh()
    agent = await _agent_with_skills(tmp_path, manager)

    prompt = agent._system_prompt_for("帮我写周报，汇总本周进展")
    assert "weekly-report" in prompt
    assert "skill.read" in prompt
    assert "unrelated" not in prompt


async def test_skill_hint_absent_when_no_match(tmp_path) -> None:
    root = tmp_path / "skills"
    _write_skill(root, "weekly-report", "写周报时按五段式结构汇总进展与风险")
    manager = SkillManager(root)
    manager.refresh()
    agent = await _agent_with_skills(tmp_path, manager)

    prompt = agent._system_prompt_for("1+1 等于几")
    assert "可能用到的技能" not in prompt
