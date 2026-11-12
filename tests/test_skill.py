from __future__ import annotations

import json

from doubao_agent.registry import CapabilityRegistry
from doubao_agent.types import ToolCall


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
