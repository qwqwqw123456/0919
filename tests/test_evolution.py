from __future__ import annotations

import asyncio
import time

from doubao_agent.core.evolution import EvolutionStore
from doubao_agent.skills import SkillManager
from doubao_agent.sources.skill import SkillSource
from doubao_agent.types import ToolCall


def _store(tmp_path, **kwargs) -> EvolutionStore:
    return EvolutionStore(tmp_path / "agent.db", **kwargs)


def test_record_counts_and_last_used(tmp_path) -> None:
    store = _store(tmp_path)
    store.record("stats", "use")
    store.record("stats", "use")
    store.record("stats", "view")
    store.record("stats", "patch")
    row = store.rows()[0]

    assert row["use_count"] == 2
    assert row["view_count"] == 1
    assert row["patch_count"] == 1
    assert row["last_used_at"] is not None
    assert row["state"] == "active"
    store.close()


def test_view_does_not_revive_but_use_does(tmp_path) -> None:
    store = _store(tmp_path)
    now = time.time()
    store.record("cold", "use", now=now - 40 * 86400)
    assert [i["to"] for i in store.audit(apply=True, now=now)] == ["archived"]

    store.record("cold", "view", now=now)
    assert store.rows()[0]["state"] == "archived"

    store.record("cold", "use", now=now)
    row = store.rows()[0]
    assert row["state"] == "active"
    assert row["archived_at"] is None
    store.close()


def test_audit_is_dry_run_by_default_and_respects_pin(tmp_path) -> None:
    store = _store(tmp_path, stale_days=14, archive_days=30)
    now = time.time()
    store.record("mid", "use", now=now - 20 * 86400)
    store.record("old", "use", now=now - 40 * 86400)
    store.record("kept", "use", now=now - 40 * 86400)
    store.pin("kept")

    plan = store.audit(now=now)
    assert {(i["name"], i["to"]) for i in plan} == {("mid", "stale"), ("old", "archived")}
    # 预演不改状态
    assert {r["name"]: r["state"] for r in store.rows()}["old"] == "active"

    applied = store.audit(apply=True, now=now)
    assert len(applied) == 2
    states = {r["name"]: r["state"] for r in store.rows()}
    assert states == {"mid": "stale", "old": "archived", "kept": "active"}
    store.close()


def test_distill_writes_inert_draft(tmp_path) -> None:
    store = _store(tmp_path)
    skills_dir = tmp_path / "skills"
    recipe = {
        "id": 7,
        "task": "统计周报字数",
        "steps": [
            {"tool": "text-stats.count", "arguments": {"text": "abc"}, "ok": True},
            {"tool": "knowledge.add", "arguments": {"title": "周报"}, "ok": False},
        ],
    }
    path = store.distill(recipe, skills_dir)
    text = path.read_text("utf-8")

    assert path.parent.parent.name == "_drafts"
    assert "配方 #7" in text
    assert "knowledge.add" in text and "需人工复核" in text
    assert "skills install" in text
    # 草案不进技能列表：扫描只认 skills_dir/*/SKILL.md
    manager = SkillManager(skills_dir)
    assert manager.refresh() == 0

    import pytest

    with pytest.raises(FileExistsError):
        store.distill(recipe, skills_dir)
    assert store.distill(recipe, skills_dir, force=True) == path
    store.close()


def _write_skill(root, name: str, meta: str, body: str) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(f"---\n{meta}\n---\n{body}", encoding="utf-8")


def test_skill_execution_is_metered(tmp_path) -> None:
    store = _store(tmp_path)
    skills_dir = tmp_path / "skills"
    _write_skill(
        skills_dir,
        "echo",
        "name: echo\ndescription: 回显输入\nentry: main.py\nruntime: python\nrisk: L1",
        "# echo",
    )
    (skills_dir / "echo" / "main.py").write_text(
        "import json,sys\nprint(json.dumps({'ok': True, 'input': sys.stdin.read()}))\n",
        encoding="utf-8",
    )
    manager = SkillManager(skills_dir, usage=store)
    manager.refresh()
    source = SkillSource(manager)

    async def run() -> None:
        result = await source.call_tool(ToolCall(name="echo", arguments={"a": 1}, call_id="c1"))
        assert result.ok
        await source.call_tool(ToolCall(name="read", arguments={"name": "echo"}, call_id="c2"))

    asyncio.run(run())
    row = store.rows()[0]
    assert row["name"] == "echo"
    assert row["use_count"] == 1
    assert row["view_count"] == 1
    store.close()


def test_meter_failure_does_not_break_skill_execution(tmp_path) -> None:
    class Boom:
        def record(self, *args, **kwargs):
            raise RuntimeError("telemetry down")

    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "echo", "name: echo\ndescription: 回显\nentry: main.py", "# echo")
    (skills_dir / "echo" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    manager = SkillManager(skills_dir, usage=Boom())
    manager.refresh()

    async def run() -> None:
        result = await SkillSource(manager)._run(
            manager.get("echo"), ToolCall(name="echo", arguments={}, call_id="c1")
        )
        assert result.ok

    asyncio.run(run())
