from __future__ import annotations

import json
from pathlib import Path

from doubao_agent.core.skill_audit import audit, format_report
from doubao_agent.core.task_eval import evaluate, prepare_workspace
from doubao_agent.skills import SkillManager


def _skill(dir_path: Path, name: str, body: str, entry: str | None = None) -> None:
    folder = dir_path / name
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", f"description: {name} 的说明", "version: 0.1"]
    if entry:
        lines += ["runtime: python", f"entry: {entry}", "risk: L0"]
    lines += ["---", "", body]
    (folder / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")


def test_skill_audit_splits_executable_and_knowledge(tmp_path: Path) -> None:
    _skill(tmp_path, "know", "纯知识")
    _skill(tmp_path, "runs", "能跑", entry="main.py")
    (tmp_path / "runs" / "main.py").write_text(
        "import json,sys\njson.load(sys.stdin)\nprint(json.dumps({'ok': True}))\n", encoding="utf-8"
    )
    _skill(tmp_path, "broken", "语法坏", entry="main.py")
    (tmp_path / "broken" / "main.py").write_text("def (:\n", encoding="utf-8")

    result = audit(SkillManager(tmp_path), smoke=False)
    names = {r["name"]: r for r in result["rows"]}
    assert result["executable"] == 2
    assert names["runs"]["static_ok"] is True
    assert names["broken"]["static_ok"] is False
    assert "静态失败" in format_report(result)


def test_skill_audit_smoke_detects_empty_output(tmp_path: Path) -> None:
    _skill(tmp_path, "silent", "没输出", entry="main.py")
    (tmp_path / "silent" / "main.py").write_text("import sys\nsys.stdin.read()\n", encoding="utf-8")
    result = audit(SkillManager(tmp_path), smoke=True)
    assert result["runnable"] == 0
    assert any("没有任何输出" in item["detail"] for item in result["smoke_bad"])


def test_evaluate_requires_file_and_answer(tmp_path: Path) -> None:
    ws = prepare_workspace(tmp_path / "ws")
    assert (ws / "seed.txt").is_file()
    case = {
        "accept": {
            "contains": ["结论"],
            "files": [{"path": "{ws}/out.md", "min_bytes": 2}],
            "max_calls": 3,
        }
    }
    result = {"status": "completed", "answer": "结论：4", "usage": {"calls": 2}}
    assert evaluate(case, result, ws)["pass"] is False  # 文件没写
    (ws / "out.md").write_text("44", encoding="utf-8")
    assert evaluate(case, result, ws)["pass"] is True


def test_evaluate_flags_incomplete_and_budget(tmp_path: Path) -> None:
    ws = prepare_workspace(tmp_path / "ws")
    case = {"accept": {"max_calls": 2}}
    result = {"status": "incomplete", "answer": "", "usage": {"calls": 5}}
    verdict = evaluate(case, result, ws)
    assert verdict["pass"] is False
    assert any("budget" in issue for issue in verdict["issues"])
    assert any("status" in issue for issue in verdict["issues"])


def test_tasks_file_covers_all_strata() -> None:
    payload = json.loads((Path(__file__).resolve().parents[1] / "eval" / "tasks.json").read_text("utf-8"))
    strata = {case["stratum"] for case in payload["cases"]}
    assert {"simple_qa", "simple_tool", "multi_step", "high_risk"} <= strata
    assert len(payload["cases"]) >= 25
