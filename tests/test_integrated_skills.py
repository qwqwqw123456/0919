"""整合技能回归：output-density / run-stats（迁自 v1.62 体系）。

两个技能走 skill 源统一契约：stdin JSON → stdout JSON。
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"


def run_skill(name: str, payload: dict) -> dict:
    entry = SKILLS / name / "main.py"
    proc = subprocess.run(
        [sys.executable, str(entry)],
        input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        timeout=30,
    )
    out = proc.stdout.decode("utf-8").strip()
    assert out, f"{name} 没有输出，stderr={proc.stderr.decode('utf-8', 'ignore')[:200]}"
    return json.loads(out)


def test_output_density_flags_skeleton():
    data = run_skill("output-density", {"text": "待补充的内容。TODO 这里。TBD。XXX 待定。"})
    assert data["ok"] is False
    assert data["counts"]["placeholders"] >= 3
    assert any("占位符" in w for w in data["warnings"])


def test_output_density_passes_concrete_text():
    body = (
        "- 版本 0.3.0，258 个用例通过\n"
        "- 覆盖 208 个技能，装配率 100%\n"
        "- 参考 https://github.com/qwqwqw123456/0919\n"
        "- 参考 https://example.com/a\n"
        "- 耗时 18.9 秒，成本 0 元\n"
    ) * 3
    data = run_skill("output-density", {"text": body})
    assert data["ok"] is True, data["warnings"]
    assert data["counts"]["digits"] >= 5
    assert data["counts"]["urls"] >= 2


def test_output_density_reads_file(tmp_path):
    target = tmp_path / "draft.md"
    target.write_text("# 草稿\n\nTODO: 待定\n", encoding="utf-8")
    data = run_skill("output-density", {"path": str(target)})
    assert data["source"] == "path"
    assert data["ok"] is False


def test_output_density_missing_file(tmp_path):
    data = run_skill("output-density", {"path": str(tmp_path / "nope.md")})
    assert data["ok"] is False
    assert "不存在" in data["error"]


def _make_state(tmp_path: Path) -> tuple[Path, Path]:
    db = tmp_path / ".agent" / "agent.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, task TEXT, status TEXT, result TEXT,
            step INTEGER DEFAULT 0, created_at REAL, updated_at REAL
        );
        """
    )
    rows = [
        ("s1", "任务一", "completed", None, 2, 100.0, 106.0),
        ("s2", "任务二", "completed", None, 3, 100.0, 118.0),
        ("s3", "任务三", "failed", None, 1, 100.0, 104.0),
        ("s4", "任务四", "running", None, 0, 100.0, 160.0),
    ]
    conn.executemany("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    runs = tmp_path / "runs"
    (runs / "s1").mkdir(parents=True)
    events = [
        {"event": "start", "step": 0},
        {"event": "tool_call", "tool": "skill.read", "ok": True, "step": 1},
        {"event": "tool_call", "tool": "system.status", "ok": True, "step": 2},
        {"event": "finish", "step": 2},
    ]
    (runs / "s1" / "audit.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8"
    )
    (runs / "s3").mkdir(parents=True)
    (runs / "s3" / "audit.jsonl").write_text(
        json.dumps({"event": "degraded", "reason": "model down"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return db, runs


def test_run_stats_rolls_up_task_level(tmp_path):
    db, runs = _make_state(tmp_path)
    data = run_skill("run-stats", {"db_path": str(db), "runs_dir": str(runs)})
    assert data["ok"] is True
    tasks = data["tasks"]
    assert tasks["total"] == 4
    assert tasks["by_status"] == {"completed": 2, "failed": 1, "running": 1}
    # 成功率分母只算 completed+failed，running 不计入
    assert tasks["success_rate"] == round(2 / 3, 4)
    assert tasks["tool_calls"] == 2
    assert tasks["degraded_sessions"] == 1


def test_run_stats_reports_missing_db(tmp_path):
    data = run_skill("run-stats", {"db_path": str(tmp_path / "none.db"), "runs_dir": str(tmp_path)})
    assert data["ok"] is False
    assert data["sources"]["db_exists"] is False
