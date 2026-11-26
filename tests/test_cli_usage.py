from __future__ import annotations

import json

from doubao_agent.cli import _collect_run_usage, main


def _write_config(tmp_path) -> str:
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        f"workspace: {tmp_path}\nruns_dir: {tmp_path / 'runs'}\n"
        f"db_path: {tmp_path / '.agent' / 'agent.db'}\nport: 8767\n"
        "mcp_auto_discover: false\nscheduler_enabled: false\n",
        encoding="utf-8",
    )
    return str(cfg_path)


def _audit(path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def test_collect_run_usage_takes_last_finish_per_session(tmp_path) -> None:
    runs = tmp_path / "runs"
    _audit(
        runs / "s1" / "audit.jsonl",
        [
            {"event": "start", "task": "写周报"},
            {"event": "finish", "usage": {"calls": 3, "total_tokens": 1200}},
            {"event": "finish", "usage": {"calls": 5, "total_tokens": 2000}},
        ],
    )
    _audit(
        runs / "s2" / "audit.jsonl",
        [
            {"event": "start", "task": "查资料"},
            {"event": "finish", "usage": {"calls": 2, "total_tokens": 800}},
        ],
    )

    rows = _collect_run_usage(str(runs))
    by_session = {row["session"]: row for row in rows}
    assert by_session["s1"]["tokens"] == 2000  # 续跑取最后一条，不重复累加
    assert by_session["s1"]["task"] == "写周报"
    assert by_session["s2"]["tokens"] == 800
    assert sum(row["tokens"] for row in rows) == 2800


def test_cli_usage_reports_quota(tmp_path, capsys) -> None:
    cfg_path = _write_config(tmp_path)
    _audit(
        tmp_path / "runs" / "s1" / "audit.jsonl",
        [
            {"event": "start", "task": "体检"},
            {"event": "finish", "usage": {"calls": 4, "total_tokens": 500_000}},
        ],
    )

    assert main(["--config", cfg_path, "usage", "--quota", "1000000", "--detail"]) == 0
    out = capsys.readouterr().out
    assert "已用 500,000" in out
    assert "剩余 500,000" in out
    assert "50.0%" in out


def test_evolve_status_counts_installed_skills(tmp_path, capsys) -> None:
    cfg_path = _write_config(tmp_path)
    skills = tmp_path / "skills"
    for name in ("alpha", "beta"):
        folder = skills / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: 测试技能\n---\n\n正文\n", encoding="utf-8"
        )

    assert main(["--config", cfg_path, "evolve", "status"]) == 0
    out = capsys.readouterr().out
    assert "已装技能 2 个" in out
    assert "有用量记账 0 个" in out
