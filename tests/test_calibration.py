"""分流标定：样本门控与离线回放的口径必须可复算。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from doubao_agent.core.calibration import (
    PER_STRATUM_MIN,
    TOTAL_MIN,
    collect_runs,
    describe,
    gate,
    replay,
)


def _write_session(root: Path, session: str, route: dict, task: str = "", ts: str = "") -> None:
    folder = root / session
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    if task:
        lines.append({"ts": ts, "session": session, "event": "start", "task": task})
    lines.append({"ts": ts, "session": session, "event": "route", **route})
    (folder / "audit.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8"
    )


def test_collect_runs_takes_last_route_and_backfills_stratum(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        "s1",
        {"decision": "single", "calls": 1, "tokens": 900, "status": "completed"},
        task="1+1？",
    )
    rows = collect_runs(tmp_path)
    assert len(rows) == 1
    assert rows[0]["stratum"] == "simple_qa"
    assert rows[0]["task"] == "1+1？"


def test_stratum_uses_calls_and_risk(tmp_path: Path) -> None:
    _write_session(tmp_path, "s1", {"decision": "single", "calls": 2}, task="几个核")
    _write_session(tmp_path, "s2", {"decision": "pipeline", "calls": 4}, task="查内存并核对")
    _write_session(tmp_path, "s3", {"decision": "pipeline", "calls": 2, "risk": True}, task="删除旧文件")
    rows = collect_runs(tmp_path)
    strata = {r["session"]: r["stratum"] for r in rows}
    assert strata == {"s1": "simple_tool", "s2": "multi_step", "s3": "high_risk"}


def test_gate_ready_requires_total_and_every_stratum(tmp_path: Path) -> None:
    strata = ["simple_qa", "simple_tool", "multi_step", "high_risk", "image"]
    for i in range(TOTAL_MIN):
        _write_session(
            tmp_path,
            f"r{i}",
            {"decision": "pipeline", "stratum": strata[i % 5], "calls": 3, "status": "completed"},
            task="任务",
        )
    assert gate(collect_runs(tmp_path))["state"] == "READY"


def test_gate_time_limited_when_72h_but_not_enough(tmp_path: Path) -> None:
    old = (datetime.now(timezone.utc) - timedelta(hours=80)).isoformat()
    _write_session(tmp_path, "s1", {"decision": "single", "calls": 1}, task="hi", ts=old)
    result = gate(collect_runs(tmp_path))
    assert result["state"] == "TIME_LIMITED"
    assert result["total"] < TOTAL_MIN


def test_gate_waiting_counts_deficit(tmp_path: Path) -> None:
    _write_session(tmp_path, "s1", {"decision": "single", "calls": 1}, task="hi")
    result = gate(collect_runs(tmp_path))
    assert result["state"] == "WAITING"
    assert result["deficits"]["simple_qa"] == PER_STRATUM_MIN - 1


def test_high_risk_leak_is_zero_when_all_pipeline(tmp_path: Path) -> None:
    _write_session(tmp_path, "s1", {"decision": "pipeline", "stratum": "high_risk", "calls": 2}, task="付款")
    _write_session(tmp_path, "s2", {"decision": "single", "stratum": "high_risk", "calls": 1}, task="付款")
    assert gate(collect_runs(tmp_path))["high_risk_leak"] == 1


def test_replay_flags_false_deescalation_only_for_escalated(tmp_path: Path) -> None:
    long_task = "帮我看看这个" * 12  # >40 字，规则会判 pipeline
    _write_session(tmp_path, "s1", {"decision": "single", "escalated": True}, task=long_task)
    _write_session(tmp_path, "s2", {"decision": "single", "status": "completed"}, task=long_task)
    table = replay(collect_runs(tmp_path), [40])
    row = table[0]
    assert row["correct_upgrade"] == 1   # s1：单循环失败，回放也判流水线，判对
    assert row["false_escalation"] == 1  # s2：单循环已完成，回放却判流水线，属误升级
    assert row["false_deescalation"] == 0
    assert row["needs_shadow"] == 0


def test_replay_counts_false_escalation(tmp_path: Path) -> None:
    long_task = "把这份报告整理一下" * 8
    _write_session(tmp_path, "s1", {"decision": "single", "status": "completed"}, task=long_task)
    table = replay(collect_runs(tmp_path), [40])
    assert table[0]["false_escalation"] == 1


def test_describe_reports_median_without_faking_completion_rate(tmp_path: Path) -> None:
    _write_session(tmp_path, "s1", {"decision": "single", "calls": 1, "tokens": 5000}, task="hi")
    _write_session(tmp_path, "s2", {"decision": "single", "calls": 2, "tokens": 12000}, task="几个核")
    strata = describe(collect_runs(tmp_path))
    assert strata["simple_tool"]["tokens"]["median"] == 12000
    assert strata["simple_tool"]["completed_rate"] is None
