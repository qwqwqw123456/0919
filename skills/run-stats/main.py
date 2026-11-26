#!/usr/bin/env python3
"""run-stats —— 任务级运行统计（迁自 v1.62 stats.py 的 win_rate 视角，改吃新引擎数据源）。

stdin: {"db_path": "...", "runs_dir": "...", "recent": 0}
stdout: {"ok": bool, "tasks": {...}, "sources": {...}}
"""
import json
import os
import sqlite3
import statistics
import sys
from pathlib import Path

HOME_STATE = Path.home() / "agent-state"


def _resolve(explicit, env_key, fallback):
    if explicit:
        return Path(str(explicit)).expanduser()
    env = os.environ.get(env_key)
    if env:
        return Path(env).expanduser()
    return fallback


def _percentile(values, pct):
    if not values:
        return 0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[idx]


def _read_audit(runs_dir: Path, session_id: str) -> list:
    path = runs_dir / session_id / "audit.jsonl"
    if not path.is_file():
        return []
    events = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def main() -> int:
    try:
        raw = sys.stdin.read().strip()
        args = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "error": f"入参不是合法 JSON: {exc}"}, ensure_ascii=False))
        return 2
    if not isinstance(args, dict):
        print(json.dumps({"ok": False, "error": "入参必须是 JSON 对象"}, ensure_ascii=False))
        return 2

    db_path = _resolve(args.get("db_path"), "AGENT_DB_PATH", HOME_STATE / ".agent" / "agent.db")
    runs_dir = _resolve(args.get("runs_dir"), "AGENT_RUNS_DIR", HOME_STATE / "runs")
    recent = args.get("recent") or 0

    sources = {
        "db_path": str(db_path),
        "db_exists": db_path.is_file(),
        "runs_dir": str(runs_dir),
        "runs_dir_exists": runs_dir.is_dir(),
    }
    if not db_path.is_file():
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "会话库不存在，未跑过任何任务或路径不对",
                    "sources": sources,
                },
                ensure_ascii=False,
            )
        )
        return 1

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT id, task, status, step, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()]
    except sqlite3.Error as exc:
        print(json.dumps({"ok": False, "error": f"读库失败: {exc}", "sources": sources}, ensure_ascii=False))
        return 1
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    if recent and isinstance(recent, int):
        rows = rows[:recent]

    by_status = {}
    durations, steps = [], []
    tool_calls = 0
    degraded_sessions = 0
    for row in rows:
        status = str(row.get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        if row.get("step"):
            steps.append(int(row["step"]))
        if row.get("created_at") and row.get("updated_at"):
            durations.append(max(0, float(row["updated_at"]) - float(row["created_at"])))
        events = _read_audit(runs_dir, str(row["id"]))
        calls = sum(1 for e in events if e.get("event") == "tool_call")
        tool_calls += calls
        audit_step = max([int(e.get("step") or 0) for e in events] + [0])
        if audit_step:
            steps.append(audit_step)
        if any(e.get("event") == "degraded" for e in events):
            degraded_sessions += 1

    completed = by_status.get("completed", 0)
    failed = by_status.get("failed", 0)
    finished = completed + failed

    print(
        json.dumps(
            {
                "ok": True,
                "tasks": {
                    "total": len(rows),
                    "by_status": by_status,
                    "success_rate": round(completed / finished, 4) if finished else None,
                    "success_note": "分母=completed+failed；running 不计入",
                    "avg_steps": round(statistics.fmean(steps), 2) if steps else 0,
                    "duration_p50_sec": round(_percentile(durations, 50), 1),
                    "duration_p95_sec": round(_percentile(durations, 95), 1),
                    "degraded_sessions": degraded_sessions,
                    "tool_calls": tool_calls,
                    "tool_calls_per_task": round(tool_calls / len(rows), 2) if rows else 0,
                },
                "sources": sources,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
