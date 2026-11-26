"""分流标定：样本门控 + 离线回放候选阈值。

门控口径（先到先做）：
- 累计 route 日志 ≥ 200 条，且五个分层每类 ≥ 30 条 → READY，可调阈值
- 跑满 72 小时但样本不足 → TIME_LIMITED，只出描述统计，不调阈值
- 都不满足 → WAITING，报告缺口

回放只用本地日志，不消耗 token、不占流量时间。
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from doubao_agent.core.router import (
    STRATA,
    STRATUM_LABELS,
    decide_route,
    stratum_of,
)

TOTAL_MIN = 200
PER_STRATUM_MIN = 30
WINDOW_HOURS = 72
CANDIDATE_CHARS = (25, 40, 55, 70)
DEFAULT_RUNS = Path("/home/user/agent-state/runs")


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def collect_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """每个会话取最后一条 route 事件；任务原文优先用 route 里的采样，回落到 start。"""
    rows: list[dict[str, Any]] = []
    for audit in sorted(runs_dir.glob("*/audit.jsonl")):
        route = None
        start_task = ""
        try:
            with audit.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    kind = event.get("event")
                    if kind == "start" and not start_task:
                        start_task = event.get("task") or ""
                    elif kind == "route":
                        route = event
        except (OSError, json.JSONDecodeError):
            continue
        if not route:
            continue
        session = route.get("session") or audit.parent.name
        row = dict(route)
        row["session"] = session
        row.setdefault("ts", "")
        row["task"] = row.get("task") or start_task
        if not row.get("stratum"):
            # 老日志没分层字段：按已有字段回溯，保证统计口径连续
            row["stratum"] = stratum_of(row, row.get("calls"), False)
        rows.append(row)
    rows.sort(key=lambda r: str(r.get("ts") or ""))
    return rows


def _stats(values: Iterable[float]) -> dict[str, float]:
    data = [float(v) for v in values if isinstance(v, (int, float))]
    if not data:
        return {"n": 0}
    return {
        "n": len(data),
        "median": statistics.median(data),
        "mean": round(statistics.fmean(data), 1),
        "max": max(data),
    }


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strata: dict[str, dict[str, Any]] = {}
    for key in STRATA:
        subset = [r for r in rows if (r.get("stratum") or "") == key]
        known = [r for r in subset if r.get("status")]
        completed = sum(1 for r in known if r.get("status") == "completed")
        strata[key] = {
            "label": STRATUM_LABELS[key],
            "count": len(subset),
            "completed": completed,
            "completed_rate": round(completed / len(known), 3) if known else None,
            "escalated": sum(1 for r in subset if r.get("escalated")),
            "calls": _stats(r.get("calls") for r in subset),
            "tokens": _stats(r.get("tokens") for r in subset),
        }
    return strata


def gate(rows: list[dict[str, Any]], now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    total = len(rows)
    counts = {key: sum(1 for r in rows if (r.get("stratum") or "") == key) for key in STRATA}
    deficits = {k: max(0, PER_STRATUM_MIN - v) for k, v in counts.items()}

    first = next((_parse_ts(r.get("ts")) for r in rows if _parse_ts(r.get("ts"))), None)
    span_hours = (now - first).total_seconds() / 3600 if first else 0.0

    high_risk_leak = sum(
        1 for r in rows if (r.get("stratum") or "") == "high_risk" and (r.get("decision") != "pipeline")
    )

    if total >= TOTAL_MIN and all(v >= PER_STRATUM_MIN for v in counts.values()):
        state = "READY"
        note = "样本够，可做离线回放并调阈值"
    elif first and span_hours >= WINDOW_HOURS:
        state = "TIME_LIMITED"
        note = "已跑满 72 小时但样本不足，只出描述统计，不调阈值"
    else:
        state = "WAITING"
        note = "样本增加中，继续收流量"

    return {
        "state": state,
        "note": note,
        "total": total,
        "total_min": TOTAL_MIN,
        "per_stratum_min": PER_STRATUM_MIN,
        "counts": counts,
        "deficits": deficits,
        "high_risk_leak": high_risk_leak,
        "span_hours": round(span_hours, 1),
        "window_hours": WINDOW_HOURS,
        "remaining_hours": round(max(0.0, WINDOW_HOURS - span_hours), 1) if first else WINDOW_HOURS,
    }


def replay(rows: list[dict[str, Any]], candidates: Iterable[int] = CANDIDATE_CHARS) -> list[dict[str, Any]]:
    """用记录下来的任务原文，按候选阈值重跑规则，和实际结果对账。

    口径：
    - actual single 且 completed，新规则判 pipeline → 误升级（浪费核验成本）
    - actual single 且 escalated（说明单循环确实扛不住），新规则判 pipeline → 该升（正确）
    - 同上但新规则仍判 single → 误降级（危险）
    - actual pipeline，新规则判 single → 待影子验证（本日志不足以判定）
    """
    usable = [r for r in rows if (r.get("task") or "").strip()]
    table: list[dict[str, Any]] = []
    for chars in candidates:
        false_escalation = 0
        false_deescalation = 0
        correct_upgrade = 0
        shadow = 0
        agree = 0
        for row in usable:
            new_route = decide_route(row["task"], max_simple_chars=chars)["route"]
            actual = "pipeline" if (row.get("decision") == "pipeline") else "single"
            if new_route == actual:
                agree += 1
            elif new_route == "pipeline":
                if row.get("status") == "completed":
                    false_escalation += 1
                elif row.get("escalated"):
                    correct_upgrade += 1
            elif row.get("escalated"):
                false_deescalation += 1
            else:
                shadow += 1
        table.append(
            {
                "max_simple_chars": chars,
                "usable": len(usable),
                "agree": agree,
                "false_escalation": false_escalation,
                "false_deescalation": false_deescalation,
                "correct_upgrade": correct_upgrade,
                "needs_shadow": shadow,
            }
        )
    return table


def _print_report(payload: dict[str, Any]) -> None:
    g = payload["gate"]
    print(f"标定门控：{g['state']} — {g['note']}")
    print(f"高风险漏判：{g['high_risk_leak']} 条（必须为 0）")
    print(
        f"样本：{g['total']}/{g['total_min']} 条｜已跨 {g['span_hours']}h / {g['window_hours']}h"
        f"（距时间窗结束 {g['remaining_hours']}h）"
    )
    for key in STRATA:
        s = payload["strata"][key]
        flag = "OK" if s["count"] >= PER_STRATUM_MIN else f"缺 {g['deficits'][key]}"
        token_med = s["tokens"].get("median", "-")
        call_med = s["calls"].get("median", "-")
        print(
            f"  {s['label']:<6} {s['count']:>3} 条（{flag}）"
            f"｜完成率 {s['completed_rate']}｜升级 {s['escalated']}"
            f"｜中位 {call_med} 调用 / {token_med} token"
        )
    if payload.get("replay"):
        print("\n离线回放（候选入口阈值）：")
        print("  阈值  可用  一致  该升  误升  误降  待影子")
        for row in payload["replay"]:
            print(
                f"  {row['max_simple_chars']:>3}字 {row['usable']:>5} {row['agree']:>5}"
                f" {row['correct_upgrade']:>5} {row['false_escalation']:>5}"
                f" {row['false_deescalation']:>5} {row['needs_shadow']:>6}"
            )
        print("  说明：误降级只统计“实际单循环已升级”的样本，最危险；"
              "待影子需真跑一次才能判定。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分流标定：样本门控 + 离线回放")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS), help="runs 目录")
    parser.add_argument("--replay", action="store_true", help="跑候选阈值回放")
    parser.add_argument("--chars", default=None, help="候选字符阈值，逗号分隔，例 25,40,55")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    rows = collect_runs(Path(args.runs_dir))
    payload: dict[str, Any] = {
        "gate": gate(rows),
        "strata": describe(rows),
        "total_rows": len(rows),
    }
    if args.replay:
        candidates = (
            [int(x) for x in args.chars.split(",") if x.strip()] if args.chars else list(CANDIDATE_CHARS)
        )
        payload["replay"] = replay(rows, candidates)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
