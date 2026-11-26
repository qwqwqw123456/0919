#!/usr/bin/env python3
"""按审计流水汇总已用 token，给出相对总额度的余量。

用法:
    python3 scripts/token_usage.py              # 默认额度 1000 万
    python3 scripts/token_usage.py --quota 10000000
口径: 每个会话取最后一条带 usage 的 finish 事件（续跑不重复计），
      runs_dir 下的 audit.jsonl 是脱敏后的账本，不含凭据。
"""

from __future__ import annotations

import argparse
import glob
import json
import os

DEFAULT_QUOTA = 10_000_000
DEFAULT_RUNS = os.environ.get("AGENT_RUNS_DIR", "/home/user/agent-state/runs")


def collect(runs_dir: str) -> tuple[int, int, int]:
    total = calls = sessions = 0
    for path in glob.glob(os.path.join(runs_dir, "*", "audit.jsonl")):
        last = None
        for line in open(path, encoding="utf-8"):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "finish" and event.get("usage"):
                last = event["usage"]
        if last:
            sessions += 1
            total += int(last.get("total_tokens") or 0)
            calls += int(last.get("calls") or 0)
    return sessions, calls, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quota", type=int, default=DEFAULT_QUOTA, help="总额度 token")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS)
    args = parser.parse_args()

    sessions, calls, used = collect(args.runs_dir)
    left = args.quota - used
    pct = (used / args.quota * 100) if args.quota else 0
    print(f"会话 {sessions} 个 / 调用 {calls} 次")
    print(f"已用 {used:,} token，总量 {args.quota:,}，剩余 {left:,}（{pct:.1f}% 已用）")
    if left <= 0:
        print("额度已耗尽，需补充或换通道")


if __name__ == "__main__":
    main()
