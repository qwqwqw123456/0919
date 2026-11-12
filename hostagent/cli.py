"""统一 CLI：契约 / 规划 / 执行 / 恢复 / 治理 / 记忆 / Skill / 审计。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .contract import SOVEREIGN_SKILLS
from .executor import Executor
from .gateway import HostRuntimeGateway
from .governance import Gate, Ledger, PolicyStore
from .memory import MemoryStore
from .models import Plan
from .planner import ContractError, describe_contract, plan_by_rules, plan_from_spec, validate
from .skills import QuotaStore, SkillError, SovereignSkillClient, route_task
from .state import TaskStore

DATA_DIR = Path(os.environ.get("HOSTAGENT_HOME", ".hostagent"))
OUTPUT_DIR = Path(os.environ.get("HOSTAGENT_OUTPUT", "output"))


def _stores() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "memory": MemoryStore(DATA_DIR / "memory.db"),
        "policy": PolicyStore(DATA_DIR / "policy.db"),
        "ledger": Ledger(DATA_DIR / "ledger.jsonl"),
        "tasks": TaskStore(DATA_DIR / "tasks.db"),
        "quota": QuotaStore(DATA_DIR / "quota.db"),
    }


def _executor(stores: dict[str, Any], *, dry_run: bool, mode: str = "auto") -> Executor:
    return Executor(
        memory=stores["memory"],
        gateway=HostRuntimeGateway(),
        gate=Gate(stores["policy"], stores["ledger"], mode=mode),
        ledger=stores["ledger"],
        output_dir=OUTPUT_DIR,
        dry_run=dry_run,
    )


def _print_report(report: dict[str, Any]) -> None:
    for node_id, outcome in report["nodes"].items():
        status = outcome["status"]
        mark = {"completed": "OK", "failed": "FAIL", "denied": "DENY",
                "blocked": "SKIP", "unavailable": "HOST"}.get(status, status)
        verdict = outcome.get("verdict") or ""
        line = f"  {node_id:<16} {mark:<5} {verdict}"
        if outcome.get("error"):
            line += f"  {outcome['error'][:120]}"
        if status == "completed" and outcome.get("output") is not None:
            preview = outcome["output"]
            if not isinstance(preview, str):
                preview = json.dumps(preview, ensure_ascii=False)
            line += f"  {preview[:100].replace(chr(10), ' ')}"
        print(line)
    print(f"状态: {report['status']}  {report['completed']}/{report['total']}")
    for artifact in report.get("artifacts") or []:
        print(f"产物: {artifact}")
    if report.get("final_text"):
        print("结果:\n" + str(report["final_text"])[:2000])


def _cmd_contract(args: argparse.Namespace) -> int:
    rows = describe_contract()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    print(f"契约节点 {len(rows)} 个（本地子 Agent 13 + 宿主连接器 8）")
    for row in rows:
        print(f"  {row['tool']:<16} {row['delegate']:<17} {row['action']:<32} "
              f"{row['risk']:<6} {row['approval']}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    try:
        plan = plan_from_spec(json.loads(Path(args.spec).read_text(encoding="utf-8"))) \
            if args.spec else plan_by_rules(args.intent or "")
    except (ContractError, json.JSONDecodeError, OSError) as exc:
        print(f"规划失败: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    stores = _stores()
    task_id = None
    if args.resume:
        task = stores["tasks"].get(args.resume)
        if task is None:
            print(f"任务不存在: {args.resume}", file=sys.stderr)
            return 2
        plan = Plan.from_dict(task["plan"])
        task_id = task["id"]
    else:
        try:
            plan = plan_from_spec(json.loads(Path(args.spec).read_text(encoding="utf-8"))) \
                if args.spec else plan_by_rules(args.intent or "")
        except (ContractError, json.JSONDecodeError, OSError) as exc:
            print(f"规划失败: {exc}", file=sys.stderr)
            return 2
        task_id = stores["tasks"].create(plan.intent, plan.to_dict())

    executor = _executor(stores, dry_run=args.dry_run, mode=args.mode)
    report = executor.run(plan)
    stores["tasks"].update(task_id, report["status"], report)
    if args.json:
        print(json.dumps({"task_id": task_id, **report}, ensure_ascii=False, indent=2))
    else:
        print(f"任务 {task_id}  intent: {plan.intent}")
        _print_report(report)
    return 0 if report["status"] == "completed" else 1


def _cmd_tasks(args: argparse.Namespace) -> int:
    stores = _stores()
    if args.action == "show":
        task = stores["tasks"].get(args.task_id)
        if task is None:
            print("任务不存在", file=sys.stderr)
            return 2
        print(json.dumps(task, ensure_ascii=False, indent=2))
        return 0
    rows = stores["tasks"].list(args.limit)
    if not rows:
        print("（无任务）")
        return 0
    for row in rows:
        print(f"{row['id']}  {row['status']:<10} {row['intent'][:50]}")
    return 0


def _cmd_policy(args: argparse.Namespace) -> int:
    stores = _stores()
    policy: PolicyStore = stores["policy"]
    if args.action == "add":
        policy_id = policy.add(args.tool, args.dimension, args.value, args.operator)
        print(f"已添加预授权规则 #{policy_id}: {args.tool} {args.dimension}={args.value}")
    elif args.action == "remove":
        print("已删除" if policy.remove(args.policy_id) else "规则不存在")
    elif args.action in ("enable", "disable"):
        ok = policy.set_enabled(args.policy_id, args.action == "enable")
        print("已更新" if ok else "规则不存在")
    else:
        rows = policy.list(args.tool)
        if not rows:
            print("（没有预授权规则）")
        for row in rows:
            flag = "on" if row["enabled"] else "off"
            print(f"#{row['id']}  {flag:<3} {row['tool']:<16} {row['dimension']}={row['value']}")
    return 0


def _cmd_memory(args: argparse.Namespace) -> int:
    stores = _stores()
    memory: MemoryStore = stores["memory"]
    if args.action == "add":
        print(f"已记住 #{memory.add(args.text, args.tag)}")
    elif args.action == "search":
        hits = memory.query(args.query, args.k)
        for hit in hits:
            print(f"#{hit['id']} [{hit['tag']}] {hit['text']}")
        if not hits:
            print("（无命中）")
    else:
        for row in memory.all():
            print(f"#{row['id']} [{row['tag']}] {row['text'][:80]}")
    return 0


def _cmd_skills(args: argparse.Namespace) -> int:
    stores = _stores()
    quota: QuotaStore = stores["quota"]
    client = SovereignSkillClient(quota=quota)
    if args.action == "invoke":
        try:
            print(json.dumps(client.invoke(args.skill, args.task, args.attachment),
                             ensure_ascii=False, indent=2))
        except SkillError as exc:
            print(f"调用失败: {exc}", file=sys.stderr)
            return 2
    elif args.action == "quota-apply":
        try:
            print(json.dumps(client.quota_apply(args.skill, args.reason), ensure_ascii=False, indent=2))
        except SkillError as exc:
            print(f"申请失败: {exc}", file=sys.stderr)
            return 2
    elif args.action == "route":
        name = route_task(args.task)
        print(name or "无此能力授权")
        return 0 if name else 1
    else:
        for row in quota.all():
            limit = row["limit"] if row["limit"] > 0 else "未设限"
            print(f"{row['skill']:<28} 今日 {row['used']} / {limit}")
        if not client.available:
            print("（sovereign-skill 未安装：invoke 需在云端 TEE 运行）")
    return 0


def _cmd_ledger(args: argparse.Namespace) -> int:
    stores = _stores()
    ledger: Ledger = stores["ledger"]
    if args.action == "verify":
        ok, count = ledger.verify()
        print(f"哈希链{'完好' if ok else '已损坏'}，共 {count} 条")
        return 0 if ok else 1
    for record in ledger.read(args.limit):
        print(json.dumps(record, ensure_ascii=False))
    return 0


def _cmd_gateway(args: argparse.Namespace) -> int:
    gateway = HostRuntimeGateway()
    print(f"HOST_RUNTIME_URL: {gateway.url or '(未配置)'}")
    print("状态: " + ("可用" if gateway.available else "不可用（external_runtime 节点会如实报 unavailable）"))
    return 0 if gateway.available else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hostagent", description="宿主契约驱动的 Agent 运行时")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("contract", help="查看契约节点")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_contract)

    p = sub.add_parser("plan", help="按规则生成计划")
    p.add_argument("intent", nargs="?", default="")
    p.add_argument("--spec", help="显式 spec JSON 文件")
    p.set_defaults(func=_cmd_plan)

    p = sub.add_parser("run", help="规划并执行")
    p.add_argument("intent", nargs="?", default="")
    p.add_argument("--spec", help="显式 spec JSON 文件")
    p.add_argument("--resume", help="恢复指定任务")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mode", choices=["auto", "strict"], default="auto")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("tasks", help="任务列表 / 详情")
    p.add_argument("action", nargs="?", choices=["list", "show"], default="list")
    p.add_argument("task_id", nargs="?", default="")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=_cmd_tasks)

    p = sub.add_parser("policy", help="预授权白名单")
    p.add_argument("action", nargs="?", choices=["list", "add", "remove", "enable", "disable"],
                   default="list")
    p.add_argument("--tool")
    p.add_argument("--dimension", default="any")
    p.add_argument("--value", default="*")
    p.add_argument("--operator")
    p.add_argument("--policy-id", type=int, dest="policy_id", default=0)
    p.set_defaults(func=_cmd_policy)

    p = sub.add_parser("memory", help="本地记忆")
    p.add_argument("action", nargs="?", choices=["list", "add", "search"], default="list")
    p.add_argument("text", nargs="?", default="")
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--tag", default="fact")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=_cmd_memory)

    p = sub.add_parser("skills", help="主权 Skill")
    p.add_argument("action", nargs="?", choices=["list", "invoke", "quota-apply", "route"],
                   default="list")
    p.add_argument("--skill", choices=list(SOVEREIGN_SKILLS))
    p.add_argument("--task", default="")
    p.add_argument("--attachment")
    p.add_argument("--reason", default="")
    p.set_defaults(func=_cmd_skills)

    p = sub.add_parser("ledger", help="审计账本")
    p.add_argument("action", nargs="?", choices=["show", "verify"], default="show")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=_cmd_ledger)

    p = sub.add_parser("gateway", help="宿主运行时状态")
    p.set_defaults(func=_cmd_gateway)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
