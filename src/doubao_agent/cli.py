from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any

from .config import Config, load_config
from .core.audit import AuditLog
from .core.memory import KINDS as MEMORY_KINDS
from .runtime import Runtime
from .types import ToolCall, ToolSpec


def _make_approver():
    async def approve(spec: ToolSpec, call: ToolCall, reason: str) -> bool:
        if not sys.stdin.isatty():
            print(f"[权限门] 拒绝 {spec.name}（{reason}）：非交互环境无审批人", file=sys.stderr)
            return False
        prompt = f"\n[权限门] {reason}\n  工具: {spec.name}\n  参数: {call.arguments}\n允许执行? [y/N] "
        answer = await asyncio.to_thread(input, prompt)
        return answer.strip().lower() in ("y", "yes")

    return approve


def _print_event(event: dict[str, Any]) -> None:
    kind = event.get("type")
    if kind == "tool_start":
        print(f"  → 调用 {event['tool']}", file=sys.stderr)
    elif kind == "tool_end":
        mark = "ok" if event.get("ok") else f"fail: {event.get('error')}"
        print(f"  ← {event['tool']} {mark}", file=sys.stderr)
    elif kind == "subtasks":
        print(f"\n拆分为 {event.get('count')} 个子任务：", file=sys.stderr)
        for index, subtask in enumerate(event.get("subtasks", []), 1):
            print(f"  {index}. {subtask}", file=sys.stderr)
    elif kind == "finish":
        print("\n" + str(event.get("answer", "")), file=sys.stderr)


def _usage_line(result: dict[str, Any]) -> str:
    usage = result.get("usage") or {}
    return f"调用={usage.get('calls', 0)} tokens={usage.get('total_tokens', 0)}"


def _audit_command(args: argparse.Namespace, cfg: Config) -> int:
    log = AuditLog(cfg.runs_dir)
    if args.action == "sessions":
        ids = log.sessions()
        if not ids:
            print("（暂无审计记录）")
            return 0
        for session_id in ids:
            print(session_id)
        return 0

    targets = [args.session] if args.session else log.sessions()
    if not targets:
        print("（暂无审计记录）")
        return 0
    failed = 0
    for session_id in targets:
        ok, count = log.verify(session_id)
        print(f"{'OK  ' if ok else 'FAIL'}\t{session_id}\t{count} 条")
        failed += 0 if ok else 1
    return 1 if failed else 0


def _daemon_command(args: argparse.Namespace, cfg: Config) -> int:
    from . import daemon

    pid_file, log_file = daemon.paths(cfg.workspace)
    port = args.port or cfg.port

    if args.action == "start":
        pid = daemon.read_pid(pid_file)
        if pid and daemon.alive(pid):
            print(f"守护进程已在运行 pid={pid} port={port}")
            return 0
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "doubao_agent.daemon",
                "--config",
                args.config or "/workspace/agent.yaml",
                "--port",
                str(port),
                "--workspace",
                str(cfg.workspace),
            ],
            cwd=str(cfg.workspace),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(40):
            if daemon.health(port):
                print(f"已启动 port={port} pid={daemon.read_pid(pid_file)} 日志={log_file}")
                return 0
            time.sleep(0.5)
        print(f"启动超时，请查看日志：{log_file}", file=sys.stderr)
        return 1

    if args.action == "stop":
        pid = daemon.read_pid(pid_file)
        if not pid or not daemon.alive(pid):
            print("守护进程未在运行")
            return 0
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            if not daemon.alive(pid):
                print("已停止")
                return 0
            time.sleep(0.5)
        print("停止超时，进程仍在运行", file=sys.stderr)
        return 1

    pid = daemon.read_pid(pid_file)
    print(
        json.dumps(
            {
                "pid": pid,
                "alive": bool(pid and daemon.alive(pid)),
                "health": daemon.health(port),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _memory_command(memory, args: argparse.Namespace) -> int:
    if args.action == "list":
        rows = memory.recent(args.limit)
        if not rows:
            print("还没有记忆")
            return 0
        for row in rows:
            print(f"#{row['id']}\t{row['kind']}\tw={row['weight']:.1f}\t{row['text']}")
        return 0

    if args.action == "add":
        if not args.value:
            print('用法: memory add --kind preference "内容"', file=sys.stderr)
            return 2
        memory_id = memory.remember(args.value, kind=args.kind)
        print(f"已记住 #{memory_id} [{args.kind}]")
        return 0

    if args.action == "search":
        rows = memory.recall(args.value or "", limit=args.limit)
        if not rows:
            print("没有匹配的记忆")
            return 0
        for row in rows:
            print(f"#{row['id']}\t{row['kind']}\tw={row['weight']:.1f}\t{row['text']}")
        return 0

    if args.action == "forget":
        if not args.value or not args.value.lstrip("#").isdigit():
            print("用法: memory forget <id>", file=sys.stderr)
            return 2
        forgotten = memory.forget(int(args.value.lstrip("#")))
        print("已遗忘" if forgotten else "未找到该记忆")
        return 0 if forgotten else 1

    stats = memory.stats()
    print(f"总计 {stats['total']} 条")
    for kind, count in stats["kinds"].items():
        print(f"  {kind}\t{count}")
    return 0


def _fmt_time(timestamp: float | None) -> str:
    return time.strftime("%F %T", time.localtime(timestamp)) if timestamp else "-"


async def _schedule_command(runtime: Runtime, args: argparse.Namespace) -> int:
    scheduler = runtime.scheduler
    if args.action == "list":
        rows = scheduler.list()
        if not rows:
            print("还没有定时任务，用 `schedule add` 创建")
            return 0
        for row in rows:
            enabled = "启用" if row["enabled"] else "停用"
            print(
                f"#{row['id']}\t{enabled}\t{row['cron']}\t{row['name']}\t"
                f"下次={_fmt_time(row.get('next_run'))}\t{row['task'][:40]}"
            )
        return 0

    if args.action == "add":
        if len(args.args) < 2:
            print('用法: schedule add "cron" "任务描述" [--name 名称]', file=sys.stderr)
            return 2
        cron, task = args.args[0], " ".join(args.args[1:])
        try:
            schedule_id = scheduler.add(args.name or "schedule", cron, task)
        except ValueError as exc:
            print(f"cron 表达式无效：{exc}", file=sys.stderr)
            return 2
        row = scheduler.get(schedule_id)
        print(f"已添加 #{schedule_id}，下次执行 {_fmt_time(row['next_run'])}")
        return 0

    if args.action in ("remove", "enable", "disable", "run"):
        if not args.args or not args.args[0].lstrip("#").isdigit():
            print(f"用法: schedule {args.action} <id>", file=sys.stderr)
            return 2
        schedule_id = int(args.args[0].lstrip("#"))

        if args.action == "remove":
            removed = scheduler.remove(schedule_id)
            print("已删除" if removed else "未找到该任务")
            return 0 if removed else 1
        if args.action == "enable":
            print("已启用" if scheduler.set_enabled(schedule_id, True) else "未找到该任务")
            return 0
        if args.action == "disable":
            print("已停用" if scheduler.set_enabled(schedule_id, False) else "未找到该任务")
            return 0

        schedule = scheduler.get(schedule_id)
        if schedule is None:
            print("未找到该任务", file=sys.stderr)
            return 1
        result = await runtime.start_run(schedule["task"], on_event=_print_event)
        scheduler.mark_ran(schedule_id)
        print(
            f"\n[schedule #{schedule_id}] status={result['status']} steps={result['steps']} "
            f"{_usage_line(result)}"
        )
        return 0 if result["status"] == "completed" else 1

    return 2


def _knowledge_command(runtime: Runtime, args: argparse.Namespace) -> int:
    store = runtime.knowledge

    if args.action == "list":
        docs = store.list_documents()
        if not docs:
            print("知识库为空，用 `knowledge add <文件>` 或 `--title/--text` 导入")
            return 0
        for doc in docs:
            source = f" · {doc['source']}" if doc["source"] else ""
            print(f"#{doc['id']}\t{doc['chunks']}块\t{doc['title']}{source}")
        return 0

    if args.action == "stats":
        stats = store.stats()
        print(f"文档 {stats['documents']} 篇 · 分块 {stats['chunks']} 个")
        return 0

    if args.action == "add":
        if args.title and args.text:
            doc_id = store.add_text(args.title, args.text)
            print(f"已写入 #{doc_id}")
            return 0
        if not args.value:
            print("用法: knowledge add <文件路径>   或   knowledge add --title 标题 --text 正文", file=sys.stderr)
            return 2
        try:
            doc_id = store.add_file(args.value)
        except (FileNotFoundError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"已导入 #{doc_id}")
        return 0

    if args.action == "search":
        if not args.value:
            print("用法: knowledge search <问题或关键词>", file=sys.stderr)
            return 2
        hits = store.search(args.value, limit=args.limit)
        if not hits:
            print("没有找到相关内容")
            return 0
        for hit in hits:
            print(f"\n## {hit['title']}（#{hit['doc_id']} · 相关度 {hit['score']}）\n{hit['text']}")
        return 0

    if args.action == "remove":
        if not args.value or not args.value.lstrip("#").isdigit():
            print("用法: knowledge remove <doc_id>", file=sys.stderr)
            return 2
        doc_id = int(args.value.lstrip("#"))
        print("已删除" if store.remove(doc_id) else "未找到该文档")
        return 0

    return 2


async def _parallel_command(runtime: Runtime, args: argparse.Namespace) -> int:
    coordinator = runtime.make_coordinator()
    subtasks = None
    if getattr(args, "subtasks", None):
        subtasks = [s.strip() for s in args.subtasks.split(",") if s.strip()]

    result = await coordinator.run_parallel(args.task, subtasks=subtasks, on_event=_print_event)
    print(f"\n状态={result['status']} 成功={result['completed']}/{result['total']}")
    return 0 if result["status"] == "completed" else 1


async def _run_cli(args: argparse.Namespace, cfg: Config) -> int:
    runtime = await Runtime.build(cfg, _make_approver())
    try:
        if args.cmd == "tools":
            for spec in runtime.registry.list_tools():
                print(f"{spec.risk.value}\t{spec.kind.value}\t{spec.name}\t{spec.description[:70]}")
            for name, error in runtime.registry.errors.items():
                print(f"[error]\t{name}\t{error}", file=sys.stderr)
            return 0

        if args.cmd == "sources":
            for row in await runtime.registry.health():
                mark = "UP  " if row["ok"] else "DOWN"
                print(f"{mark}\t{row['source']}\t{row['detail']}")
            return 0

        if args.cmd == "skills":
            if args.action == "reload":
                count = await runtime.reload_skills()
                print(f"已重载 {count} 个 Skill")
                return 0
            skills = runtime.skills.list()
            if not skills:
                print(f"技能目录为空：{runtime.skills.root}")
                return 0
            for skill in skills:
                kind = "可执行" if skill.executable else "知识"
                print(
                    f"{skill.risk}\t{kind}\t{skill.name}\tv{skill.version}\t"
                    f"{skill.entry or '-'}\t{skill.description[:60]}"
                )
            return 0

        if args.cmd == "sessions":
            rows = runtime.sessions.list()
            if not rows:
                print("还没有会话")
                return 0
            for row in rows:
                print(f"{row['id']}\t{row['status']}\t{row['task'][:60]}")
            return 0

        if args.cmd == "resume":
            info = runtime.sessions.get(args.session_id)
            if info is None:
                print(f"会话不存在：{args.session_id}", file=sys.stderr)
                return 2
            result = await runtime.start_run(
                info["task"], session_id=args.session_id, on_event=_print_event
            )
            print(
                f"\n[session {result['session_id']}] status={result['status']} "
                f"steps={result['steps']} {_usage_line(result)}"
            )
            return 0 if result["status"] == "completed" else 1

        if args.cmd == "memory":
            return _memory_command(runtime.memory, args)

        if args.cmd == "recipes":
            rows = runtime.recipes.list()
            if not rows:
                print("还没有配方：成功执行过的任务会自动录制")
                return 0
            for row in rows:
                print(f"#{row['id']}\t{row['step_count']}步\t已用{row['uses']}次\t{row['task'][:60]}")
            return 0

        if args.cmd == "replay":
            try:
                result = await runtime.agent.replay(args.recipe_id, on_event=_print_event)
            except KeyError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            print(
                f"\n[session {result['session_id']}] status={result['status']} "
                f"steps={result['steps']} {_usage_line(result)}"
            )
            return 0 if result["status"] == "completed" else 1

        if args.cmd == "schedule":
            return await _schedule_command(runtime, args)

        if args.cmd == "knowledge":
            return _knowledge_command(runtime, args)

        if args.cmd == "parallel":
            return await _parallel_command(runtime, args)

        if args.cmd == "run":
            if getattr(args, "no_record", False):
                runtime.config.record_recipes = False
            result = await runtime.start_run(args.task, args.session, on_event=_print_event)
            recipe = f" recipe=#{result['recipe_id']}" if result.get("recipe_id") else ""
            print(
                f"\n[session {result['session_id']}] status={result['status']} "
                f"steps={result['steps']}{recipe} {_usage_line(result)}"
            )
            return 0 if result["status"] == "completed" else 1

        return 2
    finally:
        await runtime.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="doubao-agent", description="豆包云电脑 Agent")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 /workspace/agent.yaml）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tools", help="列出统一注册表中的全部工具")
    sub.add_parser("sources", help="检查各接入通道健康状态")
    skills_parser = sub.add_parser("skills", help="列出 Skill 或热重载技能目录")
    skills_parser.add_argument(
        "action", nargs="?", choices=["list", "reload"], default="list", help="list | reload"
    )
    sub.add_parser("sessions", help="列出最近的会话")

    resume_parser = sub.add_parser("resume", help="从检查点恢复会话")
    resume_parser.add_argument("session_id", help="会话编号")

    run_parser = sub.add_parser("run", help="执行一个任务")
    run_parser.add_argument("task", help="任务描述")
    run_parser.add_argument("--session", default=None, help="恢复已有会话")
    run_parser.add_argument("--no-record", action="store_true", help="不把本次执行录制为配方")

    sub.add_parser("recipes", help="列出已录制的配方")
    replay_parser = sub.add_parser("replay", help="按配方零模型重放")
    replay_parser.add_argument("recipe_id", type=int, help="配方编号")

    memory_parser = sub.add_parser("memory", help="长期记忆：查看 / 写入 / 检索 / 遗忘 / 统计")
    memory_parser.add_argument("action", choices=["list", "add", "search", "forget", "stats"])
    memory_parser.add_argument("value", nargs="?", default=None)
    memory_parser.add_argument("--kind", default="fact", choices=list(MEMORY_KINDS))
    memory_parser.add_argument("--limit", type=int, default=20)

    schedule_parser = sub.add_parser("schedule", help="cron 定时任务：add / list / remove / enable / disable / run")
    schedule_parser.add_argument("action", choices=["add", "list", "remove", "enable", "disable", "run"])
    schedule_parser.add_argument("args", nargs="*", default=[])
    schedule_parser.add_argument("--name", default=None, help="任务名称（仅 add）")

    knowledge_parser = sub.add_parser("knowledge", help="知识库：文档导入 / 检索 / 删除 / 统计")
    knowledge_parser.add_argument("action", choices=["list", "add", "search", "remove", "stats"])
    knowledge_parser.add_argument("value", nargs="?", default=None)
    knowledge_parser.add_argument("--title", default=None, help="直接写入时的标题")
    knowledge_parser.add_argument("--text", default=None, help="直接写入时的正文")
    knowledge_parser.add_argument("--limit", type=int, default=4, help="检索返回片段数")

    parallel_parser = sub.add_parser("parallel", help="并行协同：拆解任务、并发执行子任务并汇总")
    parallel_parser.add_argument("task")
    parallel_parser.add_argument("--subtasks", default=None, help="逗号分隔的子任务列表（省略则自动拆解）")

    serve_parser = sub.add_parser("serve", help="启动本地控制台")
    serve_parser.add_argument("--port", type=int, default=None)

    daemon_parser = sub.add_parser("daemon", help="常驻守护：start / stop / status")
    daemon_parser.add_argument("action", choices=["start", "stop", "status"])
    daemon_parser.add_argument("--port", type=int, default=None)

    audit_parser = sub.add_parser("audit", help="审计：校验哈希链 / 列出有审计的会话")
    audit_parser.add_argument("action", nargs="?", choices=["verify", "sessions"], default="verify")
    audit_parser.add_argument("session", nargs="?", default=None, help="只校验指定会话")

    args, extra = parser.parse_known_args(argv)
    if extra and args.cmd in ("memory", "knowledge"):
        # argparse 的已知怪癖：可选参数在前时，位置参数会落进 extra。
        pieces = ([args.value] if getattr(args, "value", None) else []) + extra
        args.value = " ".join(pieces)
    elif extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    cfg = load_config(args.config)

    if args.cmd == "daemon":
        return _daemon_command(args, cfg)

    if args.cmd == "audit":
        return _audit_command(args, cfg)

    if args.cmd == "serve":
        try:
            import uvicorn
        except ImportError:
            print("需要安装 uvicorn：uv pip install uvicorn", file=sys.stderr)
            return 2
        from .server.app import create_app

        uvicorn.run(create_app(cfg), host="0.0.0.0", port=args.port or cfg.port)
        return 0

    return asyncio.run(_run_cli(args, cfg))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
