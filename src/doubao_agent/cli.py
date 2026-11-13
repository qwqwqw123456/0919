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

import yaml

from .config import Config, load_config
from .core.audit import AuditLog
from .core.evolution import EvolutionStore
from .core.recipe import RecipeStore
from .core.memory import KINDS as MEMORY_KINDS
from .mcp_server import serve
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
    elif kind == "review":
        conflicts = event.get("conflicts") or []
        suffix = f"；冲突：{'；'.join(conflicts)}" if conflicts else ""
        print(f"\n[审阅] {event.get('verdict')}{suffix}", file=sys.stderr)
    elif kind == "finish":
        print("\n" + str(event.get("answer", "")), file=sys.stderr)


def _usage_line(result: dict[str, Any]) -> str:
    usage = result.get("usage") or {}
    return f"调用={usage.get('calls', 0)} tokens={usage.get('total_tokens', 0)}"


def _evolve_command(args: argparse.Namespace, cfg: Config) -> int:
    """自进化命令组：读用量、跑生命周期、蒸馏配方。只改状态位，不删技能文件。"""
    if not cfg.evolution_enabled:
        print("自进化已关闭（evolution.enabled=false 或 AGENT_EVOLUTION=0）", file=sys.stderr)
        return 1
    store = EvolutionStore(
        cfg.db_path,
        stale_days=cfg.evolution_stale_days,
        archive_days=cfg.evolution_archive_days,
    )
    try:
        if args.action == "status":
            data = store.status(top=args.top)
            print(
                f"技能 {data['total']} 个 ｜ active {data['state_counts']['active']} / "
                f"stale {data['state_counts']['stale']} / archived {data['state_counts']['archived']} "
                f"｜ 钉住 {data['pinned']}"
            )
            if not data["total"]:
                print("（还没有遥测数据：跑一次用到技能的任务后再看）")
                return 0
            print("\n【使用排行】")
            if data["top"]:
                for row in data["top"]:
                    last = (
                        time.strftime("%Y-%m-%d %H:%M", time.localtime(row["last_used_at"]))
                        if row["last_used_at"]
                        else "—"
                    )
                    print(
                        f"  {row['name']:<28} use={row['use_count']:<4} view={row['view_count']:<4} "
                        f"patch={row['patch_count']:<3} 最后={last}"
                    )
            else:
                print("  （还没有技能被真正执行过）")
            print("\n【冷技能：一次没用过】")
            if data["cold"]:
                for row in data["cold"]:
                    mark = "📌" if row["pinned"] else ""
                    print(f"  {row['name']:<28} 闲置 {row['idle_days']} 天 {mark}")
            else:
                print("  （无）")
            return 0

        if args.action == "audit":
            plan = store.audit(apply=args.apply)
            if not plan:
                print("无流转：全部技能都在阈值内，或被 pin 保护")
                return 0
            print(f"拟流转 {len(plan)} 项：")
            for item in plan:
                print(f"  {item['name']:<28} {item['from']} → {item['to']}（闲置 {item['idle_days']} 天）")
            if not args.apply:
                print("\n这是预演，未改动状态。确认后加 --apply（只改状态位，不删文件）")
                return 0
            print("\n已流转；技能文件原样保留，路径引用不受影响")
            return 0

        if args.action in ("pin", "unpin"):
            if not args.target:
                print(f"用法：evolve {args.action} <技能名>", file=sys.stderr)
                return 2
            ok = store.pin(args.target, args.action == "pin")
            if not ok:
                print(f"技能不存在：{args.target}", file=sys.stderr)
                return 1
            print(f"{args.target} 已{'钉住（不会被自动流转）' if args.action == 'pin' else '解除钉住'}")
            return 0

        if args.action == "distill":
            if not args.target or not str(args.target).isdigit():
                print("用法：evolve distill <配方编号>", file=sys.stderr)
                return 2
            recipes = RecipeStore(cfg.db_path)
            try:
                recipe = recipes.get(int(args.target))
            finally:
                recipes.close()
            if recipe is None:
                print(f"配方不存在：{args.target}", file=sys.stderr)
                return 1
            try:
                path = store.distill(recipe, cfg.skills_dir, force=args.force)
            except (ValueError, FileExistsError) as exc:
                print(f"失败：{exc}", file=sys.stderr)
                return 1
            print(f"已生成技能草案：{path}")
            print("草案放在 _drafts 下，不会被自动加载；确认后执行：doubao-agent skills install <草案目录>")
            return 0

        if args.action == "report":
            data = store.status(top=args.top)
            rows = store.rows()
            recent = [r for r in rows if r["last_used_at"]]
            print("== 技能生态周报 ==")
            print(f"技能总数 {data['total']}｜活跃 {data['state_counts']['active']}｜"
                  f"闲置 {data['state_counts']['stale']}｜归档 {data['state_counts']['archived']}")
            print(f"最近有使用记录 {len(recent)} 个，从未使用 {data['total'] - len(recent)} 个")
            hot = data["top"][:3]
            if hot:
                print("高频：" + "、".join(f"{r['name']}({r['use_count']})" for r in hot))
            if data["cold"]:
                print("待观察：" + "、".join(r["name"] for r in data["cold"][:5]))
            print("下一步：evolve audit 预演 → evolve audit --apply 流转 → 冷技能决定补登记或下线")
            return 0
    finally:
        store.close()
    return 0


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

    if args.action == "export":
        if not args.session:
            print("用法: audit export <session> [--out 路径]", file=sys.stderr)
            return 2
        try:
            path, count = log.export(args.session, args.out)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"已导出 {count} 条 → {path}")
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

    if args.action == "decay":
        count = memory.decay(half_life_days=args.days)
        print(f"已按半衰期 {args.days} 天衰减 {count} 条记忆")
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


def _state_command(runtime: Runtime, args: argparse.Namespace) -> int:
    working = runtime.working
    if args.action == "show":
        for key, value in working.load().items():
            if key == "写入锁":
                continue
            print(f"{key}\t{value}")
        return 0
    if not args.key or args.value is None:
        print("用法: state set <字段> <值>", file=sys.stderr)
        return 2
    value: Any = args.value
    if args.key == "会话计数":
        try:
            value = int(args.value)
        except ValueError:
            print("会话计数必须是整数", file=sys.stderr)
            return 2
    try:
        working.update(**{args.key: value})
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"已更新 {args.key}")
    return 0


def _mcp_command(runtime: Runtime, args: argparse.Namespace) -> int:
    """登记 / 移除 MCP：写入 .agent/mcp.local.yaml，下次启动生效，不改 agent.yaml。"""
    path = runtime.config.db_path.parent / "mcp.local.yaml"
    entries: list[dict[str, Any]] = []
    if path.is_file():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or []
            entries = [e for e in loaded if isinstance(e, dict)]
        except (yaml.YAMLError, OSError):
            entries = []

    if args.action == "list":
        if not runtime.config.mcp_servers:
            print("未配置任何 MCP（可用 mcp add 登记）")
        for server in runtime.config.mcp_servers:
            target = server.url or f"{server.command} {' '.join(server.args or [])}"
            print(f"{server.transport}\t{server.name}\t{target}")
        return 0

    if not args.name:
        print(f"用法: mcp {args.action} <名称> [--url 地址 | --command 命令 --args ...]", file=sys.stderr)
        return 2

    if args.action == "remove":
        remaining = [e for e in entries if e.get("name") != args.name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(remaining, allow_unicode=True, sort_keys=False), encoding="utf-8")
        if len(remaining) == len(entries):
            print("未找到该 MCP（可能来自 agent.yaml，需手动删除）", file=sys.stderr)
            return 1
        print(f"已移除 {args.name}，下次启动生效")
        return 0

    if args.command:
        entry: dict[str, Any] = {
            "name": args.name,
            "transport": "stdio",
            "command": args.command,
            "args": args.args or [],
        }
    elif args.url:
        entry = {"name": args.name, "transport": args.transport or "http", "url": args.url}
    else:
        print("mcp add 需要 --url 或 --command", file=sys.stderr)
        return 2
    entries = [e for e in entries if e.get("name") != args.name] + [entry]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(entries, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"已登记 {args.name}（{entry['transport']}），下次启动生效")
    return 0


def _longterm_command(runtime: Runtime, args: argparse.Namespace) -> int:
    store = runtime.longterm
    if args.kind == "stats":
        print(" · ".join(f"{name} {count}" for name, count in store.stats().items()))
        return 0

    if args.action == "add":
        if args.kind == "lesson":
            if not args.event:
                print("用法: longterm lesson add --event 事件 [--judgment/--outcome/--lesson]", file=sys.stderr)
                return 2
            row_id = store.add_lesson(
                args.event, args.judgment, args.outcome, args.lesson, args.date
            )
        elif args.kind == "observation":
            if not args.phenomenon:
                print("用法: longterm observation add --phenomenon 现象 [--observation/--cause/--followup]", file=sys.stderr)
                return 2
            row_id = store.add_observation(
                args.phenomenon, args.observation, args.cause, args.followup, args.date
            )
        else:
            if not args.expected or not args.actual:
                print("用法: longterm deviation add --expected 值 --actual 值 [--gap/--cause/--fix]", file=sys.stderr)
                return 2
            row_id = store.add_deviation(
                args.expected, args.actual, args.gap, args.cause, args.fix, args.date
            )
        print(f"已写入 #{row_id}")
        return 0

    rows = {
        "lesson": store.lessons,
        "observation": store.observations,
        "deviation": store.deviations,
    }[args.kind](args.limit)
    if not rows:
        print("（暂无记录）")
        return 0
    for row in rows:
        parts = [
            f"{key}={value}"
            for key, value in row.items()
            if key not in ("id", "date", "created_at") and value
        ]
        print(f"{row['date']}\t{' | '.join(parts)}")
    return 0


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

    if args.action == "domains":
        for item in store.domain_stats():
            print(f"{item['code']} {item['name']}\t{item['documents']} 篇")
        return 0

    if args.action == "add":
        if args.title and args.text:
            doc_id = store.add_text(args.title, args.text, domain=args.domain or "")
            print(f"已写入 #{doc_id}")
            return 0
        if not args.value:
            print("用法: knowledge add <文件路径>   或   knowledge add --title 标题 --text 正文", file=sys.stderr)
            return 2
        try:
            doc_id = store.add_file(args.value, domain=args.domain or "")
        except (FileNotFoundError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"已导入 #{doc_id}")
        return 0

    if args.action == "search":
        if not args.value:
            print("用法: knowledge search <问题或关键词> [--domain 01]", file=sys.stderr)
            return 2
        try:
            hits = store.search(args.value, limit=args.limit, domain=args.domain)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
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

    result = await coordinator.run_parallel(
        args.task,
        subtasks=subtasks,
        on_event=_print_event,
        review=getattr(args, "review", False),
    )
    print(f"\n状态={result['status']} 成功={result['completed']}/{result['total']}")
    if result.get("review"):
        conflicts = result["review"]["conflicts"]
        print(f"审阅={result['review']['verdict']} 冲突={len(conflicts)}")
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
            if args.action in ("install", "uninstall"):
                if not args.target:
                    print(f"用法: skills {args.action} <目录或名称>", file=sys.stderr)
                    return 2
                try:
                    if args.action == "install":
                        skill = runtime.skills.install(args.target)
                        print(f"已安装 {skill.name} v{skill.version} → {skill.directory}")
                    else:
                        removed = runtime.skills.uninstall(args.target)
                        print("已卸载" if removed else "未找到该技能")
                except (ValueError, OSError) as exc:
                    print(f"失败：{exc}", file=sys.stderr)
                    return 1
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

        if args.cmd == "state":
            return _state_command(runtime, args)

        if args.cmd == "longterm":
            return _longterm_command(runtime, args)

        if args.cmd == "parallel":
            return await _parallel_command(runtime, args)

        if args.cmd == "mcp":
            return _mcp_command(runtime, args)

        if args.cmd == "mcp-serve":
            expose = args.expose or os.environ.get("AGENT_MCP_SERVER_EXPOSE", "read")
            if expose not in ("read", "all"):
                expose = "read"
            await serve(runtime.registry, expose=expose)
            return 0

        if args.cmd == "metrics":
            session = getattr(args, "session", None)
            if getattr(args, "detail", False):
                if not session:
                    print("--detail 需要配合 --session 使用", file=sys.stderr)
                    return 2
                for span in runtime.telemetry.list(session):
                    mark = "ok" if span["ok"] else "fail"
                    print(
                        f"{span['kind']}\t{span['duration_ms']}ms\t{mark}\t"
                        f"{span['name']}\t{span['attributes']}"
                    )
                return 0
            print(
                json.dumps(
                    runtime.telemetry.summary(session),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.cmd == "run":
            if getattr(args, "no_record", False):
                runtime.config.record_recipes = False
            result = await runtime.start_run(
                args.task, args.session, on_event=_print_event, images=args.image
            )
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
        "action",
        nargs="?",
        choices=["list", "reload", "install", "uninstall"],
        default="list",
        help="list | reload | install <目录> | uninstall <名称>",
    )
    skills_parser.add_argument("target", nargs="?", default=None, help="安装目录或技能名称")
    mcp_parser = sub.add_parser("mcp", help="运行时登记 MCP（写入 .agent/mcp.local.yaml）")
    mcp_parser.add_argument("action", choices=["list", "add", "remove"], help="list | add | remove")
    mcp_parser.add_argument("name", nargs="?", default=None, help="MCP 名称")
    mcp_parser.add_argument("--url", default=None, help="http / sse 地址")
    mcp_parser.add_argument("--transport", default=None, help="http | sse，默认 http")
    mcp_parser.add_argument("--command", default=None, help="stdio 启动命令")
    mcp_parser.add_argument("--args", nargs="*", default=None, help="stdio 命令参数")
    mcp_serve_parser = sub.add_parser(
        "mcp-serve", help="把注册表工具暴露为标准 MCP server（stdio，默认仅 L0 只读）"
    )
    mcp_serve_parser.add_argument(
        "--expose",
        choices=["read", "all"],
        default=None,
        help="read 仅暴露 L0 只读（默认）；all 放开全部风险等级，仅限本机受控场景",
    )
    metrics_parser = sub.add_parser("metrics", help="可观测指标：span 汇总 / 慢调用 / 失败数")
    metrics_parser.add_argument("--session", default=None, help="只看指定会话")
    metrics_parser.add_argument("--detail", action="store_true", help="按时间顺序列出该会话全部 span")
    sub.add_parser("sessions", help="列出最近的会话")

    resume_parser = sub.add_parser("resume", help="从检查点恢复会话")
    resume_parser.add_argument("session_id", help="会话编号")

    run_parser = sub.add_parser("run", help="执行一个任务")
    run_parser.add_argument("task", help="任务描述")
    run_parser.add_argument("--session", default=None, help="恢复已有会话")
    run_parser.add_argument("--no-record", action="store_true", help="不把本次执行录制为配方")
    run_parser.add_argument(
        "--image", action="append", default=None, help="附带图片（可重复），需模型支持视觉"
    )

    sub.add_parser("recipes", help="列出已录制的配方")
    replay_parser = sub.add_parser("replay", help="按配方零模型重放")
    replay_parser.add_argument("recipe_id", type=int, help="配方编号")

    memory_parser = sub.add_parser("memory", help="长期记忆：查看 / 写入 / 检索 / 遗忘 / 统计")
    memory_parser.add_argument(
        "action", choices=["list", "add", "search", "forget", "stats", "decay"]
    )
    memory_parser.add_argument("value", nargs="?", default=None)
    memory_parser.add_argument("--kind", default="fact", choices=list(MEMORY_KINDS))
    memory_parser.add_argument("--limit", type=int, default=20)
    memory_parser.add_argument("--days", type=float, default=30.0, help="衰减半衰期（天，仅 decay）")

    schedule_parser = sub.add_parser("schedule", help="cron 定时任务：add / list / remove / enable / disable / run")
    schedule_parser.add_argument("action", choices=["add", "list", "remove", "enable", "disable", "run"])
    schedule_parser.add_argument("args", nargs="*", default=[])
    schedule_parser.add_argument("--name", default=None, help="任务名称（仅 add）")

    knowledge_parser = sub.add_parser(
        "knowledge", help="知识库：文档导入 / 检索 / 删除 / 统计 / 七域索引"
    )
    knowledge_parser.add_argument(
        "action", choices=["list", "add", "search", "remove", "stats", "domains"]
    )
    knowledge_parser.add_argument("value", nargs="?", default=None)
    knowledge_parser.add_argument("--title", default=None, help="直接写入时的标题")
    knowledge_parser.add_argument("--text", default=None, help="直接写入时的正文")
    knowledge_parser.add_argument("--domain", default=None, help="知识域：01~07 或域名称")
    knowledge_parser.add_argument("--limit", type=int, default=4, help="检索返回片段数")

    state_parser = sub.add_parser("state", help="会话状态（工作记忆）：查看 / 修改")
    state_parser.add_argument("action", choices=["show", "set"], nargs="?", default="show")
    state_parser.add_argument("key", nargs="?", default=None, help="字段名，如 焦点线")
    state_parser.add_argument("value", nargs="?", default=None, help="字段新值")

    longterm_parser = sub.add_parser("longterm", help="长期记忆：萃取库 / 观察记录 / 偏差档案")
    longterm_parser.add_argument(
        "kind", choices=["lesson", "observation", "deviation", "stats"]
    )
    longterm_parser.add_argument("action", nargs="?", choices=["add", "list"], default="list")
    longterm_parser.add_argument("--limit", type=int, default=10)
    longterm_parser.add_argument("--date", default=None)
    longterm_parser.add_argument("--event", default="")
    longterm_parser.add_argument("--judgment", default="")
    longterm_parser.add_argument("--outcome", default="")
    longterm_parser.add_argument("--lesson", default="")
    longterm_parser.add_argument("--phenomenon", default="")
    longterm_parser.add_argument("--observation", default="")
    longterm_parser.add_argument("--cause", default="")
    longterm_parser.add_argument("--followup", default="")
    longterm_parser.add_argument("--expected", default="")
    longterm_parser.add_argument("--actual", default="")
    longterm_parser.add_argument("--gap", default=None)
    longterm_parser.add_argument("--fix", default="")

    evolve_parser = sub.add_parser("evolve", help="自进化：技能用量 / 生命周期审计 / 配方蒸馏")
    evolve_parser.add_argument(
        "action", choices=["status", "audit", "pin", "unpin", "distill", "report"],
        help="status 用量 | audit 流转 | pin/unpin 钉住 | distill 配方转技能草案 | report 周报",
    )
    evolve_parser.add_argument("target", nargs="?", default=None, help="技能名或配方编号")
    evolve_parser.add_argument("--apply", action="store_true", help="audit：真正执行流转（默认只预演）")
    evolve_parser.add_argument("--top", type=int, default=10, help="排行榜条数")
    evolve_parser.add_argument("--force", action="store_true", help="distill：覆盖已有草案")

    parallel_parser = sub.add_parser("parallel", help="并行协同：拆解任务、并发执行子任务并汇总")
    parallel_parser.add_argument("task")
    parallel_parser.add_argument("--subtasks", default=None, help="逗号分隔的子任务列表（省略则自动拆解）")
    parallel_parser.add_argument("--review", action="store_true", help="追加审阅者：合并结果并指出冲突")

    serve_parser = sub.add_parser("serve", help="启动本地控制台")
    serve_parser.add_argument("--port", type=int, default=None)

    daemon_parser = sub.add_parser("daemon", help="常驻守护：start / stop / status")
    daemon_parser.add_argument("action", choices=["start", "stop", "status"])
    daemon_parser.add_argument("--port", type=int, default=None)

    audit_parser = sub.add_parser("audit", help="审计：校验哈希链 / 列出有审计的会话")
    audit_parser.add_argument(
        "action", nargs="?", choices=["verify", "sessions", "export"], default="verify"
    )
    audit_parser.add_argument("session", nargs="?", default=None, help="只校验指定会话")
    audit_parser.add_argument("--out", default=None, help="导出路径（默认 runs/<session>/audit.export.jsonl）")

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

    if args.cmd == "evolve":
        return _evolve_command(args, cfg)

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
