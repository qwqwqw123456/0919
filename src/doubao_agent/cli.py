from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import subprocess
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import yaml

from .config import Config, load_config
from .core.audit import AuditLog
from .core.multimodal import ingest
from .core.evolution import EvolutionStore
from .core.recipe import RecipeStore
from .core.intake import explain as explain_intake, parse_task
from .core.memory import KINDS as MEMORY_KINDS
from .core.outbound import redact
from .core.profile import ProfileStore
from .mcp_server import serve, serve_http
from .runtime import Runtime
from .skills import SkillManager
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


def _collect_run_usage(runs_dir: str) -> list[dict[str, Any]]:
    """按会话汇总 token 用量：取每个会话最后一条带 usage 的 finish。

    续跑会在同一会话追加多条 finish，取最后一条即本轮真实累计，不重复计。
    数据来自审计流水，已脱敏，不含凭据。
    """
    rows: list[dict[str, Any]] = []
    for path in glob.glob(os.path.join(runs_dir, "*", "audit.jsonl")):
        session = os.path.basename(os.path.dirname(path))
        task = ""
        usage = None
        stamp = ""
        for line in open(path, encoding="utf-8"):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "start":
                task = str(event.get("task") or "")
            elif event.get("event") == "finish" and event.get("usage"):
                usage = event["usage"]
                stamp = str(event.get("ts") or "")
        if usage:
            rows.append(
                {
                    "session": session,
                    "task": task,
                    "ts": stamp,
                    "calls": int(usage.get("calls") or 0),
                    "tokens": int(usage.get("total_tokens") or 0),
                }
            )
    return rows


def _model_command(args: argparse.Namespace, cfg: Config) -> int:
    import asyncio

    from doubao_agent.core import model_channels as mc

    rows = mc.channels(cfg)
    action = args.action

    if action == "list":
        if not rows:
            print("没有配置任何模型通道")
            return 0
        for row in rows:
            mark = "*" if row.get("current") else " "
            key = mc.mask_key(mc.resolve_key(row))
            print(
                f"{mark} {row['name']:<10} {row.get('model') or '-':<26} "
                f"{row.get('base_url') or '-':<44} key={key}"
            )
        print("\n* 为当前主通道；切通道：doubao-agent model use <name>")
        return 0

    if action == "use":
        if not args.name:
            print("用法：doubao-agent model use <name>")
            return 2
        try:
            info = mc.switch_channel(cfg.config_path, args.name)
        except KeyError as exc:
            print(str(exc))
            return 2
        print(f"已切主通道：{info['to']}（model={info['model']}，旧主通道转 fallback={info['fallback']}）")
        return 0

    channel = mc.find_channel(cfg, args.name) if args.name else mc.find_channel(cfg, "main")
    if not channel:
        print(f"找不到通道：{args.name}")
        return 2

    if action in ("models", "probe"):
        if action == "models":
            try:
                names = asyncio.run(mc.list_models(channel))
            except Exception as exc:  # noqa: BLE001 - 如实报错
                print(f"拉取模型列表失败：{exc}")
                return 1
            print(f"通道 {channel['name']} 可用模型 {len(names)} 个：")
            for name in names:
                print("  " + name)
            return 0
        result = asyncio.run(mc.probe(channel, args.set_model))
        if result.get("ok"):
            print(
                f"通道 {channel['name']} 探活通过｜model={result['model']}"
                f"｜延迟 {result['latency_s']}s｜回复：{result.get('reply') or '(空)'}"
            )
            return 0
        print(f"通道 {channel['name']} 探活失败：{result.get('error')}")
        return 1
    return 2


def _usage_command(args: argparse.Namespace, cfg: Config) -> int:
    """累计额度视图：已用 token / 余额，防止总额度被无声耗尽。"""
    quota = max(0, int(args.quota))
    rows = _collect_run_usage(str(cfg.runs_dir))
    used = sum(r["tokens"] for r in rows)
    calls = sum(r["calls"] for r in rows)
    left = quota - used
    pct = (used / quota * 100) if quota else 0.0
    print(f"会话 {len(rows)} 个 ｜ 调用 {calls} 次 ｜ 已用 {used:,} token")
    if quota:
        print(f"总额度 {quota:,} ｜ 剩余 {left:,} ｜ 已用 {pct:.1f}%")
    if args.detail and rows:
        print("\n【按消耗排序（前 5）】")
        for row in sorted(rows, key=lambda r: -r["tokens"])[:5]:
            task = (row["task"][:28] + "…") if len(row["task"]) > 28 else (row["task"] or "（无记录）")
            print(f"  {row['session']}  {row['tokens']:>9,}  {task}")
    if quota and left <= 0:
        print("额度已耗尽：需补充额度或临时换通道，别再跑长任务")
    return 0


def _policy_command(args: argparse.Namespace, cfg: Config) -> int:
    """策略层命令组：show / set / preset / suggest / stats。全部落在 agent.yaml 的 policy 段。"""
    from .core.netpolicy import PRESETS, Policy, PolicyError

    path = str(cfg.config_path) if str(cfg.config_path) else None
    perf = Policy(path)
    cmd = args.policy_cmd
    try:
        if cmd == "show":
            for row in perf.explain():
                print(f"{row['key']}\t{row['value']}\t[{row['source']}]")
            print(f"\npreset={perf.snapshot()['preset'] or '(none)'} version={perf.version}")
            return 0
        if cmd == "set":
            items = {}
            for piece in args.items:
                if "=" not in piece:
                    raise PolicyError(f"要写成 key=value：{piece}")
                k, v = piece.split("=", 1)
                items[k.strip()] = v.strip()
            perf.set_many(items)
            for k in items:
                fam, name = k.split(".", 1)
                print(f"{k}={getattr(perf.net if fam == 'net' else perf.model, name)}  (已写入 {path or '内存'})")
            return 0
        if cmd == "preset":
            perf.preset(args.name)
            print(f"preset={args.name} version={perf.version}")
            for row in perf.explain():
                if row["source"].startswith("preset"):
                    print(f"  {row['key']}={row['value']}")
            return 0
        if cmd == "suggest":
            tips = perf.suggest()
            if not tips:
                print("暂无建议：当前没有足够实测延迟，或现有取值已合理。")
                return 0
            for tip in tips:
                print(f"  {tip['key']} → {tip['to']}   （{tip['why']}）")
            print("\n应用：doubao-agent policy set " + " ".join(f"{t['key']}={t['to']}" for t in tips))
            return 0
        if cmd == "stats":
            print(json.dumps(perf.stats(), ensure_ascii=False, indent=2))
            return 0
    except PolicyError as exc:
        print(f"策略配置不合法：{exc}", file=sys.stderr)
        return 2
    return 2


def _tree_command(args: argparse.Namespace, cfg: Config) -> int:
    """技能树台账：把运行证据回写到 A/P/B/C，并按共现自动加边。"""
    from .core.audit import AuditLog
    from .core.session import SessionStore
    from .core.skilltree import TreeLedger
    from .core.telemetry import Telemetry

    ledger = TreeLedger(Path(cfg.workspace) / "tree")
    if not ledger.ready:
        print(f"技能树索引缺失或不可读：{ledger.index_path}", file=sys.stderr)
        return 1
    cmd = args.tree_cmd
    if cmd == "status":
        state = ledger.state()
        lv = state["levels"]
        print(
            f"29 类｜证据 {state['evidence_total']} 条｜A {lv['A']} · P {lv['P']} · "
            f"B {lv['B']} · C {lv['C']}｜升级 {state['upgraded']}｜自动边 {len(state['auto_edges'])}"
        )
        reg = [f"#{no}" for no, c in state["cats"].items() if c.get("regressed")]
        if reg:
            print(f"  ⚠ 回归（有失败但等级未降）：{'、'.join(sorted(reg, key=lambda x: int(x[1:])))}")
        for no in sorted(state["cats"], key=lambda x: int(x)):
            cat = state["cats"][no]
            mark = "↑" if cat["upgraded"] else ("!" if cat["regressed"] else " ")
            print(
                f"  #{cat['no']:>2} {cat['level']}{mark} {cat['name'][:26]:<28} "
                f"ok {cat['ok']} / fail {cat['fail']}"
            )
        return 0
    if cmd == "ingest":
        sessions = SessionStore(cfg.db_path)
        audit = AuditLog(cfg.runs_dir)
        telemetry = Telemetry(cfg.db_path)
        if args.rebuild:
            ledger.evidence_path.write_text("", encoding="utf-8")
            ledger._seen_cache = set()
        ids = args.sessions or []
        if args.scan:
            ids = [row["id"] for row in sessions.list(limit=int(args.limit))]
        if not ids:
            print("给 --session <ID>，或 --scan 扫最近会话", file=sys.stderr)
            return 2
        written = 0
        for sid in ids:
            info = sessions.get(sid) or {}
            if not info:
                print(f"  {sid} 不存在", file=sys.stderr)
                continue
            rec = ledger.ingest(
                sid,
                str(info.get("task") or ""),
                audit.read(sid),
                telemetry.list(sid),
                status=str(info.get("status") or ""),
            )
            if rec:
                written += 1
                cats = ", ".join(f"#{c['no']}({c['score']})" for c in rec["cats"])
                print(f"  {sid} → {rec['level']}  {cats}")
        print(f"写入 {written} 条证据（重复会话自动跳过）")
        return 0
    if cmd == "edges":
        state = ledger.state()
        if not state["auto_edges"]:
            print("还没有自动边：需要同一次任务同时命中两类且不在手绘边里。")
            return 0
        for edge in state["auto_edges"]:
            print(f"  #{edge['from']} → #{edge['to']}  ×{edge['count']}  {'/'.join(edge['sessions'])}")
        return 0
    if cmd == "diff":
        d = ledger.diff()
        if not d.get("ok"):
            print(f"暂无可对比快照：{d.get('reason')}", file=sys.stderr)
            return 1
        print(f"{d['from']['at']} → {d['to']['at']}｜证据 +{d['evidence_delta']}")
        print(f"  新边 {len(d['new_edges'])}｜消失边 {len(d['gone_edges'])}｜新工具边 {len(d['new_tool_edges'])}")
        for e in d["new_edges"]:
            print(f"    + {e}")
        for e in d["gone_edges"]:
            print(f"    - {e}")
        for c in d["level_changes"]:
            print(f"    #{c['no']} {c['from']} → {c['to']}")
        return 0
    if cmd == "tools":
        state = ledger.state()
        edges = state["tool_edges"]
        print(f"工具边（达标）{len(edges)} 条；模板只在 A 级任务里计")
        for e in edges:
            print(f"  {e['tool']} → #{e['cat']}  ×{e['count']}  {'/'.join(e['sessions'][:2])}")
        if not edges:
            print("  还没有达标的模板→类别组合（需 ≥3 次且跨 ≥2 会话）")
        return 0
    if cmd == "triage":
        from .core.freshness import triage_markdown

        path = triage_markdown(ledger, Path(args.out) if args.out else None, only=args.cve)
        print(f"核查单已写：{path}")
        return 0
    if cmd == "fresh":
        from .core.freshness import Freshness

        fresh = Freshness(ledger)
        if args.file:
            payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
            res = fresh.ingest(payload, days=int(args.days), limit=int(args.limit))
        else:
            res = fresh.fetch(days=int(args.days), limit=int(args.limit))
        if not res.get("ok"):
            print(f"抓取失败：{res.get('error')}", file=sys.stderr)
            print(f"现有候选 {res.get('total', 0)} 条，未写新数据", file=sys.stderr)
            return 1
        print(f"新增候选 {res['added']} 条（累计 {res['total']}），已写 {res['path']}")
        for item in res.get("items", []):
            cats = ", ".join(f"#{c['no']}" for c in item["cats"])
            print(f"  {item['id']} {item['date']} [{cats}] {item['title'][:44]}")
        return 0
    if cmd == "export":
        out = Path(args.out).expanduser()
        path = ledger.export(out)
        print(f"已导出：{path}")
        return 0
    return 2


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
            installed = SkillManager(cfg.skills_dir).refresh()
            print(
                f"已装技能 {installed} 个（skills_dir）｜ 有用量记账 {data['total']} 个"
                f" ｜ active {data['state_counts']['active']} / "
                f"stale {data['state_counts']['stale']} / archived {data['state_counts']['archived']} "
                f"｜ 钉住 {data['pinned']}"
            )
            if not data["total"]:
                print(f"（{installed} 个技能均未被执行过：这是没有记账，不等于技能缺失）")
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


def _intake_command(runtime: Runtime, args: argparse.Namespace) -> int:
    text = " ".join(args.text).strip()
    if not text:
        print("用法: intake \"盯着某场演出开票，有票就提醒我\" [--save]", file=sys.stderr)
        return 2
    try:
        spec = parse_task(text)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(explain_intake(spec))
    if not args.save:
        print("\n（未登记；加 --save 就写成定时任务）")
        return 0
    schedule_id = runtime.scheduler.add(spec.title, spec.cron, spec.describe())
    row = runtime.scheduler.get(schedule_id)
    print(f"\n已登记跟进 #{schedule_id}，下次执行 {_fmt_time(row['next_run'] if row else None)}")
    if spec.confirm_level >= 2:
        print("确认级 2：执行前会先找你要一次明确同意，不会直接动手")
    return 0


def _profile_command(runtime: Runtime, args: argparse.Namespace) -> int:
    store = ProfileStore(runtime.config.db_path)
    try:
        if args.action == "set":
            if len(args.args) < 2:
                print("用法: profile set <键> <值> [--source 来源] [--confidence 0~1]", file=sys.stderr)
                return 2
            try:
                entry = store.set(
                    args.args[0],
                    " ".join(args.args[1:]),
                    source=args.source,
                    confidence=args.confidence,
                    note=args.note or "",
                )
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            print(f"已记住：{entry['key']}={entry['value']}")
            return 0

        if args.action == "get":
            if not args.args:
                print("用法: profile get <键>", file=sys.stderr)
                return 2
            entry = store.get(args.args[0])
            print(
                f"{entry['key']}={entry['value']}（来源 {entry['source']}，置信 {entry['confidence']}，"
                f"{_fmt_time(entry['updated_at'])}）"
                if entry
                else "没有这条"
            )
            return 0

        if args.action in ("list", "show"):
            if args.action == "show":
                print(store.context_text() or "还没有画像，用 `profile set 称呼 老板` 开始")
                return 0
            rows = store.list("all")
            if not rows:
                print("还没有画像")
                return 0
            for row in rows:
                print(
                    f"{row['status']}\t{row['key']}={row['value']}\t来源={row['source']}\t"
                    f"置信={row['confidence']}\t更新={_fmt_time(row['updated_at'])}"
                )
            return 0

        if args.action == "forget":
            if not args.args:
                print("用法: profile forget <键>", file=sys.stderr)
                return 2
            ok = store.forget(args.args[0])
            print("已停用，不再注入" if ok else "没有这条")
            return 0 if ok else 1

        if args.action == "history":
            if not args.args:
                print("用法: profile history <键>", file=sys.stderr)
                return 2
            rows = store.history(args.args[0])
            if not rows:
                print("没有变更记录")
                return 0
            for row in rows:
                print(f"{_fmt_time(row['changed_at'])}\t{row['old_value']} → {row['new_value']}")
            return 0

        print(f"未知动作: {args.action}", file=sys.stderr)
        return 2
    finally:
        store.close()


def _notify_command(runtime: Runtime, args: argparse.Namespace) -> int:
    text = " ".join(args.text).strip()
    if not text:
        print("用法: notify \"要推给用户的内容\" [--fingerprint 去重键]", file=sys.stderr)
        return 2
    result = runtime.notifier.send(text, fingerprint=args.fingerprint)
    if result["ok"]:
        print(f"已发送（{runtime.notifier.config.channel}）")
        return 0
    print(f"未发送：{result['status']} / {result.get('reason', '')}", file=sys.stderr)
    print(f"[脱敏后内容] {redact(text)}")
    return 1


def _secrets_command(cfg: Config, args: argparse.Namespace) -> int:
    """凭据后端自查：只报可用性，不回显任何密钥值。"""
    from .core.secrets import status

    for row in status():
        mark = "可用" if row["available"] else "不可用"
        print(f"{mark}\t{row['name']}\t{row['note']}")
    if cfg.secret_sources:
        sources = "、".join(f"{name}←{source}" for name, source in cfg.secret_sources.items())
        print(f"本次启动已兜底补入：{sources}")
    else:
        print("本次启动无兜底补入（环境变量已覆盖或后端均不可用）")
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
    if args.cmd == "eval-tasks":
        # 评测只量本体系闭环，不挂宿主 MCP：它的传输层在不同任务里关停会抛 cancel scope 错，把整批带崩
        cfg.mcp_auto_discover = False
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
            if args.action == "audit":
                from .core.skill_audit import audit, format_report

                result = audit(runtime.skills, smoke=not getattr(args, "no_smoke", False))
                if getattr(args, "json", False):
                    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))
                else:
                    print(format_report(result))
                return 0 if not result["static_bad"] and not result["smoke_bad"] else 1
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

        if args.cmd == "route":
            from .core.router import decide_route, stratum_of

            info = decide_route(args.task, max_simple_chars=args.chars)
            stratum = stratum_of({"route": info["route"], "risk": info["risk"]}, None)
            print(
                json.dumps(
                    {
                        "route": info["route"],
                        "reason": info["reason"],
                        "risk": info["risk"],
                        "stratum": stratum,
                        "chars": len(args.task),
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        if args.cmd == "eval-tasks":
            from .core.task_eval import load_tasks, prepare_workspace, run_tasks, save_report

            payload = load_tasks()
            cases = list(payload.get("cases") or [])
            if getattr(args, "ids", None):
                wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
                cases = [c for c in cases if c.get("id") in wanted]
            if getattr(args, "limit", 0):
                cases = cases[: args.limit]
            workspace = prepare_workspace(
                str(payload.get("workspace") or "/home/user/agent-state/eval-workspace")
            )

            def _show(row: dict) -> None:
                mark = "PASS" if row["pass"] else "FAIL"
                print(
                    f"{mark}\t{row['id']}\t{row['stratum']}\t{row['route']}\t"
                    f"{row['calls']} 调用 / {row['tokens']} token / {row['seconds']}s"
                    + (f"\n     问题：{'；'.join(row['issues'])}" if row["issues"] else "")
                )

            report = await run_tasks(
                runtime, cases, workspace, timeout=args.timeout, on_case=_show,
                concurrency=max(1, int(args.concurrency)),
            )
            report["env"] = {
                "model": (runtime.config.model or {}).get("model"),
                "base_url": (runtime.config.model or {}).get("base_url"),
                "concurrency": max(1, int(args.concurrency)),
                "timeout": args.timeout,
                "commit": _git_commit(),
                "cwd": str(Path.cwd()),
            }
            target = save_report(report)
            if args.json:
                print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))
            else:
                print(
                    f"\n通过 {report['pass']}/{report['total']}（{report['pass_rate']:.1%}）"
                    f"｜token {report['tokens']}｜明细 {target}"
                )
                for key, val in report["by_stratum"].items():
                    print(f"  {key}: {val['pass']}/{val['total']} = {val['rate']:.0%}")
            return 0 if report["pass"] == report["total"] else 1

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

        if args.cmd == "voice":
            from .types import ToolCall

            if args.voice_cmd == "say":
                call = ToolCall(call_id="cli", name="voice.say",
                                arguments={"text": args.text, "filename": args.out})
            elif args.voice_cmd == "transcribe":
                call = ToolCall(call_id="cli", name="voice.transcribe",
                                arguments={"path": args.path})
            else:
                call = ToolCall(call_id="cli", name="voice.call",
                                arguments={"path": args.file, "text": args.text})
            res = await runtime.registry.call(call)
            print(res.content if res.ok else f"[error] {res.error}")
            return 0 if res.ok else 1

        if args.cmd == "intake":
            return _intake_command(runtime, args)

        if args.cmd == "profile":
            return _profile_command(runtime, args)

        if args.cmd == "notify":
            return _notify_command(runtime, args)

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
            transport = args.transport or os.environ.get(
                "AGENT_MCP_SERVER_TRANSPORT", "stdio"
            )
            if transport == "http":
                host = args.host or os.environ.get("AGENT_MCP_SERVER_HOST", "127.0.0.1")
                port = int(
                    args.port
                    or os.environ.get("AGENT_MCP_SERVER_PORT", "8766")
                )
                await serve_http(runtime.registry, expose=expose, host=host, port=port)
            else:
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
            bundle = ingest(
                list(getattr(args, "attach", None) or []),
                screen=bool(getattr(args, "screen", False)),
                ocr_enabled=bool(getattr(args, "ocr", False)),
            )
            for warning in bundle.warnings:
                print(f"[附件] {warning}", file=sys.stderr)
            images = list(args.image or []) + bundle.images
            task = args.task
            if bundle.notes:
                task = f"{bundle.notes}\n\n任务：{args.task}"
            result = await runtime.start_run(
                task, args.session, on_event=_print_event, images=images or None
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


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


def check_serve_host(host: str, auth_token: str) -> str | None:
    """控制台监听地址准入：非回环且没有 auth_token 时拒绝启动，返回问题描述。"""
    if (host or "").strip() in LOOPBACK_HOSTS:
        return None
    if (auth_token or "").strip():
        return None
    return (
        f"监听 {host} 属于对外暴露，但 auth_token 为空，/api/* 将无鉴权开放；"
        "请先配置 auth_token 或改回 127.0.0.1。"
    )


def _git_commit() -> str:
    """评测报告要能对上代码版本，不然两次结果没法比。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="doubao-agent", description="豆包云电脑 Agent")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 /workspace/agent.yaml）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tools", help="列出统一注册表中的全部工具")
    sub.add_parser("sources", help="检查各接入通道健康状态")
    skills_parser = sub.add_parser("skills", help="列出 Skill / 热重载 / 审计")
    skills_parser.add_argument(
        "action",
        nargs="?",
        choices=["list", "reload", "install", "uninstall", "audit"],
        default="list",
        help="list | reload | install <目录> | uninstall <名称>",
    )
    skills_parser.add_argument("target", nargs="?", default=None, help="安装目录或技能名称")
    skills_parser.add_argument("--no-smoke", action="store_true", help="审计时跳过实跑")
    skills_parser.add_argument("--json", action="store_true", help="审计输出 JSON")
    mcp_parser = sub.add_parser("mcp", help="运行时登记 MCP（写入 .agent/mcp.local.yaml）")
    mcp_parser.add_argument("action", choices=["list", "add", "remove"], help="list | add | remove")
    mcp_parser.add_argument("name", nargs="?", default=None, help="MCP 名称")
    mcp_parser.add_argument("--url", default=None, help="http / sse 地址")
    mcp_parser.add_argument("--transport", default=None, help="http | sse，默认 http")
    mcp_parser.add_argument("--command", default=None, help="stdio 启动命令")
    mcp_parser.add_argument("--args", nargs="*", default=None, help="stdio 命令参数")
    mcp_serve_parser = sub.add_parser(
        "mcp-serve", help="把注册表工具暴露为标准 MCP server（默认 stdio，仅 L0 只读）"
    )
    mcp_serve_parser.add_argument(
        "--expose",
        choices=["read", "all"],
        default=None,
        help="read 仅暴露 L0 只读（默认）；all 放开全部风险等级，仅限本机受控场景",
    )
    mcp_serve_parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=None,
        help="stdio 子进程标准输入输出（默认）；http 为 Streamable HTTP，用 URL 直挂",
    )
    mcp_serve_parser.add_argument(
        "--host", default=None, help="http 传输绑定地址，默认 127.0.0.1（仅本机）"
    )
    mcp_serve_parser.add_argument(
        "--port", type=int, default=None, help="http 传输端口，默认 8766"
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
    run_parser.add_argument(
        "--attach", action="append", default=None,
        help="附带任意附件（可重复）：图片 / 视频（抽帧+字幕）/ 音频（读同名转写稿）",
    )
    run_parser.add_argument("--screen", action="store_true", help="先截当前桌面再跑（需图形会话）")
    run_parser.add_argument("--ocr", action="store_true", help="对附带的图片/帧做 OCR，文字进上下文")

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

    voice_parser = sub.add_parser("voice", help="语音通道：say 合成 / transcribe 识别 / call 跑一轮通话")
    voice_sub = voice_parser.add_subparsers(dest="voice_cmd", required=True)
    v_say = voice_sub.add_parser("say", help="文字 → wav")
    v_say.add_argument("text")
    v_say.add_argument("--out", default="")
    v_trans = voice_sub.add_parser("transcribe", help="音频 → 文字")
    v_trans.add_argument("path")
    v_call = voice_sub.add_parser("call", help="跑一轮语音通话（--file 给音频，或用 --text 模拟对方说话）")
    v_call.add_argument("--file", default="")
    v_call.add_argument("--text", default="")
    intake_parser = sub.add_parser("intake", help="人话入口：一句话 → 任务卡（--save 直接登记）")
    intake_parser.add_argument("text", nargs="*", default=[], help="一句话交代")
    intake_parser.add_argument("--save", action="store_true", help="直接登记成定时任务")

    profile_parser = sub.add_parser("profile", help="关于你：键值画像 set/get/list/show/forget/history")
    profile_parser.add_argument("action", choices=["set", "get", "list", "show", "forget", "history"])
    profile_parser.add_argument("args", nargs="*", default=[])
    profile_parser.add_argument("--source", default="user", help="来源：user / 推断 / 导入")
    profile_parser.add_argument("--confidence", type=float, default=1.0, help="置信 0~1")
    profile_parser.add_argument("--note", default="", help="备注")

    notify_parser = sub.add_parser("notify", help="出站通道：把一条消息推给用户")
    notify_parser.add_argument("text", nargs="*", default=[])
    notify_parser.add_argument("--fingerprint", default=None, help="去重键；默认按内容哈希")

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

    a2a_parser = sub.add_parser("a2a", help="A2A 客户端：discover / send / cancel 远端 Agent")
    a2a_parser.add_argument("action", choices=["discover", "send", "cancel"])
    a2a_parser.add_argument("--url", required=True, help="对端 A2A 端点，如 http://127.0.0.1:8799/a2a")
    a2a_parser.add_argument("--task", default=None, help="send 时的任务文本")
    a2a_parser.add_argument("--task-id", default=None, help="cancel 时的 task id")
    a2a_parser.add_argument("--timeout", type=float, default=600.0, help="最长等待秒数，默认 600")
    a2a_parser.add_argument("--allow", action="append", default=None, help="追加出站白名单（可重复）")

    serve_parser = sub.add_parser("serve", help="启动本地控制台")
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument(
        "--host", default=None, help="监听地址（默认 127.0.0.1；绑非回环必须配置 auth_token）"
    )

    daemon_parser = sub.add_parser("daemon", help="常驻守护：start / stop / status")
    daemon_parser.add_argument("action", choices=["start", "stop", "status"])
    daemon_parser.add_argument("--port", type=int, default=None)

    secrets_parser = sub.add_parser("secrets", help="凭据后端：查可用性（keyring / 平台 vault / 加密库）")
    secrets_parser.add_argument("action", choices=["status"], nargs="?", default="status")

    audit_parser = sub.add_parser("audit", help="审计：校验哈希链 / 列出有审计的会话")
    audit_parser.add_argument(
        "action", nargs="?", choices=["verify", "sessions", "export"], default="verify"
    )
    audit_parser.add_argument("session", nargs="?", default=None, help="只校验指定会话")
    audit_parser.add_argument("--out", default=None, help="导出路径（默认 runs/<session>/audit.export.jsonl）")

    model_parser = sub.add_parser("model", help="模型通道：list / use / probe / models")
    model_parser.add_argument("action", choices=["list", "use", "probe", "models"], help="动作")
    model_parser.add_argument("name", nargs="?", default=None, help="通道名（use/probe/models）")
    model_parser.add_argument("--set-model", default=None, help="探活时临时指定模型名")

    eval_parser = sub.add_parser("eval-tasks", help="任务级评测：真实任务 + 确定性验收")
    eval_parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    eval_parser.add_argument("--ids", default=None, help="指定用例 id，逗号分隔")
    eval_parser.add_argument("--timeout", type=float, default=240.0, help="单条超时秒数")
    eval_parser.add_argument("--json", action="store_true", help="打印 JSON 汇总")
    eval_parser.add_argument("--concurrency", type=int, default=4, help="并发条数（默认 4，1 为串行）")

    calibrate_parser = sub.add_parser("calibrate", help="分流标定：样本门控 + 离线回放候选阈值")
    route_parser = sub.add_parser("route", help="看一条任务会被分发到哪条路径")
    route_parser.add_argument("--task", required=True, help="待判定的任务原文")
    route_parser.add_argument("--chars", type=int, default=40, help="短任务字数阈值")

    calibrate_parser.add_argument("--replay", action="store_true", help="跑候选阈值回放")
    calibrate_parser.add_argument("--chars", default=None, help="候选字符阈值，逗号分隔")
    calibrate_parser.add_argument("--json", action="store_true", help="输出 JSON")

    policy_parser = sub.add_parser(
        "policy", help="策略层：网络通信 / 模型等待的超时、重试、退避、轮询（改完即生效）"
    )
    policy_sub = policy_parser.add_subparsers(dest="policy_cmd", required=True)
    policy_sub.add_parser("show", help="列出全部策略项：当前值与来源")
    p_set = policy_sub.add_parser("set", help="改一项或多项，如 model.retries=3 net.read_timeout=20")
    p_set.add_argument("items", nargs="+", help="key=value，可给多个")
    p_preset = policy_sub.add_parser("preset", help=f"套预设：{', '.join(['turbo', 'balanced', 'careful'])}")
    p_preset.add_argument("name")
    policy_sub.add_parser("suggest", help="按实测延迟给提速/更稳的建议（只建议不改）")
    policy_sub.add_parser("stats", help="近期调用延迟分位与失败数")

    tree_parser = sub.add_parser("tree", help="技能树台账：证据回写 A/P/B/C、自动加边、导出")
    tree_sub = tree_parser.add_subparsers(dest="tree_cmd", required=True)
    tree_sub.add_parser("status", help="29 类当前等级与证据计数")
    t_ingest = tree_sub.add_parser("ingest", help="把会话证据回写图谱")
    t_ingest.add_argument("--session", dest="sessions", action="append", help="会话 ID，可给多个")
    t_ingest.add_argument("--scan", action="store_true", help="扫最近会话批量回写")
    t_ingest.add_argument("--limit", type=int, default=30, help="--scan 时最多扫多少条（默认 30）")
    t_ingest.add_argument("--rebuild", action="store_true", help="清空证据后全量重算（模板/分类都按当前口径）")
    tree_sub.add_parser("edges", help="列出由共现自动生成的边")
    tree_sub.add_parser("diff", help="对比最近两份快照：新边/等级变化/证据增量")
    tree_sub.add_parser("tools", help="工具模板台账：模板→类别命中与达标工具边")
    t_triage = tree_sub.add_parser("triage", help="给 P 候选生成复现核查单（只读检查，不含利用载荷）")
    t_triage.add_argument("--id", dest="cve", default="", help="只处理某一条 CVE")
    t_triage.add_argument("--out", default="", help="输出 Markdown 路径")
    t_fresh = tree_sub.add_parser("fresh", help="抓公开漏洞源进 P 候选队列（不自动升 A）")
    t_fresh.add_argument("--days", type=int, default=30, help="只看最近 N 天（0=不限）")
    t_fresh.add_argument("--limit", type=int, default=50, help="本次最多入队多少条")
    t_fresh.add_argument("--file", default="", help="离线 JSON（跳过网络，用于复现）")
    t_export = tree_sub.add_parser("export", help="导出成图谱可吃的 EV/edges JSON")
    t_export.add_argument("--out", default="tree-export.json", help="输出路径")

    usage_parser = sub.add_parser("usage", help="额度视图：累计 token 用量与总额度余额")
    usage_parser.add_argument("--quota", type=int, default=10_000_000, help="总额度 token（默认 1000 万）")
    usage_parser.add_argument("--detail", action="store_true", help="列出消耗最高的会话")

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

    if args.cmd == "secrets":
        return _secrets_command(cfg, args)

    if args.cmd == "audit":
        return _audit_command(args, cfg)

    if args.cmd == "evolve":
        return _evolve_command(args, cfg)

    if args.cmd == "policy":
        return _policy_command(args, cfg)

    if args.cmd == "tree":
        return _tree_command(args, cfg)

    if args.cmd == "model":
        return _model_command(args, cfg)

    if args.cmd == "calibrate":
        from doubao_agent.core.calibration import main as calibrate_main

        argv = ["--runs-dir", str(cfg.runs_dir)]
        if getattr(args, "replay", False):
            argv.append("--replay")
        if getattr(args, "chars", None):
            argv += ["--chars", args.chars]
        if getattr(args, "json", False):
            argv.append("--json")
        return calibrate_main(argv)

    if args.cmd == "usage":
        return _usage_command(args, cfg)

    if args.cmd == "serve":
        try:
            import uvicorn
        except ImportError:
            print("需要安装 uvicorn：uv pip install uvicorn", file=sys.stderr)
            return 2
        from .server.app import create_app

        host = (args.host or cfg.serve_host or "127.0.0.1").strip()
        problem = check_serve_host(host, cfg.auth_token)
        if problem:
            print(f"拒绝启动：{problem}", file=sys.stderr)
            return 2
        cfg.serve_host = host
        cfg.port = args.port or cfg.port
        uvicorn.run(create_app(cfg), host=host, port=cfg.port)
        return 0

    if args.cmd == "a2a":
        return asyncio.run(_a2a_command(args, cfg))

    return asyncio.run(_run_cli(args, cfg))


async def _a2a_command(args: argparse.Namespace, cfg: Config) -> int:
    """A2A 客户端命令行；token 只从环境变量读，不接受明文参数。"""
    from .core.a2a_client import A2AClient, A2AError, summarize_card, summarize_task

    allow = list(getattr(cfg, "a2a_allow", []) or [])
    if args.allow:
        allow.extend(args.allow)
    token = os.environ.get("AGENT_A2A_TOKEN", "")
    client: A2AClient | None = None
    try:
        client = A2AClient(args.url, token=token, allow=allow, timeout=max(10.0, args.timeout))
        if args.action == "discover":
            print(summarize_card(await client.card()))
            return 0
        if args.action == "send":
            if not (args.task or "").strip():
                print("send 需要 --task", file=sys.stderr)
                return 2
            task = await client.send(args.task, wait=True, max_wait=args.timeout)
            print(summarize_task(task))
            return 0 if str(task.get("status", {}).get("state")) == "completed" else 1
        if args.action == "cancel":
            if not (args.task_id or "").strip():
                print("cancel 需要 --task-id", file=sys.stderr)
                return 2
            print(summarize_task(await client.cancel(args.task_id)))
            return 0
        return 2
    except A2AError as exc:
        print(f"A2A 失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            await client.aclose()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
