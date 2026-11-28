from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..config import Config
from ..core.netpolicy import PRESETS, PolicyError
from ..core.a2a import A2AHandler
from ..core.watcher import _run_watcher
from ..runtime import Runtime
from ..voice.ws import handle_voice_ws
from ..types import ToolSpec

STATIC_DIR = Path(__file__).parent / "static"


def _base_url(config: Config) -> str:
    host = (config.serve_host or "127.0.0.1").strip() or "127.0.0.1"
    port = int(getattr(config, "port", 0) or 8765)
    return f"http://{host}:{port}"


def _tool_dict(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "kind": spec.kind.value,
        "risk": spec.risk.value,
        "source": spec.source,
        "schema": spec.input_schema,
    }


def _verify_webhook_signature(token: str, body: bytes, header: str) -> bool:
    """校验入站 webhook 签名（``X-Agent-Signature: sha256=<hexdigest>``）。

    未带签名头表示不走签名，交由令牌中间件把关；带了签名头但未配令牌则无法验签。
    """
    if not header:
        return True
    if not token:
        return False
    expected = hmac.new(token.encode(), body, hashlib.sha256).hexdigest()
    supplied = header.split("=", 1)[1].strip() if "=" in header else header.strip()
    return hmac.compare_digest(expected, supplied)


class EventBus:
    """Fan-out of agent events to SSE subscribers (bounded, drop-on-full)."""

    def __init__(self) -> None:
        self._subs: set[asyncio.Queue[dict[str, Any]]] = set()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subs.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subs.discard(queue)

    def publish(self, event: dict[str, Any]) -> None:
        for queue in list(self._subs):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass


FRESH_INTERVAL = 20 * 3600  # 20 小时；避免同一天重复抓


async def _run_tree_fresh(runtime: Runtime) -> None:
    """每天最多一次抓 CISA KEV 进 P 队列；失败只记日志，不影响服务。"""
    from ..core.freshness import Freshness
    from ..core.skilltree import TreeLedger

    while True:
        try:
            root = Path(runtime.config.workspace) / "tree"
            mark = root / "fresh-last.json"
            due = True
            if mark.is_file():
                try:
                    last = float(json.loads(mark.read_text(encoding="utf-8")).get("ts") or 0)
                    due = time.time() - last >= FRESH_INTERVAL
                except (OSError, ValueError, json.JSONDecodeError):
                    due = True
            if due and root.is_dir():
                ledger = TreeLedger(root)
                res = await asyncio.to_thread(Freshness(ledger).fetch, days=30, limit=40)
                mark.write_text(json.dumps({"ts": round(time.time(), 3), "ok": res.get("ok"),
                                            "added": res.get("added", 0), "total": res.get("total", 0),
                                            "error": res.get("error", "")}, ensure_ascii=False),
                                encoding="utf-8")
        except Exception as exc:  # 后台任务不能把服务带下去
            print(f"[tree-fresh] {type(exc).__name__}: {exc}", flush=True)
        await asyncio.sleep(3600)


async def _run_scheduler(runtime: Runtime, bus: EventBus) -> None:
    """Poll due cron schedules and dispatch them concurrently (bounded, non-blocking)."""

    interval = runtime.config.scheduler_interval
    semaphore = asyncio.Semaphore(3)
    inflight: set[str] = set()
    tasks: set[asyncio.Task[Any]] = set()

    async def dispatch(schedule: dict[str, Any]) -> None:
        schedule_id = schedule["id"]
        async with semaphore:
            try:
                bus.publish(
                    {"type": "schedule_start", "id": schedule_id, "name": schedule["name"]}
                )
                await runtime.start_run(schedule["task"])
            except Exception:  # noqa: BLE001 - one schedule must not kill the loop
                pass
            finally:
                runtime.scheduler.mark_ran(schedule_id)
                inflight.discard(schedule_id)

    try:
        while True:
            try:
                for schedule in runtime.scheduler.due():
                    if schedule["id"] in inflight:
                        continue
                    inflight.add(schedule["id"])
                    task = asyncio.create_task(dispatch(schedule))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
            except Exception:  # noqa: BLE001 - the loop must never die
                pass
            await asyncio.sleep(interval)
    finally:
        for task in list(tasks):
            task.cancel()


def create_app(config: Config) -> FastAPI:
    state: dict[str, Runtime] = {}
    bus = EventBus()
    # 持有后台任务强引用，避免事件循环在任务完成前将其回收。
    _bg_tasks: set[asyncio.Task[Any]] = set()

    def _spawn(coro: Any) -> None:
        task = asyncio.create_task(coro)
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = await Runtime.build(config)
        state["runtime"] = runtime
        state["a2a"] = A2AHandler(runtime, _base_url(config))
        scheduler_task = None
        watcher_task = None
        fresh_task = asyncio.create_task(_run_tree_fresh(runtime))
        if runtime.config.scheduler_enabled:
            scheduler_task = asyncio.create_task(_run_scheduler(runtime, bus))
        if runtime.config.watch_enabled:
            watcher_task = asyncio.create_task(_run_watcher(runtime, bus))
        try:
            yield
        finally:
            for task in list(_bg_tasks):
                task.cancel()
            if scheduler_task is not None:
                scheduler_task.cancel()
            if watcher_task is not None:
                watcher_task.cancel()
            fresh_task.cancel()
            await runtime.aclose()

    app = FastAPI(title="Doubao Agent Console", lifespan=lifespan)

    auth_token = (config.auth_token or "").strip()

    def _authorized(request: Request) -> bool:
        """与 /api 中间件同一套凭据口径，供 A2A 端点复用。"""
        if not auth_token:
            return True
        bearer = request.headers.get("authorization", "")
        if bearer.lower().startswith("bearer "):
            bearer = bearer[7:]
        provided = (
            request.headers.get("x-agent-token")
            or bearer
            or request.query_params.get("token", "")
        )
        return bool(provided) and hmac.compare_digest(provided, auth_token)

    @app.middleware("http")
    async def require_token(request, call_next):
        """配置了 auth_token 时，所有 /api/* 需携带令牌。

        令牌可来自 X-Agent-Token、Authorization: Bearer、查询参数 ?token= 或 cookie；
        用查询参数通过一次后会种下 HttpOnly cookie，方便浏览器直接打开控制台。
        静态页本身不拦，页面里由 JS 从 localStorage 读取令牌。
        """
        if not auth_token or not request.url.path.startswith("/api/"):
            return await call_next(request)
        bearer = request.headers.get("authorization", "")
        if bearer.lower().startswith("bearer "):
            bearer = bearer[7:]
        provided = (
            request.headers.get("x-agent-token")
            or bearer
            or request.query_params.get("token", "")
            or request.cookies.get("agent_token", "")
        )
        # 带签名头的 webhook 交由端点做 HMAC 验签，不在中间件按令牌拦截。
        if not provided and not request.headers.get("x-agent-signature"):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        if provided and not hmac.compare_digest(provided, auth_token):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        response = await call_next(request)
        if request.query_params.get("token"):
            response.set_cookie(
                "agent_token", auth_token, httponly=True, samesite="strict"
            )
        return response

    def rt() -> Runtime:
        return state["runtime"]

    @app.websocket("/voice/ws")
    async def voice_ws(websocket: WebSocket):
        """手机/浏览器一路麦：发 PCM16 20ms 帧，收事件 JSON 与回复音频二进制。"""
        await websocket.accept()
        runtime = rt()
        if getattr(runtime, "voice", None) is None:
            await websocket.send_json({"type": "error", "text": "语音通道未启用"})
            await websocket.close()
            return
        session = runtime.voice.make_session()
        pending: list[Any] = []

        async def recv():
            if pending:
                return pending.pop(0)
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                return None
            if msg.get("bytes") is not None:
                return msg["bytes"]
            return msg.get("text")

        async def send_json(payload):
            await websocket.send_json(payload)

        async def send_bytes(data):
            await websocket.send_bytes(data)

        try:
            await handle_voice_ws(recv, send_json, send_bytes, session)
        except WebSocketDisconnect:
            pass
        finally:
            session.close()

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        runtime = rt()
        return {
            "status": "ok",
            "registry": runtime.registry.snapshot(),
            "sources": await runtime.registry.health(),
            "usage": runtime.usage.snapshot(),
        }

    @app.get("/api/tools")
    async def tools() -> list[dict[str, Any]]:
        return [_tool_dict(spec) for spec in rt().registry.list_tools()]

    @app.get("/api/skills")
    async def skills() -> list[dict[str, Any]]:
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "version": skill.version,
                "tags": list(skill.tags),
                "executable": skill.executable,
                "entry": skill.entry,
                "risk": skill.risk,
            }
            for skill in rt().skills.list()
        ]

    @app.get("/api/knowledge")
    async def knowledge_list() -> list[dict[str, Any]]:
        return rt().knowledge.list_documents()

    @app.post("/api/knowledge")
    async def knowledge_add(payload: dict[str, Any]) -> dict[str, Any]:
        title = (payload or {}).get("title", "").strip()
        text = (payload or {}).get("text", "").strip()
        if not title or not text:
            raise HTTPException(status_code=400, detail="title and text are required")
        doc_id = rt().knowledge.add_text(
            title,
            text,
            source=(payload or {}).get("source", ""),
            domain=(payload or {}).get("domain", ""),
        )
        return {"id": doc_id}

    @app.get("/api/knowledge/search")
    async def knowledge_search(q: str = "", limit: int = 4, domain: str = "") -> list[dict[str, Any]]:
        return await rt().knowledge.asearch(q, limit=limit, domain=domain or None)

    @app.delete("/api/knowledge/{doc_id}")
    async def knowledge_remove(doc_id: int) -> dict[str, Any]:
        if not rt().knowledge.remove(doc_id):
            raise HTTPException(status_code=404, detail="document not found")
        return {"deleted": doc_id}

    @app.get("/api/sessions")
    async def sessions() -> list[dict[str, Any]]:
        return rt().sessions.list()

    @app.get("/api/sessions/{session_id}")
    async def session_detail(session_id: str) -> dict[str, Any]:
        runtime = rt()
        info = runtime.sessions.get(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail="session not found")
        return {
            "session": info,
            "messages": runtime.sessions.messages(session_id),
            "artifacts": runtime.sessions.artifacts(session_id),
        }

    @app.get("/api/sessions/{session_id}/audit")
    async def session_audit(session_id: str) -> list[dict[str, Any]]:
        return rt().agent.audit.read(session_id)

    @app.get("/api/metrics")
    async def metrics(session: str | None = None) -> dict[str, Any]:
        return rt().telemetry.summary(session)

    @app.get("/api/sessions/{session_id}/trace")
    async def session_trace(session_id: str) -> list[dict[str, Any]]:
        return rt().telemetry.list(session_id)

    def _perf() -> Any:
        perf = getattr(rt().agent, "perf", None)
        if perf is None:
            raise HTTPException(status_code=404, detail="策略层未启用")
        return perf

    @app.get("/api/policy")
    async def policy_show() -> dict[str, Any]:
        perf = _perf()
        snap = perf.snapshot()
        return {
            "preset": snap.get("preset", ""),
            "version": snap.get("version", 0),
            "presets": sorted(PRESETS),
            "rows": perf.explain(),
            "stats": perf.stats(),
            "suggest": perf.suggest(),
            "path": str(perf.path) if perf.path else "",
        }

    @app.post("/api/policy/preset")
    async def policy_preset(payload: dict[str, Any]) -> dict[str, Any]:
        name = str((payload or {}).get("name", "")).strip()
        try:
            return _perf().preset(name)
        except PolicyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/policy/set")
    async def policy_set(payload: dict[str, Any]) -> dict[str, Any]:
        items = (payload or {}).get("items")
        if not isinstance(items, dict) or not items:
            raise HTTPException(status_code=400, detail="items 需要是 {key: value} 的非空对象")
        try:
            return _perf().set_many(items)
        except PolicyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _tree() -> Any:
        tree = getattr(rt(), "tree", None)
        if tree is None or not tree.ready:
            raise HTTPException(status_code=404, detail="技能树台账未启用")
        return tree

    @app.get("/api/tree")
    async def tree_state() -> dict[str, Any]:
        return _tree().state()

    @app.post("/api/tree/ingest")
    async def tree_ingest(payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str((payload or {}).get("session", "")).strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="session 不能为空")
        rec = rt().ingest_tree(session_id)
        return {"ingested": bool(rec), "record": rec}

    @app.post("/api/tree/scan")
    async def tree_scan(payload: dict[str, Any]) -> dict[str, Any]:
        limit = int((payload or {}).get("limit", 20) or 20)
        rows = rt().sessions.list()[: max(1, min(limit, 200))]
        done, skipped, failed = 0, 0, 0
        for row in rows:
            try:
                rec = rt().ingest_tree(str(row.get("id") or ""))
            except Exception:  # noqa: BLE001 - 单条失败不能中断扫描
                failed += 1
                continue
            done += 1 if rec else 0
            skipped += 0 if rec else 1
        return {"scanned": len(rows), "written": done, "skipped": skipped, "failed": failed}

    @app.post("/api/tree/fresh")
    async def tree_fresh(payload: dict[str, Any]) -> dict[str, Any]:
        from ..core.freshness import Freshness

        days = int((payload or {}).get("days", 30) or 30)
        limit = int((payload or {}).get("limit", 50) or 50)
        fresh = Freshness(_tree())
        return await asyncio.to_thread(fresh.fetch, days=max(0, days), limit=max(1, min(limit, 200)))

    @app.get("/api/tree/export")
    async def tree_export() -> dict[str, Any]:
        tree = _tree()
        path = tree.export(tree.root / "tree-export.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        return payload

    @app.get("/api/tree/lookup")
    async def tree_lookup(q: str = "", limit: int = 20) -> dict[str, Any]:
        from ..core.wordindex import lookup

        return lookup(q, limit=limit)

    @app.get("/api/tree/diff")
    async def tree_diff() -> dict[str, Any]:
        return _tree().diff()

    @app.get("/api/tree/pulse")
    async def tree_pulse(q: str = "", limit: int = 6) -> dict[str, Any]:
        """任务文本 → 推演路径（T0→T4），页面据此跑流光。"""
        return _tree().pulse(q, limit=max(1, min(limit, 10)))

    @app.get("/api/tree/tools")
    async def tree_tools(no: int = 0) -> dict[str, Any]:
        state = _tree().state()
        if no:
            cat = state["cats"].get(str(no))
            if cat is None:
                raise HTTPException(status_code=404, detail=f"没有第 {no} 类")
            return {"ok": True, "no": no, "tools": cat.get("tools_detail") or [],
                    "fails": cat.get("fails") or 0, "fail_last": cat.get("fail_last"),
                    "host": state["host"]}
        return {"ok": True, "tools": state["tool_totals"], "host": state["host"],
                "strategy": state["strategy"]}

    @app.get("/api/tree/fails")
    async def tree_fails(no: int = 0) -> dict[str, Any]:
        data = _tree().fails()
        if no:
            return {"ok": True, "traces": [t for t in data["traces"] if t["no"] == no]}
        return data

    @app.get("/api/tree/strategy")
    async def tree_strategy_get() -> dict[str, Any]:
        return {"ok": True, **_tree().strategy()}

    @app.post("/api/tree/strategy")
    async def tree_strategy_set(payload: dict[str, Any]) -> dict[str, Any]:
        text = str((payload or {}).get("text") or "")
        strat = await asyncio.to_thread(_tree().save_strategy, text)
        # 立刻灌进权限层，下一次工具调用就按新策略拦
        try:
            rt().agent.policy.deny = _tree().deny_rules()
        except Exception:  # noqa: BLE001 - 热更新失败不回滚已保存策略
            pass
        return {"ok": True, **strat}

    @app.post("/api/tree/triage")
    async def tree_triage(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        from ..core.freshness import triage_markdown

        only = str((payload or {}).get("id", "") or "")
        path = await asyncio.to_thread(triage_markdown, _tree(), None, only=only)
        return {"ok": True, "path": str(path), "markdown": path.read_text(encoding="utf-8")}

    @app.get("/api/tree/download")
    async def tree_download(which: str = "agent") -> FileResponse:
        """词条库 / Agent 索引原始 JSON，本地直出，不依赖远端 CDN。"""
        name = {"agent": "sec-agent-index.json", "words": "sec-words.json"}.get(which)
        if name is None:
            raise HTTPException(status_code=400, detail="which 只能是 agent 或 words")
        path = Path(__file__).resolve().parents[3] / "data" / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"缺少 {name}")
        return FileResponse(path, media_type="application/json",
                            filename=("网安Agent索引-2667条.json" if which == "agent" else "网安词条库-2667条.json"))

    @app.post("/api/sessions/{session_id}/resume")
    async def resume_session(session_id: str) -> dict[str, Any]:
        runtime = rt()
        info = runtime.sessions.get(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail="session not found")

        async def _background() -> None:
            try:
                await runtime.start_run(info["task"], session_id, on_event=bus.publish)
            except Exception as exc:  # noqa: BLE001 - keep the console alive
                runtime.sessions.finish(session_id, f"{type(exc).__name__}: {exc}", "failed")

        _spawn(_background())
        return {"session_id": session_id, "status": "resuming"}

    @app.post("/api/sessions/{session_id}/cancel")
    async def cancel_session(session_id: str) -> dict[str, Any]:
        runtime = rt()
        if runtime.sessions.get(session_id) is None:
            raise HTTPException(status_code=404, detail="session not found")
        cancelled = runtime.cancel_session(session_id)
        return {"session_id": session_id, "status": "cancelling" if cancelled else "idle"}

    @app.post("/api/run")
    async def run(payload: dict[str, Any]) -> dict[str, Any]:
        task = (payload or {}).get("task", "").strip()
        if not task:
            raise HTTPException(status_code=400, detail="task is required")
        runtime = rt()
        session_id = runtime.sessions.create(task)

        async def _background() -> None:
            try:
                await runtime.start_run(task, session_id, on_event=bus.publish)
            except Exception as exc:  # noqa: BLE001 - keep the console alive
                runtime.sessions.finish(session_id, f"{type(exc).__name__}: {exc}", "failed")

        _spawn(_background())
        return {"session_id": session_id, "status": "started"}

    @app.post("/api/webhook")
    async def webhook(request: Request) -> dict[str, Any]:
        """入站事件触发：外部系统带令牌或 HMAC 签名起一个任务。

        body: ``{"task": "...", "event": "可选事件名", "payload": {...}}``。
        签名头 ``X-Agent-Signature: sha256=<raw body 的 HMAC-SHA256 摘要>``。
        """
        runtime = rt()
        body = await request.body()
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="invalid JSON")
        task = str((payload or {}).get("task", "")).strip()
        if not task:
            raise HTTPException(status_code=400, detail="task is required")
        if not _verify_webhook_signature(auth_token, body, request.headers.get("x-agent-signature", "")):
            raise HTTPException(status_code=401, detail="invalid signature")

        session_id = runtime.sessions.create(task)
        event = str((payload or {}).get("event", "")).strip()

        async def _background() -> None:
            try:
                await runtime.start_run(task, session_id, on_event=bus.publish)
            except Exception as exc:  # noqa: BLE001 - keep the console alive
                runtime.sessions.finish(session_id, f"{type(exc).__name__}: {exc}", "failed")

        _spawn(_background())
        bus.publish({"type": "webhook_start", "session": session_id, "event": event})
        return {"session_id": session_id, "status": "started", "event": event}

    @app.post("/api/parallel")
    async def parallel(payload: dict[str, Any]) -> dict[str, Any]:
        task = (payload or {}).get("task", "").strip()
        if not task:
            raise HTTPException(status_code=400, detail="task is required")
        subtasks = (payload or {}).get("subtasks")
        if subtasks is not None and isinstance(subtasks, list):
            subtasks = [str(s).strip() for s in subtasks if str(s).strip()]
        review = bool((payload or {}).get("review", False))

        runtime = rt()

        async def _background() -> None:
            try:
                await runtime.make_coordinator().run_parallel(
                    task, subtasks=subtasks, on_event=bus.publish, review=review
                )
            except Exception as exc:  # noqa: BLE001 - keep the console alive
                bus.publish({"type": "finish", "task": task, "answer": f"{type(exc).__name__}: {exc}"})

        _spawn(_background())
        return {"task": task, "status": "started"}

    @app.get("/api/events")
    async def events() -> StreamingResponse:
        queue = bus.subscribe()

        async def generator():
            try:
                while True:
                    event = await queue.get()
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                raise
            finally:
                bus.unsubscribe(queue)

        return StreamingResponse(generator(), media_type="text/event-stream")

    @app.get("/api/schedules")
    async def schedules() -> list[dict[str, Any]]:
        return rt().scheduler.list()

    @app.post("/api/schedules")
    async def add_schedule(payload: dict[str, Any]) -> dict[str, Any]:
        name = (payload or {}).get("name", "").strip() or "schedule"
        cron = (payload or {}).get("cron", "").strip()
        task = (payload or {}).get("task", "").strip()
        if not cron or not task:
            raise HTTPException(status_code=400, detail="cron and task are required")
        try:
            schedule_id = rt().scheduler.add(name, cron, task)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"id": schedule_id, "next_run": rt().scheduler.get(schedule_id)["next_run"]}

    @app.delete("/api/schedules/{schedule_id}")
    async def remove_schedule(schedule_id: int) -> dict[str, Any]:
        if not rt().scheduler.remove(schedule_id):
            raise HTTPException(status_code=404, detail="schedule not found")
        return {"deleted": schedule_id}

    @app.post("/api/schedules/{schedule_id}/toggle")
    async def toggle_schedule(schedule_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        enabled = bool((payload or {}).get("enabled", True))
        if not rt().scheduler.set_enabled(schedule_id, enabled):
            raise HTTPException(status_code=404, detail="schedule not found")
        return {"id": schedule_id, "enabled": enabled}

    @app.post("/api/schedules/{schedule_id}/run")
    async def run_schedule(schedule_id: int) -> dict[str, Any]:
        runtime = rt()
        schedule = runtime.scheduler.get(schedule_id)
        if schedule is None:
            raise HTTPException(status_code=404, detail="schedule not found")
        bus.publish({"type": "schedule_start", "id": schedule_id, "name": schedule["name"]})
        _spawn(runtime.start_run(schedule["task"], on_event=bus.publish))
        runtime.scheduler.mark_ran(schedule_id, now=time.time())
        return {"id": schedule_id, "status": "started"}

    @app.get("/.well-known/agent-card.json")
    @app.get("/agent.json")
    async def agent_card() -> JSONResponse:
        """A2A Agent Card：公开可读，只暴露名称/能力清单，不含令牌与路径。"""
        return JSONResponse(state["a2a"].card())

    @app.get("/a2a")
    async def a2a_card() -> JSONResponse:
        return JSONResponse(state["a2a"].card())

    @app.post("/a2a")
    async def a2a_endpoint(request: Request):
        """A2A JSON-RPC 入口；``message/stream`` 走 SSE，其余返回单个 JSON-RPC 响应。"""
        if not _authorized(request):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        try:
            payload = json.loads(await request.body() or b"{}")
        except json.JSONDecodeError:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "invalid JSON"},
                }
            )
        handler = state["a2a"]
        method = str((payload or {}).get("method") or "") if isinstance(payload, dict) else ""
        if method == "message/stream":
            async def generator():
                async for item in handler.stream(payload):
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

            return StreamingResponse(generator(), media_type="text/event-stream")
        return JSONResponse(await handler.handle(payload))

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/static/tree.html")
    async def tree_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "tree.html")

    return app
