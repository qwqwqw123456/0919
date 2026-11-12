from __future__ import annotations

import asyncio
import hmac
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..config import Config
from ..runtime import Runtime
from ..types import ToolSpec

STATIC_DIR = Path(__file__).parent / "static"


def _tool_dict(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "kind": spec.kind.value,
        "risk": spec.risk.value,
        "source": spec.source,
        "schema": spec.input_schema,
    }


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
        scheduler_task = None
        if runtime.config.scheduler_enabled:
            scheduler_task = asyncio.create_task(_run_scheduler(runtime, bus))
        try:
            yield
        finally:
            for task in list(_bg_tasks):
                task.cancel()
            if scheduler_task is not None:
                scheduler_task.cancel()
            await runtime.aclose()

    app = FastAPI(title="Doubao Agent Console", lifespan=lifespan)

    auth_token = (config.auth_token or "").strip()

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
        if not hmac.compare_digest(provided, auth_token):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        response = await call_next(request)
        if request.query_params.get("token"):
            response.set_cookie(
                "agent_token", auth_token, httponly=True, samesite="strict"
            )
        return response

    def rt() -> Runtime:
        return state["runtime"]

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
        doc_id = rt().knowledge.add_text(title, text, source=(payload or {}).get("source", ""))
        return {"id": doc_id}

    @app.get("/api/knowledge/search")
    async def knowledge_search(q: str = "", limit: int = 4) -> list[dict[str, Any]]:
        return rt().knowledge.search(q, limit=limit)

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

    @app.post("/api/parallel")
    async def parallel(payload: dict[str, Any]) -> dict[str, Any]:
        task = (payload or {}).get("task", "").strip()
        if not task:
            raise HTTPException(status_code=400, detail="task is required")
        subtasks = (payload or {}).get("subtasks")
        if subtasks is not None and isinstance(subtasks, list):
            subtasks = [str(s).strip() for s in subtasks if str(s).strip()]

        runtime = rt()

        async def _background() -> None:
            try:
                await runtime.make_coordinator().run_parallel(
                    task, subtasks=subtasks, on_event=bus.publish
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

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
