"""A2A（Agent2Agent）协议最小实现。

对外暴露两件东西：
- Agent Card：``GET /.well-known/agent-card.json``（另有 ``/agent.json`` 兼容旧客户端）
- JSON-RPC 端点：``POST /a2a``，方法 ``message/send`` / ``message/stream`` / ``tasks/get`` / ``tasks/cancel``

设计取向：A2A 的 task 直接映射为本机 session，不另建一套任务存储 ——
``tasks/get`` 读到的就是控制台里那条会话，``tasks/cancel`` 走的是同一套取消链路，
审计链与用量也随之连续。长任务默认异步（返回 working 的 Task，客户端轮询），
需要一次拿结果时传 ``configuration.blocking = true``。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator

A2A_VERSION = "0.3.0"

# JSON-RPC / A2A 错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
TASK_NOT_FOUND = -32001
TASK_NOT_CANCELABLE = -32002


class A2AError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _part_text(part: Any) -> str:
    """把 A2A 的 part 转成可读文本；不认识的类型安静跳过，不猜内容。"""
    if not isinstance(part, dict):
        return str(part or "")
    kind = str(part.get("kind") or part.get("type") or "").strip()
    if kind == "text":
        return str(part.get("text") or "")
    if kind == "data":
        data = part.get("data")
        try:
            return json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(data)
    if kind == "file":
        file = part.get("file") if isinstance(part.get("file"), dict) else {}
        name = str(file.get("name") or part.get("name") or "文件")
        uri = str(file.get("uri") or "")
        return f"[附件] {name} {uri}".strip()
    return ""


def parts_to_text(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    chunks = [_part_text(p) for p in parts]
    return "\n".join(c for c in chunks if c.strip()).strip()


def _artifacts(text: str, name: str = "answer") -> list[dict[str, Any]]:
    if not text:
        return []
    return [
        {
            "artifactId": uuid.uuid4().hex[:12],
            "name": name,
            "parts": [{"kind": "text", "text": text}],
        }
    ]


STATE_MAP = {
    "running": "working",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "canceled",
    "canceled": "canceled",
}


def task_from_session(
    session: dict[str, Any],
    context_id: str,
    artifacts: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    state = STATE_MAP.get(str(session.get("status") or ""), "unknown")
    status: dict[str, Any] = {"state": state, "timestamp": _iso(session.get("updated_at"))}
    result = str(session.get("result") or "")
    if state == "failed" and result:
        status["message"] = {
            "role": "agent",
            "messageId": uuid.uuid4().hex[:12],
            "parts": [{"kind": "text", "text": result}],
        }
    task: dict[str, Any] = {
        "kind": "task",
        "id": str(session.get("id")),
        "contextId": context_id,
        "status": status,
        "artifacts": artifacts if artifacts is not None else _artifacts(result),
        "history": history or [],
    }
    return task


def _iso(ts: Any) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))
    except (TypeError, ValueError):
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_agent_card(config: Any, runtime: Any, base_url: str) -> dict[str, Any]:
    name = str(getattr(config, "name", "") or "doubao-agent")
    tokens = 0
    try:
        tokens = int(getattr(config, "max_tokens", 0) or 0)
    except (TypeError, ValueError):
        tokens = 0
    skills: list[dict[str, Any]] = []
    try:
        for spec in runtime.registry.list_tools()[:24]:
            skills.append(
                {
                    "id": spec.name.replace(".", "_"),
                    "name": spec.name,
                    "description": spec.description,
                    "tags": [spec.kind.value, spec.risk.value, spec.source],
                }
            )
    except Exception:  # noqa: BLE001 - 卡片生成不应因注册表异常失败
        skills = []
    card: dict[str, Any] = {
        "protocolVersion": A2A_VERSION,
        "name": name,
        "description": "本地数字员工：可调用工具、读写本地文件、按审计链留痕的任务执行 Agent。",
        "url": f"{base_url.rstrip('/')}/a2a",
        "preferredTransport": "JSONRPC",
        "version": str(getattr(config, "version", "") or "0.3.0"),
        "capabilities": {"streaming": True, "pushNotifications": False},
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain"],
        "skills": skills,
    }
    if tokens:
        card["x-limits"] = {"maxTokensPerRun": tokens}
    if getattr(config, "auth_token", ""):
        card["securitySchemes"] = {
            "bearer": {"type": "http", "scheme": "bearer", "description": "AGENT_AUTH_TOKEN"}
        }
        card["security"] = [{"bearer": []}]
    return card


class A2AHandler:
    """把 A2A JSON-RPC 请求翻译成本机 runtime 调用。"""

    def __init__(self, runtime: Any, base_url: str) -> None:
        self.runtime = runtime
        self.base_url = base_url
        self._background: set[asyncio.Task[Any]] = set()
        self._contexts: dict[str, str] = {}  # contextId -> 最近一次 taskId

    # ---- 卡片 -------------------------------------------------------
    def card(self) -> dict[str, Any]:
        return build_agent_card(self.runtime.config, self.runtime, self.base_url)

    # ---- 请求入口 ---------------------------------------------------
    async def handle(self, payload: Any) -> dict[str, Any]:
        rid = payload.get("id") if isinstance(payload, dict) else None
        try:
            if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
                raise A2AError(INVALID_REQUEST, "仅接受 JSON-RPC 2.0")
            method = str(payload.get("method") or "")
            params = payload.get("params") or {}
            if method == "message/send":
                result = await self._message(params, blocking=None)
            elif method == "tasks/get":
                result = self._get_task(params)
            elif method == "tasks/cancel":
                result = self._cancel(params)
            elif method in {"agent/getCard", "agent/get_card"}:
                result = self.card()
            else:
                raise A2AError(METHOD_NOT_FOUND, f"不支持的方法：{method}")
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        except A2AError as err:
            error: dict[str, Any] = {"code": err.code, "message": err.message}
            if err.data is not None:
                error["data"] = err.data
            return {"jsonrpc": "2.0", "id": rid, "error": error}
        except Exception as err:  # noqa: BLE001 - JSON-RPC 必须回错误而不是 500
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": INTERNAL_ERROR, "message": f"{type(err).__name__}: {err}"},
            }

    async def stream(self, payload: Any) -> AsyncIterator[dict[str, Any]]:
        """``message/stream``：按 runtime 事件发 status-update，结束时发最终 Task。"""
        rid = payload.get("id") if isinstance(payload, dict) else None
        params = payload.get("params") or {}
        text, session_id, context_id, history = self._prepare(params)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)

        def emit(event: dict[str, Any]) -> None:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

        async def runner() -> None:
            try:
                await self.runtime.start_run(text, session_id, on_event=emit)
            except Exception as err:  # noqa: BLE001 - 失败落会话并照常收尾
                self.runtime.sessions.finish(session_id, f"{type(err).__name__}: {err}", "failed")
            finally:
                try:
                    queue.put_nowait({"type": "__done__"})
                except asyncio.QueueFull:
                    pass

        task = asyncio.create_task(runner())
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        while True:
            event = await queue.get()
            if event.get("type") == "__done__":
                break
            yield self._status_event(rid, session_id, context_id, event)
        info = self.runtime.sessions.get(session_id) or {}
        yield {
            "jsonrpc": "2.0",
            "id": rid,
            "result": task_from_session(info, context_id, history=history),
        }

    def _status_event(
        self, rid: Any, session_id: str, context_id: str, event: dict[str, Any]
    ) -> dict[str, Any]:
        text = _event_text(event)
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "kind": "status-update",
                "taskId": session_id,
                "contextId": context_id,
                "status": {
                    "state": "working",
                    "timestamp": _iso(time.time()),
                    "message": {
                        "role": "agent",
                        "messageId": uuid.uuid4().hex[:12],
                        "parts": [{"kind": "text", "text": text}],
                    },
                },
                "final": False,
            },
        }

    # ---- message/send -----------------------------------------------
    def _prepare(
        self, params: dict[str, Any]
    ) -> tuple[str, str, str, list[dict[str, Any]]]:
        message = params.get("message") if isinstance(params, dict) else None
        if not isinstance(message, dict):
            raise A2AError(INVALID_PARAMS, "缺少 message")
        text = parts_to_text(message.get("parts"))
        if not text:
            raise A2AError(INVALID_PARAMS, "message.parts 里没有可用文本")

        task_id = str(message.get("taskId") or params.get("taskId") or "").strip()
        context_id = str(message.get("contextId") or task_id or uuid.uuid4().hex[:12])
        if task_id:
            session = self.runtime.sessions.get(task_id)
            if session is None:
                raise A2AError(TASK_NOT_FOUND, f"task 不存在：{task_id}")
            if session.get("status") == "running":
                raise A2AError(
                    INVALID_PARAMS, f"task 仍在运行，先 tasks/get 或 tasks/cancel：{task_id}"
                )
            session_id = task_id
        else:
            session_id = self.runtime.sessions.create(text)
        self._contexts[context_id] = session_id
        return text, session_id, context_id, [self._echo_message(message)]

    async def _message(
        self,
        params: dict[str, Any],
        blocking: bool | None,
        on_event: Any = None,
    ) -> dict[str, Any]:
        text, session_id, context_id, history = self._prepare(params)
        configuration = params.get("configuration") or {}
        if blocking is None:
            blocking = bool(configuration.get("blocking", False))

        if blocking:
            try:
                await self.runtime.start_run(text, session_id, on_event=on_event)
            except Exception as err:  # noqa: BLE001 - 失败也按 Task.failed 回给对端
                self.runtime.sessions.finish(session_id, f"{type(err).__name__}: {err}", "failed")
            info = self.runtime.sessions.get(session_id) or {}
            return task_from_session(info, context_id, history=history)

        async def background() -> None:
            try:
                await self.runtime.start_run(text, session_id)
            except Exception as err:  # noqa: BLE001 - 后台失败落会话状态
                self.runtime.sessions.finish(session_id, f"{type(err).__name__}: {err}", "failed")

        task = asyncio.create_task(background())
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        info = self.runtime.sessions.get(session_id) or {}
        return task_from_session(info, context_id, history=history)

    def _echo_message(self, message: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "user",
            "messageId": str(message.get("messageId") or uuid.uuid4().hex[:12]),
            "parts": message.get("parts") or [],
        }

    # ---- tasks/get, tasks/cancel ------------------------------------
    def _get_task(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = str((params or {}).get("id") or (params or {}).get("taskId") or "").strip()
        if not task_id:
            raise A2AError(INVALID_PARAMS, "缺少 id")
        info = self.runtime.sessions.get(task_id)
        if info is None:
            raise A2AError(TASK_NOT_FOUND, f"task 不存在：{task_id}")
        context_id = next((c for c, t in self._contexts.items() if t == task_id), task_id)
        return task_from_session(info, context_id)

    def _cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = str((params or {}).get("id") or (params or {}).get("taskId") or "").strip()
        if not task_id:
            raise A2AError(INVALID_PARAMS, "缺少 id")
        info = self.runtime.sessions.get(task_id)
        if info is None:
            raise A2AError(TASK_NOT_FOUND, f"task 不存在：{task_id}")
        if info.get("status") != "running":
            raise A2AError(TASK_NOT_CANCELABLE, f"task 已结束：{task_id}")
        if not self.runtime.cancel_session(task_id):
            raise A2AError(TASK_NOT_CANCELABLE, f"运行态里找不到该 task（可能已自行结束）：{task_id}")
        context_id = next((c for c, t in self._contexts.items() if t == task_id), task_id)
        return task_from_session(self.runtime.sessions.get(task_id) or info, context_id)


def _event_text(event: dict[str, Any]) -> str:
    kind = str(event.get("type") or "")
    if kind == "tool_call":
        return f"调用工具 {event.get('name') or event.get('tool') or ''}".strip()
    if kind == "tool_result":
        return f"工具返回 {str(event.get('result') or '')[:200]}".strip()
    if kind == "finish":
        return str(event.get("answer") or "完成")
    if kind == "step":
        return f"第 {event.get('step')} 步"
    return kind or "working"
