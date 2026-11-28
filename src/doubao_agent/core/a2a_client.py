"""A2A 客户端：调别的 Agent。

只做协议本身该做的事：拉 Agent Card 认能力、发 message/send、按需轮询 tasks/get、必要时 tasks/cancel。
安全口径：
- 只允许 http/https，拒绝带 userinfo 的 URL（防止把凭据塞进地址里被日志带走）；
- 出站地址必须在白名单内，默认只有回环（127.0.0.1 / localhost / ::1），要调外部 Agent 得显式加白名单。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any
from urllib.parse import urlsplit

import httpx

LOOPBACK_ALLOW = ["127.0.0.1", "localhost", "::1"]
TERMINAL_STATES = {"completed", "failed", "canceled", "rejected", "unknown"}


class A2AError(Exception):
    pass


def _host_port(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise A2AError(f"只支持 http/https，收到：{parts.scheme or '空'}")
    if parts.username or parts.password:
        raise A2AError("URL 里不允许带用户名/密码")
    host = (parts.hostname or "").strip().lower()
    if not host:
        raise A2AError("URL 缺少主机名")
    try:
        port = parts.port
    except ValueError as exc:
        raise A2AError(f"端口不合法：{exc}") from exc
    default_port = 443 if parts.scheme == "https" else 80
    return host, parts.scheme, port or default_port


def _match(host: str, port: int, entry: str) -> bool:
    entry = entry.strip().lower()
    if not entry:
        return False
    if entry == "*":
        return True
    if ":" in entry and not entry.endswith("]"):  # host:port 精确匹配
        host_part, _, port_part = entry.rpartition(":")
        if port_part.isdigit() and int(port_part) != port:
            return False
        entry = host_part
    entry = entry.strip("[]")
    if entry.startswith("*."):
        return host.endswith(entry[1:]) and host != entry[2:]
    return host == entry


def check_endpoint(url: str, allow: list[str] | None = None) -> str:
    """校验并规整 A2A 端点；不在白名单则抛 A2AError。"""
    host, _scheme, port = _host_port(url)
    allowlist = allow if allow is not None else LOOPBACK_ALLOW
    if not any(_match(host, port, entry) for entry in allowlist):
        raise A2AError(f"目标不在白名单：{host}:{port}（当前允许：{', '.join(allowlist) or '空'}）")
    endpoint = url.split("#", 1)[0].rstrip("/")
    if not endpoint.endswith("/a2a"):
        endpoint = endpoint + "/a2a"
    return endpoint


def _card_url(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    return f"{parts.scheme}://{parts.netloc}/.well-known/agent-card.json"


def _parts_text(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    out: list[str] = []
    for part in parts:
        if isinstance(part, dict) and str(part.get("kind") or part.get("type") or "") == "text":
            out.append(str(part.get("text") or ""))
    return "\n".join(t for t in out if t)


def task_text(task: dict[str, Any]) -> str:
    """把一个 Task 里的 artifacts 摊平成文本。"""
    out: list[str] = []
    for artifact in task.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        text = _parts_text(artifact.get("parts"))
        if text:
            out.append(text)
    return "\n".join(out)


class A2AClient:
    """一个远端 Agent 的客户端封装；用完记得 ``aclose``。"""

    def __init__(
        self,
        url: str,
        token: str = "",
        allow: list[str] | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
        policy: Any = None,
    ) -> None:
        self.endpoint = check_endpoint(url, allow)
        self.card_url = _card_url(self.endpoint)
        self.token = token
        self.policy = policy
        if policy is not None and transport is None:
            timeout = policy.model.timeout
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport, follow_redirects=False)

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.token}"} if self.token else {}

    async def card(self) -> dict[str, Any]:
        resp = await self._client.get(self.card_url, headers=self._headers())
        if resp.status_code != 200:
            raise A2AError(f"拉取 Agent Card 失败：HTTP {resp.status_code}")
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise A2AError(f"Agent Card 不是 JSON：{exc}") from exc

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": uuid.uuid4().hex[:8], "method": method, "params": params}
        try:
            resp = await self._client.post(self.endpoint, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise A2AError(f"请求失败：{type(exc).__name__}: {exc}") from exc
        if resp.status_code == 401:
            raise A2AError("对端要求鉴权，401（需要带 token）")
        if resp.status_code >= 400:
            raise A2AError(f"对端 HTTP {resp.status_code}")
        try:
            body = resp.json()
        except json.JSONDecodeError as exc:
            raise A2AError(f"对端返回不是 JSON：{exc}") from exc
        if body.get("error"):
            err = body["error"]
            raise A2AError(f"{err.get('message') or '对端报错'}（code={err.get('code')}）")
        result = body.get("result")
        if not isinstance(result, dict):
            raise A2AError("对端 result 不是对象")
        return result

    async def send(
        self,
        text: str,
        blocking: bool = False,
        wait: bool = True,
        poll_interval: float = 2.0,
        max_wait: float = 600.0,
        context_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """发一条消息；非 blocking 时默认轮询到终态后返回 Task。"""
        text = (text or "").strip()
        if not text:
            raise A2AError("任务文本为空")
        message: dict[str, Any] = {
            "role": "user",
            "messageId": uuid.uuid4().hex[:12],
            "parts": [{"kind": "text", "text": text}],
        }
        if context_id:
            message["contextId"] = context_id
        if task_id:
            message["taskId"] = task_id
        task = await self._rpc(
            "message/send", {"message": message, "configuration": {"blocking": blocking}}
        )
        if blocking or not wait or task.get("status", {}).get("state") in TERMINAL_STATES:
            return task
        return await self.wait(str(task["id"]), poll_interval=poll_interval, max_wait=max_wait)

    async def get(self, task_id: str) -> dict[str, Any]:
        return await self._rpc("tasks/get", {"id": task_id})

    async def wait(
        self, task_id: str, poll_interval: float | None = None, max_wait: float | None = None
    ) -> dict[str, Any]:
        spec = self.policy.model if self.policy is not None else None
        if poll_interval is None:
            poll_interval = spec.poll_interval if spec is not None else 2.0
        if max_wait is None:
            max_wait = spec.poll_timeout if spec is not None and spec.poll_timeout else 600.0
        deadline = time.monotonic() + max_wait
        round_i = 0
        while True:
            task = await self.get(task_id)
            if str(task.get("status", {}).get("state") or "") in TERMINAL_STATES:
                return task
            if time.monotonic() >= deadline:
                raise A2AError(f"等待超时（{int(max_wait)}s），task 仍未结束：{task_id}")
            # 前 6 轮按固定间隔快查（短任务要快）；之后才逐轮放大到 poll_max（长任务别空转）。
            if spec is not None:
                grown = poll_interval if round_i < 6 else min(
                    spec.poll_max, poll_interval * (1.6 ** (round_i - 5))
                )
            else:
                grown = poll_interval
            await asyncio.sleep(max(0.2, grown))
            round_i += 1

    async def cancel(self, task_id: str) -> dict[str, Any]:
        return await self._rpc("tasks/cancel", {"id": task_id})

    async def aclose(self) -> None:
        await self._client.aclose()


def summarize_card(card: dict[str, Any], limit: int = 12) -> str:
    skills = card.get("skills") or []
    names = [str(s.get("name") or s.get("id") or "") for s in skills[:limit] if isinstance(s, dict)]
    caps = card.get("capabilities") or {}
    lines = [
        f"名称：{card.get('name') or '-'}",
        f"版本：{card.get('version') or '-'}（协议 {card.get('protocolVersion') or '-'}）",
        f"地址：{card.get('url') or '-'}",
        f"流式：{'是' if caps.get('streaming') else '否'} · 推送：{'是' if caps.get('pushNotifications') else '否'}",
        f"技能（{len(skills)}）：{', '.join(n for n in names if n) or '未声明'}",
    ]
    desc = str(card.get("description") or "").strip()
    if desc:
        lines.insert(1, f"描述：{desc}")
    return "\n".join(lines)


def summarize_task(task: dict[str, Any], limit: int = 2000) -> str:
    state = str(task.get("status", {}).get("state") or "unknown")
    body = task_text(task)
    if not body:
        msg = task.get("status", {}).get("message") or {}
        body = _parts_text(msg.get("parts")) if isinstance(msg, dict) else ""
    if len(body) > limit:
        body = body[:limit] + "…"
    return f"task {task.get('id')} · 状态 {state}\n{body or '（无文本产物）'}"
