from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx


class OpenAICompatProvider:
    """OpenAI 兼容端点。

    超时、重试、退避、连接池参数全部来自 `core.netpolicy.Policy`（policy=None 时用内置默认）。
    provider 只负责「同一次请求内的瞬时错误」（网络错、429、5xx），
    模型级重试交给 Agent._complete，两层不叠加放大。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        extra_headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        policy: Any = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.policy = policy
        key = api_key or os.environ.get(api_key_env, "")
        headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if client is not None:
            self._client = client
        else:
            snap = policy.snapshot() if policy is not None else None
            n = snap["net"] if snap else None
            limits = (
                httpx.Limits(
                    max_keepalive_connections=n.keepalive,
                    max_connections=n.max_connections,
                )
                if n is not None
                else httpx.Limits()
            )
            timeout = (
                policy.httpx_timeout("model") if policy is not None else httpx.Timeout(120.0)
            )
            self._client = httpx.AsyncClient(
                base_url=base_url.rstrip("/"),
                headers=headers,
                timeout=timeout,
                trust_env=True,
                limits=limits,
                http2=bool(n.http2) if n is not None else False,
            )

    def _spec(self) -> tuple[Any, Any]:
        if self.policy is None:
            return None, None
        snap = self.policy.snapshot()
        return snap["net"], snap["model"]

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        net, model = self._spec()
        attempts = (net.retries + 1) if net is not None else 1
        timeout = self.policy.httpx_timeout("model") if self.policy is not None else None
        last: Exception | None = None
        started = time.perf_counter()
        for attempt in range(attempts):
            try:
                kwargs: dict[str, Any] = {"json": payload}
                if timeout is not None:
                    kwargs["timeout"] = timeout
                response = await self._client.post("/chat/completions", **kwargs)
                if response.status_code >= 400:
                    retryable = net is not None and response.status_code in net.retry_status
                    # 用异常把状态码带上，避免上层只能看字符串
                    err = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}", request=response.request, response=response
                    )
                    if retryable and attempt + 1 < attempts:
                        last = err
                        await asyncio.sleep(self.policy.sleep_backoff("net", attempt + 1))
                        continue
                    raise err
                body = response.json()
                if self.policy is not None:
                    self.policy.record("model", time.perf_counter() - started, ok=True)
                break
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last = exc
                timed_out = isinstance(exc, httpx.TimeoutException)
                can_retry = not timed_out or (net is not None and net.retry_on_timeout)
                if can_retry and attempt + 1 < attempts and self.policy is not None:
                    await asyncio.sleep(self.policy.sleep_backoff("net", attempt + 1))
                    continue
                if self.policy is not None:
                    self.policy.record("model", time.perf_counter() - started, ok=False)
                raise
        else:  # pragma: no cover - 上面的分支都会 break/raise
            if last is not None:
                raise last

        choices = body.get("choices") or []
        first = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}

        tool_calls: list[dict[str, Any]] = []
        for call in message.get("tool_calls") or []:
            raw_args = call["function"].get("arguments") or "{}"
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = {"_raw": raw_args}
            tool_calls.append(
                {"id": call.get("id", ""), "name": call["function"]["name"], "arguments": arguments}
            )
        return {
            "content": message.get("content"),
            "tool_calls": tool_calls,
            "usage": body.get("usage"),
        }

    async def aclose(self) -> None:
        await self._client.aclose()
