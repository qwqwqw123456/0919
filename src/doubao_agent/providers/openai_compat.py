from __future__ import annotations

import json
import os
from typing import Any

import httpx


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        key = api_key or os.environ.get(api_key_env, "")
        headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=120.0, trust_env=True
        )

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

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        message = body["choices"][0]["message"]

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
