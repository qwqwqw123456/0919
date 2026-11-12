from __future__ import annotations

import json
from typing import Any

from ..sources.bridge import HostBridge


class HostProvider:
    """模型推理委托给豆包宿主，无需 API key。

    宿主以 capability `model.complete` 暴露推理能力，本 Provider 通过
    HostBridge.invoke 调用它。宿主返回 OpenAI 风格、简化风格或纯文本均可。
    """

    def __init__(
        self,
        bridge: HostBridge,
        capability: str = "model.complete",
        model: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.bridge = bridge
        self.capability = capability
        self.model = model
        self.temperature = temperature

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "temperature": self.temperature,
        }
        if self.model:
            payload["model"] = self.model
        result = await self.bridge.invoke(self.capability, payload)
        return self._normalize(result)

    async def aclose(self) -> None:
        # bridge 归 registry 统一关闭，此处不持有独立连接。
        return None

    @staticmethod
    def _coerce_args(arguments: Any) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except (ValueError, TypeError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @classmethod
    def _normalize(cls, result: Any) -> dict[str, Any]:
        if isinstance(result, str):
            return {"content": result, "tool_calls": []}
        if not isinstance(result, dict):
            return {"content": "", "tool_calls": []}

        content = result.get("content")
        tool_calls = result.get("tool_calls")

        # 兼容 OpenAI 风格 {"choices": [{"message": {...}}]}
        if content is None and tool_calls is None:
            message = result.get("message")
            if message is None:
                choices = result.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    message = choices[0].get("message")
            if isinstance(message, dict):
                content = message.get("content")
                tool_calls = message.get("tool_calls")
            elif isinstance(message, str):
                content = message

        if content is None and "text" in result:
            content = result["text"]

        normalized_calls: list[dict[str, Any]] = []
        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name")
            arguments = tc.get("arguments")
            fn = tc.get("function")
            if isinstance(fn, dict):
                name = name or fn.get("name")
                arguments = fn.get("arguments") if arguments is None else arguments
            if name is None:
                continue
            normalized_calls.append(
                {
                    "id": tc.get("id") or f"call_{len(normalized_calls)}",
                    "name": name,
                    "arguments": cls._coerce_args(arguments),
                }
            )

        return {"content": content if isinstance(content, str) else "", "tool_calls": normalized_calls}
