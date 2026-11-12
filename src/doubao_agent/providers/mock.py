from __future__ import annotations

from typing import Any


class MockProvider:
    """Deterministic provider for smoke tests and offline demos.

    Feed it a script of OpenAI-shaped assistant messages; once exhausted it
    returns a final answer.
    """

    def __init__(self, script: list[dict[str, Any]] | None = None) -> None:
        self.script = list(script or [])

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if self.script:
            return self.script.pop(0)
        return {"content": "（mock 模型）任务结束。", "tool_calls": []}

    async def aclose(self) -> None:
        return None
