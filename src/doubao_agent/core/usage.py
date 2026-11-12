from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UsageMeter:
    """进程级调用计量：调用次数 + token，并可设置预算上限（0 表示不限）。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    max_calls: int = 0
    max_tokens: int = 0

    def record(self, usage: dict[str, Any] | None) -> None:
        self.calls += 1
        if not usage:
            return
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)

    def exceeded(self) -> str | None:
        """超出预算时返回原因，未超限返回 None。"""
        if self.max_calls and self.calls >= self.max_calls:
            return f"调用预算已用尽（{self.calls}/{self.max_calls} 次）"
        if self.max_tokens and self.prompt_tokens + self.completion_tokens >= self.max_tokens:
            total = self.prompt_tokens + self.completion_tokens
            return f"token 预算已用尽（{total}/{self.max_tokens}）"
        return None

    def snapshot(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "max_calls": self.max_calls,
            "max_tokens": self.max_tokens,
        }
