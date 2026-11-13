from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageMeter:
    """调用计量：可派生 run 级子计量，子计量独立判预算，父级只做累计统计。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    max_calls: int = 0
    max_tokens: int = 0
    parent: "UsageMeter | None" = field(default=None, repr=False, compare=False)

    def child(self) -> "UsageMeter":
        """派生一个预算相同但计数独立的子计量，用于单次运行/单个子任务。"""
        return UsageMeter(
            max_calls=self.max_calls,
            max_tokens=self.max_tokens,
            parent=self,
        )

    def record(self, usage: dict[str, Any] | None) -> None:
        meter: UsageMeter | None = self
        while meter is not None:
            meter._add(usage)
            meter = meter.parent

    def _add(self, usage: dict[str, Any] | None) -> None:
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
