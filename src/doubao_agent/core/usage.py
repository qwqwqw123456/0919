from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UsageMeter:
    """进程级调用计量：调用次数 + token（provider 返回 usage 时才计 token）。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def record(self, usage: dict[str, Any] | None) -> None:
        self.calls += 1
        if not usage:
            return
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)

    def snapshot(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }
