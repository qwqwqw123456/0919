from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Callable


class RateLimiter:
    """按来源的滑动窗口限流：默认每分钟 N 次，`default` 作为兜底键。

    `limits` 形如 `{"browser": 30, "default": 120}`；未配置的来源不限流。
    """

    def __init__(
        self,
        limits: dict[str, int] | None = None,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limits = {str(k): int(v) for k, v in (limits or {}).items() if int(v) > 0}
        self.window = window
        self.clock = clock
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def limit_for(self, key: str) -> int:
        return self.limits.get(key, self.limits.get("default", 0))

    def allow(self, key: str) -> bool:
        limit = self.limit_for(key)
        if limit <= 0:
            return True
        now = self.clock()
        with self._lock:
            bucket = self._calls[key]
            while bucket and bucket[0] <= now - self.window:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True

    def remaining(self, key: str) -> int:
        limit = self.limit_for(key)
        if limit <= 0:
            return -1
        now = self.clock()
        with self._lock:
            bucket = self._calls[key]
            while bucket and bucket[0] <= now - self.window:
                bucket.popleft()
            return max(0, limit - len(bucket))
