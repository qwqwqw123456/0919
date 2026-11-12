"""独立决策审计：追加写、哈希链、可校验。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last = "genesis"
        if self.path.exists():
            last = ""
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        last = line
            if last:
                try:
                    self._last = json.loads(last).get("hash", "genesis")
                except json.JSONDecodeError:
                    self._last = "genesis"

    def record(self, event: str, **payload: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            **payload,
        }
        digest = hashlib.sha256((self._last + _dumps(record)).encode("utf-8")).hexdigest()[:16]
        record["prev"] = self._last
        record["hash"] = digest
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_dumps(record) + "\n")
        self._last = digest
        return record

    def read(self, limit: int = 500) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows[-limit:]

    def verify(self) -> tuple[bool, int]:
        """校验哈希链完整性，返回 (是否完好, 记录数)。"""
        prev = "genesis"
        count = 0
        if not self.path.exists():
            return True, 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                claimed = record.pop("hash", "")
                stored_prev = record.pop("prev", "")
                if stored_prev != prev:
                    return False, count
                digest = hashlib.sha256((prev + _dumps(record)).encode("utf-8")).hexdigest()[:16]
                if digest != claimed:
                    return False, count
                prev = claimed
                count += 1
        return True, count
