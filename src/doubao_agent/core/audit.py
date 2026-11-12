from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils import dumps


class AuditLog:
    """Append-only, hash-chained JSONL. Supports replay and tamper detection."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self._last: dict[str, str] = {}

    def _path(self, session_id: str) -> Path:
        return self.runs_dir / session_id / "audit.jsonl"

    def _prev_hash(self, path: Path) -> str:
        key = str(path)
        if key in self._last:
            return self._last[key]
        if not path.exists():
            return "genesis"
        last = ""
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = line
        if not last:
            return "genesis"
        try:
            return json.loads(last).get("hash", "genesis")
        except json.JSONDecodeError:
            return "genesis"

    def append(self, session_id: str, event: str, **payload: Any) -> dict[str, Any]:
        path = self._path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        prev = self._prev_hash(path)
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session": session_id,
            "event": event,
            **payload,
        }
        digest = hashlib.sha256(
            (prev + dumps(record, sort_keys=True)).encode("utf-8")
        ).hexdigest()[:16]
        record["prev"] = prev
        record["hash"] = digest
        with path.open("a", encoding="utf-8") as handle:
            handle.write(dumps(record) + "\n")
        self._last[str(path)] = digest
        return record

    def read(self, session_id: str, limit: int = 500) -> list[dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows[-limit:]
