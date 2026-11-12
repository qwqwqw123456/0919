from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils import dumps


_SECRET_KEYS = {
    "token",
    "authorization",
    "password",
    "passwd",
    "secret",
    "cookie",
    "credential",
    "api_key",
    "apikey",
    "private_key",
}
_SECRET_SUFFIXES = ("_token", "_secret", "_password", "_api_key", "_apikey", "_private_key")


def redact(value: Any) -> Any:
    """把疑似凭据的字段替换成 ***，避免 token / 密钥落进审计日志。"""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _SECRET_KEYS or lowered.endswith(_SECRET_SUFFIXES):
                out[key] = "***"
            else:
                out[key] = redact(item)
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


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
        record: dict[str, Any] = redact(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "session": session_id,
                "event": event,
                **payload,
            }
        )
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

    def sessions(self) -> list[str]:
        """列出有审计记录的会话，按最近修改时间倒序。"""
        if not self.runs_dir.exists():
            return []
        paths = [p for p in self.runs_dir.iterdir() if (p / "audit.jsonl").is_file()]
        paths.sort(key=lambda p: (p / "audit.jsonl").stat().st_mtime, reverse=True)
        return [p.name for p in paths]

    def verify(self, session_id: str) -> tuple[bool, int]:
        """校验哈希链：逐条重算，任何一条被改过或断链都返回 False。"""
        path = self._path(session_id)
        if not path.exists():
            return True, 0
        prev = "genesis"
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    return False, count
                if not isinstance(record, dict):
                    return False, count
                stored_prev = record.pop("prev", None)
                stored_hash = record.pop("hash", None)
                expected = hashlib.sha256(
                    (prev + dumps(record, sort_keys=True)).encode("utf-8")
                ).hexdigest()[:16]
                if stored_prev != prev or stored_hash != expected:
                    return False, count
                prev = expected
                count += 1
        return True, count
