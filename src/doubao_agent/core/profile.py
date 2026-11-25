"""关于你：稳定画像（不是教训库）。

longterm 记的是"这件事做错在哪"，这里记的是"你是谁、你要什么"：
称呼、地域、作息、长期目标、口味、工作口径、禁忌与红线。
每条都带来源、置信、更新时间；冲突只留最新一条，旧值进历史表，可追溯、可撤回。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from .db import connect as connect_db

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profile_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'user',
    confidence REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'active',
    note TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS profile_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    old_value TEXT NOT NULL,
    new_value TEXT NOT NULL,
    changed_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_profile_history_key ON profile_history(key, changed_at);
"""

SENSITIVE_KEYS = ("password", "token", "secret", "key", "密码", "密钥", "凭证")


class ProfileStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = connect_db(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def set(
        self,
        key: str,
        value: str,
        source: str = "user",
        confidence: float = 1.0,
        note: str = "",
        now: float | None = None,
    ) -> dict[str, Any]:
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError("键和值都不能为空")
        if any(word in key.lower() for word in SENSITIVE_KEYS):
            raise ValueError("画像不存凭证类信息，请走凭据保管")
        now = time.time() if now is None else now
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM profile_entries WHERE key = ?", (key,)
            ).fetchone()
            if row and row["value"] != value:
                self._conn.execute(
                    "INSERT INTO profile_history(key, old_value, new_value, changed_at) VALUES (?, ?, ?, ?)",
                    (key, row["value"], value, now),
                )
            self._conn.execute(
                "INSERT INTO profile_entries(key, value, source, confidence, status, note, updated_at) "
                "VALUES (?, ?, ?, ?, 'active', ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, source = excluded.source, "
                "confidence = excluded.confidence, status = 'active', note = excluded.note, "
                "updated_at = excluded.updated_at",
                (key, value, source.strip() or "user", float(confidence), note.strip(), now),
            )
            self._conn.commit()
        return self.get(key) or {}

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM profile_entries WHERE key = ?", (key.strip(),)
            ).fetchone()
        return dict(row) if row else None

    def list(self, status: str = "active", limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            if status == "all":
                rows = self._conn.execute(
                    "SELECT * FROM profile_entries ORDER BY updated_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM profile_entries WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def forget(self, key: str) -> bool:
        """停用而不是硬删：留痕，下一次不再注入。"""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE profile_entries SET status = 'forgotten', updated_at = ? WHERE key = ?",
                (time.time(), key.strip()),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def history(self, key: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM profile_history WHERE key = ? ORDER BY changed_at DESC LIMIT ?",
                (key.strip(), limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def context_text(self, limit: int = 10, max_chars: int = 600) -> str:
        """注入系统提示的一行画像。没有就返回空串，不占预算。"""
        rows = self.list("active", limit)
        if not rows:
            return ""
        parts = [f"{row['key']}={row['value']}" for row in rows]
        prefix = "关于用户（画像，按此口径）："
        text = "；".join(parts)
        budget = max(20, max_chars - len(prefix) - 1)
        if len(text) > budget:
            text = text[:budget].rstrip("；") + "…"
        return prefix + text

    def close(self) -> None:
        with self._lock:
            self._conn.close()
