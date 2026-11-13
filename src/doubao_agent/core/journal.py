from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step INTEGER NOT NULL DEFAULT 0,
    tool TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 1,
    elapsed_ms INTEGER NOT NULL DEFAULT 0,
    preview TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_runs_session ON tool_runs(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_runs_key ON tool_runs(session_id, args_hash);
"""

_PREVIEW_LIMIT = 2000


class ToolJournal:
    """工具执行日志：按 (工具, 参数) 指纹去重，支撑断点续跑与幂等重放。

    只记录，不判断业务语义；是否允许重放由调用方按风险等级决定。
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @staticmethod
    def key(tool: str, arguments: dict[str, Any]) -> str:
        """参数规范化后取 sha1，避免 dict 顺序不同被当成不同调用。"""
        payload = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha1(f"{tool}:{payload}".encode("utf-8")).hexdigest()

    def succeeded(self, session_id: str, args_hash: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tool_runs WHERE session_id = ? AND args_hash = ? "
                "AND status = 'ok' ORDER BY id DESC LIMIT 1",
                (session_id, args_hash),
            ).fetchone()
        if row is None:
            return None
        return {
            "tool": row["tool"],
            "step": int(row["step"]),
            "attempts": int(row["attempts"]),
            "elapsed_ms": int(row["elapsed_ms"]),
            "preview": row["preview"],
        }

    def record(
        self,
        session_id: str,
        tool: str,
        args_hash: str,
        *,
        status: str,
        step: int = 0,
        attempts: int = 1,
        elapsed_ms: int = 0,
        preview: str = "",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO tool_runs(session_id, step, tool, args_hash, status, attempts, "
                "elapsed_ms, preview, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    step,
                    tool,
                    args_hash,
                    status,
                    attempts,
                    max(0, int(elapsed_ms)),
                    (preview or "")[:_PREVIEW_LIMIT],
                    time.time(),
                ),
            )
            self._conn.commit()

    def entries(self, session_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tool_runs WHERE session_id = ? ORDER BY id LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [
            {
                "step": int(row["step"]),
                "tool": row["tool"],
                "status": row["status"],
                "attempts": int(row["attempts"]),
                "elapsed_ms": int(row["elapsed_ms"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
