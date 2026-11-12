from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    result TEXT,
    step INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    state TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    path TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'file',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_cp_session ON checkpoints(session_id);
"""


class SessionStore:
    """SQLite-backed sessions, messages and checkpoints.

    Everything is written to disk so a reclaimed cloud desktop can resume.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _now(self) -> float:
        return time.time()

    def create(self, task: str) -> str:
        session_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(id, task, status, created_at, updated_at) "
                "VALUES (?, ?, 'running', ?, ?)",
                (session_id, task, self._now(), self._now()),
            )
            self._conn.commit()
        return session_id

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def touch(self, session_id: str, step: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET step = ?, updated_at = ? WHERE id = ?",
                (step, self._now(), session_id),
            )
            self._conn.commit()

    def finish(self, session_id: str, result: str, status: str = "completed") -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET status = ?, result = ?, updated_at = ? WHERE id = ?",
                (status, result, self._now(), session_id),
            )
            self._conn.commit()

    def append_message(self, session_id: str, step: int, payload: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages(session_id, step, payload, created_at) VALUES (?, ?, ?, ?)",
                (session_id, step, json.dumps(payload, ensure_ascii=False), self._now()),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (self._now(), session_id)
            )
            self._conn.commit()

    def messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def save_checkpoint(self, session_id: str, step: int, state: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO checkpoints(session_id, step, state, created_at) VALUES (?, ?, ?, ?)",
                (session_id, step, json.dumps(state, ensure_ascii=False), self._now()),
            )
            self._conn.execute(
                "UPDATE sessions SET step = ?, updated_at = ? WHERE id = ?",
                (step, self._now(), session_id),
            )
            self._conn.commit()

    def latest_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT step, state, created_at FROM checkpoints WHERE session_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return {"step": row["step"], "state": json.loads(row["state"]), "created_at": row["created_at"]}

    def add_artifact(self, session_id: str, path: str, kind: str = "file") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO artifacts(session_id, path, kind, created_at) VALUES (?, ?, ?, ?)",
                (session_id, path, kind, self._now()),
            )
            self._conn.commit()

    def artifacts(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, kind, created_at FROM artifacts WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
