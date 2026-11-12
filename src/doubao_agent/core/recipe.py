from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recipes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    steps TEXT NOT NULL,
    session_id TEXT,
    uses INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
"""


class RecipeStore:
    """Records successful tool sequences so repeated tasks replay with zero model calls."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, task: str, steps: list[dict[str, Any]], session_id: str | None = None) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO recipes(task, steps, session_id, created_at) VALUES (?, ?, ?, ?)",
                (task, json.dumps(steps, ensure_ascii=False), session_id, time.time()),
            )
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, task, steps, session_id, uses, created_at FROM recipes "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["step_count"] = len(json.loads(item.pop("steps")))
            out.append(item)
        return out

    def get(self, recipe_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM recipes WHERE id = ?", (recipe_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        item["steps"] = json.loads(item["steps"])
        return item

    def mark_used(self, recipe_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE recipes SET uses = uses + 1 WHERE id = ?", (recipe_id,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
