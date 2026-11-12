"""任务状态持久化：支持中断后 resume。"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    intent TEXT NOT NULL,
    status TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    report_json TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


class TaskStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create(self, intent: str, plan: dict[str, Any], status: str = "planned") -> str:
        task_id = "T-" + uuid.uuid4().hex[:8]
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO tasks(id, intent, status, plan_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, intent, status, json.dumps(plan, ensure_ascii=False), now, now),
            )
            self._conn.commit()
        return task_id

    def update(self, task_id: str, status: str, report: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status = ?, report_json = ?, updated_at = ? WHERE id = ?",
                (status, json.dumps(report, ensure_ascii=False) if report is not None else None,
                 time.time(), task_id),
            )
            self._conn.commit()

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(r) for r in rows]

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["plan"] = json.loads(item.pop("plan_json"))
        report = item.pop("report_json")
        item["report"] = json.loads(report) if report else None
        return item

    def close(self) -> None:
        with self._lock:
            self._conn.close()
