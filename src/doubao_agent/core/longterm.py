from __future__ import annotations

import sqlite3
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

# 第二层：三张长期经验表。字段固定，语义与用户口径一一对应。
TABLES: dict[str, tuple[str, ...]] = {
    "lessons": ("date", "event", "judgment", "outcome", "lesson"),
    "observations": ("date", "phenomenon", "observation", "cause", "followup"),
    "deviations": ("date", "expected", "actual", "gap", "cause", "fix"),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    event TEXT NOT NULL,
    judgment TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT '',
    lesson TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    phenomenon TEXT NOT NULL,
    observation TEXT NOT NULL DEFAULT '',
    cause TEXT NOT NULL DEFAULT '',
    followup TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS deviations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    expected TEXT NOT NULL,
    actual TEXT NOT NULL,
    gap TEXT NOT NULL DEFAULT '',
    cause TEXT NOT NULL DEFAULT '',
    fix TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lessons_date ON lessons(date);
CREATE INDEX IF NOT EXISTS idx_observations_date ON observations(date);
CREATE INDEX IF NOT EXISTS idx_deviations_date ON deviations(date);
"""


def _today(value: str | None) -> str:
    return (value or "").strip() or date.today().isoformat()


def _gap(expected: str, actual: str, provided: str | None) -> str:
    """偏差值：显式给就用给的；两边都是数字时算差值，否则留空。"""
    if provided:
        return provided.strip()
    try:
        return f"{float(expected) - float(actual):+g}"
    except (TypeError, ValueError):
        return ""


class LongTermMemory:
    """长期记忆：萃取库 / 观察记录 / 偏差档案三张表。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _add(self, table: str, payload: dict[str, Any]) -> int:
        columns = TABLES[table]
        values = [payload.get(column, "") for column in columns]
        placeholders = ", ".join("?" for _ in columns)
        with self._lock:
            cursor = self._conn.execute(
                f"INSERT INTO {table}({', '.join(columns)}, created_at) "
                f"VALUES ({placeholders}, ?)",
                (*values, time.time()),
            )
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    def _list(self, table: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(row) for row in rows]

    def add_lesson(
        self,
        event: str,
        judgment: str = "",
        outcome: str = "",
        lesson: str = "",
        date_str: str | None = None,
    ) -> int:
        return self._add(
            "lessons",
            {
                "date": _today(date_str),
                "event": event,
                "judgment": judgment,
                "outcome": outcome,
                "lesson": lesson,
            },
        )

    def add_observation(
        self,
        phenomenon: str,
        observation: str = "",
        cause: str = "",
        followup: str = "",
        date_str: str | None = None,
    ) -> int:
        return self._add(
            "observations",
            {
                "date": _today(date_str),
                "phenomenon": phenomenon,
                "observation": observation,
                "cause": cause,
                "followup": followup,
            },
        )

    def add_deviation(
        self,
        expected: str,
        actual: str,
        gap: str | None = None,
        cause: str = "",
        fix: str = "",
        date_str: str | None = None,
    ) -> int:
        return self._add(
            "deviations",
            {
                "date": _today(date_str),
                "expected": expected,
                "actual": actual,
                "gap": _gap(expected, actual, gap),
                "cause": cause,
                "fix": fix,
            },
        )

    def lessons(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._list("lessons", limit)

    def observations(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._list("observations", limit)

    def deviations(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._list("deviations", limit)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                table: int(
                    self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in TABLES
            }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
