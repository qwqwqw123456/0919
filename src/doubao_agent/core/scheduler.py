from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    cron TEXT NOT NULL,
    task TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_run REAL,
    next_run REAL,
    created_at REAL NOT NULL
);
"""


def _expand(field: str, lo: int, hi: int) -> set[int]:
    """Expand one cron field. Supports *, n, a-b, */step, a,b, a-b/step."""
    out: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            part, step_s = part.split("/", 1)
            step = int(step_s)
            if step <= 0:
                raise ValueError(f"cron step must be positive: {field}")
        if part == "*":
            lo_v, hi_v = lo, hi
        elif "-" in part:
            a, b = part.split("-", 1)
            lo_v, hi_v = int(a), int(b)
        else:
            lo_v = hi_v = int(part)
        if lo_v < lo or hi_v > hi or lo_v > hi_v:
            raise ValueError(f"cron field out of range: {field}")
        for value in range(lo_v, hi_v + 1):
            if (value - lo) % step == 0:
                out.add(value)
    return out


def parse_cron(cron: str) -> dict[str, set[int]] | None:
    fields = cron.split()
    if len(fields) != 5:
        return None
    try:
        return {
            "minute": _expand(fields[0], 0, 59),
            "hour": _expand(fields[1], 0, 23),
            "dom": _expand(fields[2], 1, 31),
            "month": _expand(fields[3], 1, 12),
            "dow": _expand(fields[4], 0, 6),  # 0=Sunday .. 6=Saturday
        }
    except ValueError:
        return None


def next_run(cron: str, after: float | None = None) -> float | None:
    """Return the next UTC epoch satisfying `cron`, strictly after `after`."""
    spec = parse_cron(cron)
    if spec is None:
        return None
    after = time.time() if after is None else after
    dt = datetime.fromtimestamp(after).replace(second=0, microsecond=0) + timedelta(minutes=1)
    deadline = datetime.fromtimestamp(after) + timedelta(days=730)
    while dt <= deadline:
        if (
            dt.minute in spec["minute"]
            and dt.hour in spec["hour"]
            and dt.day in spec["dom"]
            and dt.month in spec["month"]
            and ((dt.weekday() + 1) % 7) in spec["dow"]
        ):
            return dt.timestamp()
        dt += timedelta(minutes=1)
    return None


class ScheduleStore:
    """Cron-triggered tasks. The runtime polls `due()` and dispatches to the agent."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add(self, name: str, cron: str, task: str) -> int:
        when = next_run(cron)
        if when is None:
            raise ValueError(f"invalid cron expression: {cron}")
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO schedules(name, cron, task, next_run, created_at) VALUES (?, ?, ?, ?, ?)",
                (name, cron, task, when, time.time()),
            )
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM schedules ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, schedule_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM schedules WHERE id = ?", (schedule_id,)
            ).fetchone()
        return dict(row) if row else None

    def remove(self, schedule_id: int) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
            self._conn.commit()
        return cursor.rowcount > 0

    def set_enabled(self, schedule_id: int, enabled: bool) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE schedules SET enabled = ? WHERE id = ?", (1 if enabled else 0, schedule_id)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def due(self, now: float | None = None) -> list[dict[str, Any]]:
        now = time.time() if now is None else now
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM schedules WHERE enabled = 1 AND next_run IS NOT NULL "
                "AND next_run <= ? ORDER BY next_run",
                (now,),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_ran(self, schedule_id: int, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            row = self._conn.execute(
                "SELECT cron FROM schedules WHERE id = ?", (schedule_id,)
            ).fetchone()
            if row is None:
                return
            when = next_run(row["cron"], after=now)
            self._conn.execute(
                "UPDATE schedules SET last_run = ?, next_run = ? WHERE id = ?",
                (now, when, schedule_id),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
