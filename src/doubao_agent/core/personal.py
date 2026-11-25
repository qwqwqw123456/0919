"""Personal Agent 层（小豆式）：跟进任务卡 + 四模块状态卡 + 确认闸门。

与原框架的分工：记忆用 core/longterm + core/working，定时用 core/scheduler，
审计用 core/audit。本模块只补"个人助理"这一层——把一句话交代的事变成可核查
的任务卡，并把"在跟进 / 命中 / 临期 / 待确认"合成一张不超过 5 行的状态卡。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

STATUSES = ("active", "paused", "done")
CONFIRM_LEVELS = {
    0: "只推送",
    1: "执行前需点头",
    2: "涉及付费/外发，必须确认",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pa_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    target TEXT NOT NULL,
    condition TEXT NOT NULL,
    freq TEXT NOT NULL DEFAULT 'daily',
    action TEXT NOT NULL DEFAULT 'push',
    confirm_level INTEGER NOT NULL DEFAULT 0,
    due_at REAL,
    status TEXT NOT NULL DEFAULT 'active',
    note TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    last_check REAL,
    last_result TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS pa_pushes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER,
    summary TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class ConfirmRequired(PermissionError):
    """确认级别未满足时抛出，调用方必须把话术原样转给用户。"""


def check_gate(confirm_level: int, *, user_confirmed: bool = False) -> None:
    """执行前闸门。0 级直接放行；1/2 级必须拿到用户明确点头。"""
    level = int(confirm_level)
    if level not in CONFIRM_LEVELS:
        raise ValueError(f"未知确认级别: {confirm_level}")
    if level == 0:
        return
    if user_confirmed:
        return
    raise ConfirmRequired(
        f"该任务确认级别为 {level}（{CONFIRM_LEVELS[level]}），需要用户先明确同意再执行"
    )


class PersonalStore:
    """跟进任务卡与推送记录，落在与原框架同一个 sqlite 库里。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add(
        self,
        name: str,
        target: str,
        condition: str,
        freq: str = "daily",
        action: str = "push",
        confirm_level: int = 0,
        due_at: float | None = None,
        note: str = "",
    ) -> int:
        if not name.strip():
            raise ValueError("任务名不能为空")
        if not target.strip():
            raise ValueError("监控对象不能为空")
        if not condition.strip():
            raise ValueError("触发条件不能为空")
        if int(confirm_level) not in CONFIRM_LEVELS:
            raise ValueError(f"未知确认级别: {confirm_level}")
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO pa_tasks(name, target, condition, freq, action, confirm_level, "
                "due_at, status, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)",
                (
                    name.strip(),
                    target.strip(),
                    condition.strip(),
                    freq.strip() or "daily",
                    action.strip() or "push",
                    int(confirm_level),
                    due_at,
                    note.strip(),
                    time.time(),
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    def list(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            if status:
                if status not in STATUSES:
                    raise ValueError(f"未知状态: {status}")
                rows = self._conn.execute(
                    "SELECT * FROM pa_tasks WHERE status = ? ORDER BY id DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM pa_tasks ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(row) for row in rows]

    def get(self, task_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM pa_tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def set_status(self, task_id: int, status: str) -> bool:
        if status not in STATUSES:
            raise ValueError(f"未知状态: {status}")
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE pa_tasks SET status = ? WHERE id = ?", (status, task_id)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def touch(self, task_id: int, result: str, checked_at: float | None = None) -> bool:
        """记录一次检查结果，供状态卡判断"今日已命中"和"连续空转"。"""
        checked_at = time.time() if checked_at is None else checked_at
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE pa_tasks SET last_check = ?, last_result = ? WHERE id = ?",
                (checked_at, result.strip(), task_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def log_push(self, task_id: int | None, summary: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO pa_pushes(task_id, summary, created_at) VALUES (?, ?, ?)",
                (task_id, summary.strip(), time.time()),
            )
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    def pushes(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM pa_pushes ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _same_day(a: float, b: float) -> bool:
    return datetime.fromtimestamp(a).date() == datetime.fromtimestamp(b).date()


def build_briefing(
    tasks: list[dict[str, Any]],
    pushes: list[dict[str, Any]] | None = None,
    *,
    now: float | None = None,
    due_soon_seconds: float = 86400,
    max_lines: int = 5,
) -> list[str]:
    """四模块状态卡。没有在跟进的事、也没有命中记录时，只回一句"今天无事"。"""
    now = time.time() if now is None else now
    pushes = pushes or []

    active = [t for t in tasks if t.get("status") == "active"]
    paused = [t for t in tasks if t.get("status") == "paused"]

    if not active and not paused and not pushes:
        return ["今天无事：没有在跟进的事，也没有命中记录"]

    lines: list[str] = [f"在跟进 {len(active)} 条" + (f"，暂停 {len(paused)} 条" if paused else "")]

    today_hits = [p for p in pushes if _same_day(float(p["created_at"]), now)]
    if today_hits:
        for hit in today_hits[:2]:
            lines.append(f"今日命中：{hit['summary']}")
    else:
        lines.append("今日命中：无")

    due_items = [t for t in active if t.get("due_at") and float(t["due_at"]) - now <= due_soon_seconds]
    if due_items:
        first = min(due_items, key=lambda t: float(t["due_at"]))
        when = datetime.fromtimestamp(float(first["due_at"])).strftime("%m-%d %H:%M")
        lines.append(f"临期 {len(due_items)} 条：{first['name']}（{when}）")

    blocked = [t for t in active if int(t.get("confirm_level") or 0) > 0]
    if blocked:
        lines.append(f"待你点头 {len(blocked)} 条：{blocked[0]['name']}")

    idle = [t for t in active if not t.get("last_result") or "无命中" in str(t.get("last_result"))]
    if idle and len(lines) < max_lines:
        lines.append(f"空转待巡 {len(idle)} 条：{idle[0]['name']}")

    return lines[:max_lines]
