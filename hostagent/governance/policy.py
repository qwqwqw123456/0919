"""白名单预授权策略存储与匹配。

把宿主的「人工批准」改写成「规则预授权」：命中白名单即自动放行。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool TEXT NOT NULL,
    dimension TEXT NOT NULL,
    value TEXT NOT NULL,
    operator TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
"""

# 参数里可能承载「对象」的键，按维度映射
_DIMENSION_PARAM = {
    "target": ("target_user_id", "target_id", "user_id"),
    "channel": ("chat_id", "channel", "target_id"),
}


class PolicyStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add(self, tool: str, dimension: str, value: str, operator: str | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO policies(tool, dimension, value, operator, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tool, dimension, value, operator, time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def remove(self, policy_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM policies WHERE id = ?", (policy_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def set_enabled(self, policy_id: int, enabled: bool) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE policies SET enabled = ? WHERE id = ?",
                (1 if enabled else 0, policy_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def list(self, tool: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if tool:
                rows = self._conn.execute(
                    "SELECT * FROM policies WHERE tool = ? ORDER BY id", (tool,)
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM policies ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def match_rule(self, tool: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """返回命中的第一条 enabled 预授权规则；未命中返回 None。"""
        for rule in self.list(tool):
            if not rule["enabled"]:
                continue
            dimension = rule["dimension"]
            expected = rule["value"]
            if dimension == "any" or expected == "*":
                return rule
            keys = _DIMENSION_PARAM.get(dimension, (dimension,))
            for key in keys:
                actual = params.get(key)
                if actual is not None and str(actual) == str(expected):
                    return rule
        return None

    def match(self, tool: str, params: dict[str, Any]) -> bool:
        return self.match_rule(tool, params) is not None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
