"""本地记忆召回（memory_retrieve 的数据源）。

零依赖：关键词命中 + 字符 bigram 重叠打分。不联网。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    tag TEXT NOT NULL DEFAULT 'fact',
    created_at REAL NOT NULL
);
"""


def _bigrams(text: str) -> set[str]:
    clean = "".join(ch for ch in text if not ch.isspace())
    return {clean[i : i + 2] for i in range(len(clean) - 1)} or ({clean} if clean else set())


class MemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add(self, text: str, tag: str = "fact") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memories(text, tag, created_at) VALUES (?, ?, ?)",
                (text.strip(), tag, time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def query(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        q_grams = _bigrams(query)
        with self._lock:
            rows = self._conn.execute("SELECT * FROM memories").fetchall()
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            item = dict(row)
            text = item["text"]
            hits = sum(1 for g in q_grams if g and g in text)
            overlap = len(q_grams & _bigrams(text))
            score = float(hits + overlap)
            if query and query in text:
                score += 3.0
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], -pair[1]["created_at"]))
        return [item for _, item in scored[: max(1, k)]]

    def all(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
