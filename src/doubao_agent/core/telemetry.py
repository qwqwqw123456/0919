from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    parent_id TEXT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'tool',
    ok INTEGER NOT NULL DEFAULT 1,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    attributes TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spans_session ON spans(session_id);
CREATE INDEX IF NOT EXISTS idx_spans_kind ON spans(kind);
"""


class Telemetry:
    """Span 级追踪：记录模型调用、工具调用与计划步骤的耗时和结果。

    与审计分离：审计保证「谁做了什么」不可篡改，追踪只负责「花了多久、慢在哪」。
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(
        self,
        session_id: str,
        name: str,
        *,
        kind: str = "tool",
        parent_id: str | None = None,
        ok: bool = True,
        duration_ms: int = 0,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        span_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO spans(session_id, span_id, parent_id, name, kind, ok, "
                "duration_ms, attributes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    span_id,
                    parent_id,
                    name,
                    kind,
                    int(ok),
                    max(0, int(duration_ms)),
                    json.dumps(attributes or {}, ensure_ascii=False),
                    time.time(),
                ),
            )
            self._conn.commit()
        return span_id

    def list(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM spans WHERE session_id = ? ORDER BY id LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [
            {
                "span_id": row["span_id"],
                "parent_id": row["parent_id"],
                "name": row["name"],
                "kind": row["kind"],
                "ok": bool(row["ok"]),
                "duration_ms": int(row["duration_ms"]),
                "attributes": json.loads(row["attributes"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def summary(self, session_id: str | None = None) -> dict[str, Any]:
        """单会话或全局的聚合视图：总量、失败数、耗时分布、按工具的调用次数。"""
        if session_id is not None:
            rows = self.list(session_id, limit=10000)
            return self._aggregate(rows, session_id=session_id)

        with self._lock:
            session_count = int(
                self._conn.execute(
                    "SELECT COUNT(DISTINCT session_id) AS n FROM spans"
                ).fetchone()["n"]
            )
            rows = [
                dict(row)
                for row in self._conn.execute(
                    "SELECT * FROM spans ORDER BY id DESC LIMIT 5000"
                ).fetchall()
            ]
        for row in rows:
            row["ok"] = bool(row["ok"])
            row["attributes"] = json.loads(row.get("attributes") or "{}")
        summary = self._aggregate(rows, session_id=None)
        summary["sessions"] = session_count
        return summary

    @staticmethod
    def _aggregate(rows: list[dict[str, Any]], session_id: str | None) -> dict[str, Any]:
        total_ms = sum(int(row["duration_ms"]) for row in rows)
        failures = sum(1 for row in rows if not row["ok"])
        kinds: dict[str, int] = {}
        tools: dict[str, int] = {}
        for row in rows:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
            if row["kind"] == "tool":
                tools[row["name"]] = tools.get(row["name"], 0) + 1
        slowest = sorted(rows, key=lambda row: -int(row["duration_ms"]))[:5]
        return {
            "session_id": session_id,
            "spans": len(rows),
            "failed": failures,
            "ok": len(rows) - failures,
            "total_ms": total_ms,
            "avg_ms": round(total_ms / len(rows), 1) if rows else 0.0,
            "kinds": kinds,
            "tools": tools,
            "slowest": [
                {
                    "name": row["name"],
                    "kind": row["kind"],
                    "duration_ms": int(row["duration_ms"]),
                    "ok": row["ok"],
                }
                for row in slowest
            ],
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
