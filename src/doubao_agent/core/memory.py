from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

KINDS = ("preference", "fact", "experience", "procedure")

_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'fact',
    tags TEXT NOT NULL DEFAULT '',
    weight REAL NOT NULL DEFAULT 1.0,
    hits INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT '',
    fingerprint TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    used_at REAL NOT NULL
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_facts_kind ON facts(kind);
CREATE INDEX IF NOT EXISTS idx_facts_fp ON facts(fingerprint);
"""

# 零成本的偏好抽取：不调用模型，只认显式表达。
_PREFERENCE_PATTERNS = (
    re.compile(r"记住[：:，,\s]*([^。；;\n]+)"),
    re.compile(r"以后(?:都|默认|一律)?(?:用|按|要|走)\s*([^。；;\n]+)"),
    re.compile(r"我(?:更)?(?:喜欢|偏好|习惯|倾向)\s*([^。；;\n]+)"),
)

_KIND_BOOST = {"preference": 0.6, "procedure": 0.5, "experience": 0.35, "fact": 0.25}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _fingerprint(text: str) -> str:
    return hashlib.sha1(_normalize(text).encode("utf-8")).hexdigest()


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def extract_preferences(text: str) -> list[str]:
    """从一句话里抽出用户偏好，去重且丢弃被更长条目包含的碎片。"""
    found: list[str] = []
    for pattern in _PREFERENCE_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(1).strip(" 。．.；;，,、")
            if 2 <= len(value) <= 200 and value not in found:
                found.append(value)
    kept: list[str] = []
    for value in sorted(found, key=len, reverse=True):
        if not any(value in longer for longer in kept):
            kept.append(value)
    return list(reversed(kept))


class MemoryStore:
    """长期记忆与经验沉淀。

    四类记忆：preference / fact / experience / procedure。
    相同内容按 fingerprint 合并加权，越常用权重越高。
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_BASE_SCHEMA)
        self._migrate()
        self._conn.executescript(_INDEXES)
        self._conn.commit()

    def _migrate(self) -> None:
        """兼容早期只有 text/tags 的旧表。"""
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(facts)")}
        additions = {
            "kind": "TEXT NOT NULL DEFAULT 'fact'",
            "weight": "REAL NOT NULL DEFAULT 1.0",
            "hits": "INTEGER NOT NULL DEFAULT 0",
            "source": "TEXT NOT NULL DEFAULT ''",
            "fingerprint": "TEXT NOT NULL DEFAULT ''",
            "used_at": "REAL NOT NULL DEFAULT 0",
        }
        for name, ddl in additions.items():
            if name not in columns:
                self._conn.execute(f"ALTER TABLE facts ADD COLUMN {name} {ddl}")
        self._conn.execute("UPDATE facts SET used_at = created_at WHERE used_at = 0")
        self._conn.commit()

    def remember(
        self,
        text: str,
        kind: str = "fact",
        tags: str = "",
        source: str = "",
        weight: float = 1.0,
    ) -> int:
        clean = text.strip()
        if not clean:
            raise ValueError("memory text is empty")
        if kind not in KINDS:
            kind = "fact"
        fingerprint = _fingerprint(clean)
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT id, weight, tags FROM facts WHERE fingerprint = ? ORDER BY id LIMIT 1",
                (fingerprint,),
            ).fetchone()
            if row:
                merged_tags = tags or row["tags"]
                new_weight = min(5.0, float(row["weight"]) + 0.3)
                self._conn.execute(
                    "UPDATE facts SET weight = ?, tags = ?, used_at = ?, hits = hits + 1 WHERE id = ?",
                    (new_weight, merged_tags, now, row["id"]),
                )
                self._conn.commit()
                return int(row["id"])
            cursor = self._conn.execute(
                "INSERT INTO facts(text, kind, tags, weight, hits, source, fingerprint, created_at, used_at) "
                "VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)",
                (clean, kind, tags, weight, source, fingerprint, now, now),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def recall(
        self, query: str, limit: int = 6, kinds: tuple[str, ...] | None = None
    ) -> list[dict[str, Any]]:
        query = query.strip()
        tokens = [t for t in re.split(r"[\s,，。;；/、]+", query) if len(t) >= 2][:6]
        patterns: list[tuple[str, float]] = []
        if query:
            patterns.append((query.lower(), 2.0))
        patterns.extend((token.lower(), 1.0) for token in tokens)
        if not patterns:
            return self.recent(limit)

        where = " OR ".join(["lower(text) LIKE ? OR lower(tags) LIKE ?"] * len(patterns))
        params: list[Any] = []
        for needle, _ in patterns:
            params.extend([f"%{needle}%", f"%{needle}%"])
        kind_clause = ""
        if kinds:
            kind_clause = f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)

        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM facts WHERE ({where}){kind_clause} ORDER BY used_at DESC LIMIT 200",
                params,
            ).fetchall()

        now = time.time()
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            haystack = f"{row['text']} {row['tags']}".lower()
            score = sum(weight for needle, weight in patterns if needle in haystack)
            score += float(row["weight"]) * 0.5
            score += _KIND_BOOST.get(row["kind"], 0.2)
            score += 0.1 / (1.0 + (now - float(row["used_at"])) / 86400 / 30)
            scored.append((score, row))

        scored.sort(key=lambda item: (-item[0], -item[1]["id"]))
        picked = [row for _, row in scored[:limit]]
        self._reinforce([int(row["id"]) for row in picked])
        return [self._row_to_dict(row) for row in picked]

    def _reinforce(self, ids: list[int]) -> None:
        if not ids:
            return
        now = time.time()
        with self._lock:
            self._conn.executemany(
                "UPDATE facts SET hits = hits + 1, used_at = ? WHERE id = ?",
                [(now, memory_id) for memory_id in ids],
            )
            self._conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "text": row["text"],
            "kind": row["kind"],
            "tags": row["tags"],
            "weight": float(row["weight"]),
            "hits": int(row["hits"]),
        }

    def top(self, kind: str, limit: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE kind = ? ORDER BY weight DESC, used_at DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM facts ORDER BY used_at DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def forget(self, memory_id: int) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM facts WHERE id = ?", (memory_id,))
            self._conn.commit()
        return cursor.rowcount > 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, COUNT(*) AS count FROM facts GROUP BY kind ORDER BY kind"
            ).fetchall()
        kinds = {row["kind"]: int(row["count"]) for row in rows}
        return {"total": sum(kinds.values()), "kinds": kinds}

    def consolidate(
        self,
        task: str,
        answer: str,
        status: str,
        steps: int,
        session_id: str = "",
        recipe_id: int | None = None,
    ) -> list[int]:
        """任务结束后自动沉淀：偏好、经验、可复用配方。全程零模型调用。"""
        ids: list[int] = []
        for preference in extract_preferences(task):
            ids.append(
                self.remember(preference, kind="preference", tags="auto", source=session_id)
            )

        if status == "completed":
            if recipe_id is not None:
                ids.append(
                    self.remember(
                        f"任务「{_clip(task, 40)}」已沉淀为配方 #{recipe_id}，可零模型重放",
                        kind="procedure",
                        tags=f"recipe:{recipe_id}",
                        source=session_id,
                    )
                )
            if steps >= 2:
                outcome = f"完成「{_clip(task, 40)}」"
                if answer:
                    outcome += f"：{_clip(answer, 160)}"
                ids.append(
                    self.remember(outcome, kind="experience", tags="task", source=session_id)
                )
        else:
            outcome = f"未完成「{_clip(task, 40)}」"
            if answer:
                outcome += f"：{_clip(answer, 160)}"
            ids.append(
                self.remember(
                    outcome, kind="experience", tags="lesson", source=session_id, weight=0.8
                )
            )
        return ids

    def close(self) -> None:
        with self._lock:
            self._conn.close()
