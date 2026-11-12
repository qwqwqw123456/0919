from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""

_CHUNK_SIZE = 500   # 目标块长（字符）
_SENTENCE_RE = re.compile(r"[^。！？；!?;\n]+[。！？；!?;]?")
_VEC_CACHE_LIMIT = 20000   # 分块向量缓存条数上限（按文本摘要键控）

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _char_bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def _chunk_text(text: str, size: int = _CHUNK_SIZE) -> list[str]:
    """按段落/句子切分，尽量接近目标块长，中文友好。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        for sentence in _SENTENCE_RE.findall(para):
            piece = sentence.strip()
            if not piece:
                continue
            if buffer and len(buffer) + len(piece) > size:
                chunks.append(buffer)
                buffer = piece
            else:
                buffer = f"{buffer} {piece}".strip() if buffer else piece
    if buffer:
        chunks.append(buffer)
    return chunks


def _score_chunk(query: str, chunk: str) -> float:
    """混合打分：关键词命中 + 字符 bigram 重叠（零依赖，中文友好）。"""
    tokens = _tokenize(query)
    hay = chunk.lower()
    keyword = sum(1.0 for token in tokens if token in hay)
    q_bg = _char_bigrams(query)
    c_bg = _char_bigrams(chunk)
    overlap = len(q_bg & c_bg) / max(1, len(q_bg | c_bg))
    return keyword * 2.0 + overlap * 3.0


class KnowledgeStore:
    """知识库：文档分块存储 + 语义化检索，供 agent 引用。

    与 Memory 分工：Memory 记"短事实/偏好/经验"，Knowledge 记"长文档/资料"。
    `embed` 可选：传入返回向量的函数后，检索切换为向量余弦相似度（接豆包 embedding）。
    """

    def __init__(self, db_path: str | Path, embed: Callable[[str], list[float]] | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embed = embed
        self._lock = threading.Lock()
        self._vec_cache: dict[str, tuple[float, ...]] = {}
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add_text(self, title: str, text: str, source: str = "") -> int:
        clean = text.strip()
        if not clean:
            raise ValueError("knowledge text is empty")
        now = time.time()
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO documents(title, source, created_at) VALUES (?, ?, ?)",
                (title.strip() or "未命名", source, now),
            )
            doc_id = int(cursor.lastrowid or 0)
            for idx, chunk in enumerate(_chunk_text(clean)):
                self._conn.execute(
                    "INSERT INTO chunks(doc_id, idx, text) VALUES (?, ?, ?)",
                    (doc_id, idx, chunk),
                )
            self._conn.commit()
        return doc_id

    def add_file(self, path: str | Path) -> int:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        title = p.stem
        return self.add_text(title, p.read_text("utf-8", errors="replace"), source=str(p))

    def search(self, query: str, limit: int = 4, per_doc: int = 2) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.id, c.doc_id, c.idx, c.text, d.title, d.source "
                "FROM chunks c JOIN documents d ON d.id = c.doc_id"
            ).fetchall()

        if self.embed is not None:
            query_vec = self.embed(query)
            scored: list[tuple[float, sqlite3.Row]] = []
            for row in rows:
                scored.append((self._cosine(query_vec, self._embed_chunk(row["text"])), row))
        else:
            scored = [(_score_chunk(query, row["text"]), row) for row in rows]

        scored.sort(key=lambda item: -item[0])
        picked: list[dict[str, Any]] = []
        seen_docs: dict[int, int] = {}
        for score, row in scored:
            if score <= 0:
                continue
            if seen_docs.get(int(row["doc_id"]), 0) >= per_doc:
                continue
            seen_docs[int(row["doc_id"])] = seen_docs.get(int(row["doc_id"]), 0) + 1
            picked.append(
                {
                    "chunk_id": int(row["id"]),
                    "doc_id": int(row["doc_id"]),
                    "title": row["title"],
                    "source": row["source"],
                    "score": round(float(score), 4),
                    "text": row["text"],
                }
            )
            if len(picked) >= limit:
                break
        return picked

    def _embed_chunk(self, text: str) -> tuple[float, ...]:
        """分块向量按文本摘要缓存，避免每次检索重复向量化。"""
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        vec = self._vec_cache.get(key)
        if vec is None:
            vec = tuple(self.embed(text))
            if len(self._vec_cache) >= _VEC_CACHE_LIMIT:
                self._vec_cache.clear()
            self._vec_cache[key] = vec
        return vec

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT d.id, d.title, d.source, d.created_at, COUNT(c.id) AS chunks "
                "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id "
                "GROUP BY d.id ORDER BY d.id DESC"
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "title": row["title"],
                "source": row["source"],
                "chunks": int(row["chunks"]),
            }
            for row in rows
        ]

    def remove(self, doc_id: int) -> bool:
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cursor = self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._conn.commit()
        return cursor.rowcount > 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            docs = self._conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
            chunks = self._conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
        return {"documents": int(docs), "chunks": int(chunks)}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
