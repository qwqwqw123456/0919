from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

# 第三层：七个知识域（域索引），编号固定，名称可匹配。
DOMAINS: dict[str, str] = {
    "01": "认知与思维",
    "02": "自我成长",
    "03": "财富与商业",
    "04": "人际与关系",
    "05": "行业与赛道",
    "06": "工具与参考",
    "07": "国家政策",
}


# 知识域路由关键词：查询命中哪一域的词最多，就优先在该域检索；全部为 0 时退回全库。
_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "01": ("思维", "认知", "决策", "逻辑", "学习", "记忆", "专注", "偏见", "判断"),
    "02": ("习惯", "自律", "情绪", "目标", "复盘", "成长", "时间管理", "精力"),
    "03": ("投资", "财务", "商业", "创业", "收入", "成本", "盈利", "定价", "变现", "融资"),
    "04": ("沟通", "关系", "团队", "领导", "谈判", "冲突", "社交", "协作"),
    "05": ("行业", "赛道", "竞争", "趋势", "产业链", "格局", "市场份额"),
    "06": ("工具", "软件", "教程", "文档", "配置", "代码", "接口", "模型", "agent"),
    "07": ("政策", "法规", "补贴", "监管", "标准", "政府", "申报", "合规"),
}


def route_domain(query: str) -> str:
    """按关键词把查询路由到知识域；无法判断返回空串（表示全库检索）。"""
    text = (query or "").lower()
    best, best_score = "", 0
    for code, keywords in _DOMAIN_KEYWORDS.items():
        score = sum(text.count(word) for word in keywords)
        if score > best_score:
            best, best_score = code, score
    return best


def normalize_domain(value: str | None) -> str:
    """把 '01' / '1' / '认知与思维' / '01 认知与思维' 归一成 '01'；空值放行。"""
    text = (value or "").strip()
    if not text:
        return ""
    key = text[:2]
    if key in DOMAINS:
        return key
    if text in DOMAINS.values():
        return next(code for code, name in DOMAINS.items() if name == text)
    raise ValueError(f"未知知识域：{text}（可用：{'、'.join(f'{k} {v}' for k, v in DOMAINS.items())}）")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    domain TEXT NOT NULL DEFAULT '',
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
    `embed` 为同步向量函数；`aembed` 为异步向量函数（宿主桥这类 IO 型 embedding 用）。
    任一提供即切换为向量余弦检索，异步路径失败时退回关键词检索而不是报错。
    """

    def __init__(
        self,
        db_path: str | Path,
        embed: Callable[[str], list[float]] | None = None,
        aembed: Callable[[str], Awaitable[list[float]]] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embed = embed
        self.aembed = aembed
        self._lock = threading.Lock()
        self._vec_cache: dict[str, tuple[float, ...]] = {}
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """老库补 domain 列，避免升级后读写失败。"""
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(documents)")}
        if "domain" not in columns:
            self._conn.execute(
                "ALTER TABLE documents ADD COLUMN domain TEXT NOT NULL DEFAULT ''"
            )

    def add_text(self, title: str, text: str, source: str = "", domain: str = "") -> int:
        clean = text.strip()
        if not clean:
            raise ValueError("knowledge text is empty")
        code = normalize_domain(domain)
        now = time.time()
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO documents(title, source, domain, created_at) VALUES (?, ?, ?, ?)",
                (title.strip() or "未命名", source, code, now),
            )
            doc_id = int(cursor.lastrowid or 0)
            for idx, chunk in enumerate(_chunk_text(clean)):
                self._conn.execute(
                    "INSERT INTO chunks(doc_id, idx, text) VALUES (?, ?, ?)",
                    (doc_id, idx, chunk),
                )
            self._conn.commit()
        return doc_id

    def add_file(self, path: str | Path, domain: str = "") -> int:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        title = p.stem
        return self.add_text(
            title, p.read_text("utf-8", errors="replace"), source=str(p), domain=domain
        )

    def search(
        self, query: str, limit: int = 4, per_doc: int = 2, domain: str | None = None
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        code = self._domain_code(query, domain)
        rows = [row for row in self._rows() if not code or row["domain"] == code]
        if self.embed is not None:
            query_vec = self.embed(query)
            scored: list[tuple[float, sqlite3.Row]] = [
                (self._cosine(query_vec, self._embed_chunk(row["text"])), row) for row in rows
            ]
        else:
            scored = [(_score_chunk(query, row["text"]), row) for row in rows]
        return self._pick(scored, limit, per_doc)

    async def asearch(
        self, query: str, limit: int = 4, per_doc: int = 2, domain: str | None = None
    ) -> list[dict[str, Any]]:
        """异步检索：优先用 aembed；宿主 embedding 不可用时退回关键词检索。"""
        query = query.strip()
        if not query:
            return []
        if self.aembed is None:
            return self.search(query, limit=limit, per_doc=per_doc, domain=domain)
        code = self._domain_code(query, domain)
        rows = [row for row in self._rows() if not code or row["domain"] == code]
        try:
            query_vec = await self.aembed(query)
            scored: list[tuple[float, sqlite3.Row]] = [
                (self._cosine(query_vec, await self._aembed_chunk(row["text"])), row) for row in rows
            ]
        except Exception:  # noqa: BLE001 - 语义检索失败不应让检索整体失败
            scored = [(_score_chunk(query, row["text"]), row) for row in rows]
        return self._pick(scored, limit, per_doc)

    @staticmethod
    def _domain_code(query: str, domain: str | None) -> str:
        """domain='auto' 时按查询自动路由；其余按显式域名归一。"""
        if (domain or "").strip().lower() == "auto":
            return route_domain(query)
        return normalize_domain(domain)

    def _rows(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT c.id, c.doc_id, c.idx, c.text, d.title, d.source, d.domain "
                "FROM chunks c JOIN documents d ON d.id = c.doc_id"
            ).fetchall()

    @staticmethod
    def _pick(
        scored: list[tuple[float, sqlite3.Row]], limit: int, per_doc: int
    ) -> list[dict[str, Any]]:
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
                    "domain": row["domain"],
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

    async def _aembed_chunk(self, text: str) -> tuple[float, ...]:
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        vec = self._vec_cache.get(key)
        if vec is None:
            embed = self.aembed
            assert embed is not None
            vec = tuple(await embed(text))
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
                "SELECT d.id, d.title, d.source, d.domain, d.created_at, COUNT(c.id) AS chunks "
                "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id "
                "GROUP BY d.id ORDER BY d.id DESC"
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "title": row["title"],
                "source": row["source"],
                "domain": row["domain"],
                "chunks": int(row["chunks"]),
            }
            for row in rows
        ]

    def domain_stats(self) -> list[dict[str, Any]]:
        """七个知识域的文档数（域索引），空域也列出。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT domain, COUNT(*) AS documents FROM documents GROUP BY domain"
            ).fetchall()
        counts = {row["domain"]: int(row["documents"]) for row in rows}
        return [
            {"code": code, "name": name, "documents": counts.get(code, 0)}
            for code, name in DOMAINS.items()
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
