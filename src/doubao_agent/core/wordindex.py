"""Agent 索引检索：2667 条词条的类/子类/正文全文查找（只读，不改图谱）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_INDEX = Path(__file__).resolve().parents[3] / "data" / "sec-agent-index.json"
_cache: dict[str, Any] = {"path": None, "rows": []}


def _rows(path: Path) -> list[dict[str, Any]]:
    if _cache["path"] == str(path) and _cache["rows"]:
        return _cache["rows"]
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for cat in data.get("cats") or []:
        for sub in cat.get("subs") or []:
            for item in sub.get("items") or []:
                rows.append({
                    "no": int(cat.get("no") or 0),
                    "cat": str(cat.get("name") or ""),
                    "sub": str(sub.get("name") or ""),
                    "gid": item.get("gid"),
                    "name": str(item.get("name") or ""),
                    "status": str(item.get("status") or ""),
                    "note": str(item.get("note") or ""),
                })
    _cache.update({"path": str(path), "rows": rows})
    return rows


def lookup(query: str, *, limit: int = 20, path: str | Path | None = None) -> dict[str, Any]:
    """按名称/正文做子串检索，名称命中排在正文命中前。"""
    q = (query or "").strip().lower()
    if not q:
        return {"ok": False, "error": "query 为空", "total": 0, "hits": []}
    rows = _rows(Path(path) if path else DEFAULT_INDEX)
    name_hits = [r for r in rows if q in r["name"].lower()]
    note_hits = [r for r in rows if q not in r["name"].lower() and q in r["note"].lower()]
    hits = name_hits + note_hits
    out = []
    for r in hits[: max(1, min(limit, 100))]:
        out.append({**r, "note": r["note"][:260], "match": "name" if q in r["name"].lower() else "note"})
    return {"ok": True, "query": query, "total": len(hits), "hits": out}
