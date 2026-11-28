"""图谱新鲜度：拉公开漏洞源，按 29 类分类后进 P 队列（不自动升 A）。

口径：
- 只做「公开来源 → 候选」这一段；进 P 不代表已验证，转 A 必须本机沙箱复现。
- 抓取失败只返回 error，不写半截数据；同一 CVE 重复抓取自动跳过。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx

from .skilltree import TreeLedger

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
MAX_NOTE = 220


def parse_kev(
    payload: dict[str, Any],
    ledger: TreeLedger,
    *,
    days: int = 30,
    limit: int = 50,
    now: datetime | None = None,
    min_score: float = 5.0,
) -> list[dict[str, Any]]:
    """把 KEV JSON 拆成候选；days<=0 表示不按日期过滤。"""
    rows = payload.get("vulnerabilities") or []
    now = now or datetime.now(timezone.utc)
    floor = None
    if days > 0:
        floor = (now - timedelta(days=days)).date()
    out: list[dict[str, Any]] = []
    for row in rows:
        cve = str(row.get("cveID") or "").strip()
        if not cve:
            continue
        added = str(row.get("dateAdded") or "")
        if floor:
            try:
                if datetime.strptime(added, "%Y-%m-%d").date() < floor:
                    continue
            except ValueError:
                continue
        text = " ".join(
            str(row.get(k) or "")
            for k in ("vendorProject", "product", "vulnerabilityName", "shortDescription")
        )
        cats = [c for c in ledger.classify(text, limit=3) if c["score"] >= min_score]
        if not cats:
            continue
        out.append({
            "id": cve,
            "title": str(row.get("vulnerabilityName") or cve)[:120],
            "date": added,
            "vendor": str(row.get("vendorProject") or ""),
            "product": str(row.get("product") or ""),
            "cats": [{"no": c["no"], "name": c["name"], "score": c["score"], "hits": c["hits"]} for c in cats],
            "level": "P",
            "source": KEV_URL,
            "status": "pending",
            "note": str(row.get("shortDescription") or "")[:MAX_NOTE],
        })
        if len(out) >= limit:
            break
    return out


class Freshness:
    def __init__(self, ledger: TreeLedger, root: str | Path | None = None) -> None:
        self.ledger = ledger
        self.root = Path(root) if root else ledger.root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "fresh.jsonl"

    def pending(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def fetch(self, *, days: int = 30, limit: int = 50, timeout: float = 20.0) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(KEV_URL)
                resp.raise_for_status()
                payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "added": 0, "total": len(self.pending())}
        return self.ingest(payload, days=days, limit=limit)

    def ingest(self, payload: dict[str, Any], *, days: int = 30, limit: int = 50) -> dict[str, Any]:
        seen = {str(r.get("id")) for r in self.pending()}
        items = [c for c in parse_kev(payload, self.ledger, days=days, limit=limit) if c["id"] not in seen]
        with self.path.open("a", encoding="utf-8") as fh:
            for item in items:
                item["ts"] = round(time.time(), 3)
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        return {"ok": True, "added": len(items), "total": len(self.pending()), "items": items, "path": str(self.path)}
