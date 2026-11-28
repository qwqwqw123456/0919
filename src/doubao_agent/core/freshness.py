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


def triage_markdown(ledger: TreeLedger, path: Path | None = None, *, only: str = "") -> Path:
    """把 P 候选整理成复现核查单：只写版本/配置核对步骤，不写利用载荷。"""
    fresh = Freshness(ledger)
    rows = fresh.pending()
    if only:
        rows = [r for r in rows if str(r.get("id")) == only]
    out = Path(path) if path else ledger.root / f"triage-{time.strftime('%Y%m%d')}.md"
    lines = [
        "# KEV 候选复现核查单",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}｜候选 {len(rows)} 条｜来源：CISA KEV（公开）",
        "",
        "口径：本单只做资产/版本/配置核对；任何复现必须在已授权的隔离资产与快照上进行，"
        "不在本单提供利用载荷。跑通后按会话回写等级，P→A 只认本机证据。",
        "",
    ]
    for r in rows:
        cats = "、".join(f"#{c['no']}{c['name']}" for c in r.get("cats") or [])
        lines += [
            f"## {r.get('id')} {r.get('title', '')}",
            "",
            f"- 厂商/产品：{r.get('vendor', '')} / {r.get('product', '')}",
            f"- 收录日期：{r.get('date', '')}｜归类：{cats or '—'}",
            f"- 摘要：{r.get('note', '')}",
            "",
            "核查步骤（只读）：",
            "1. 资产确认：从自建资产清单核对是否存在该产品与暴露面，不做外部扫描。",
            "2. 版本核对：`<产品> --version` 或管理台版本页，与厂商公告的受影响区间比对；"
            "拿不到版本就先标 `未确认`，不要猜。",
            "3. 配置核对：按公告列出的前置条件（组件开关、端口、权限）逐条打勾。",
            "4. 隔离验证：仅在已授权资产 + 快照 + 断网/内网环境复现，记录命令与输出。",
            "5. 回写：`doubao-agent tree ingest --session <验证会话ID>`；A 只认本机跑通的证据。",
            "",
        ]
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    if rows:
        by_id = {str(r.get("id")) for r in rows}
        rewritten = []
        for row in fresh.pending():
            if str(row.get("id")) in by_id:
                row["status"] = "triaged"
            rewritten.append(json.dumps(row, ensure_ascii=False))
        fresh.path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return out
