"""新鲜度：KEV 解析、日期过滤、噪声门槛、去重。"""
from datetime import datetime, timezone
from pathlib import Path

from doubao_agent.core.freshness import Freshness, parse_kev
from doubao_agent.core.skilltree import DEFAULT_INDEX, TreeLedger

NOW = datetime(2026, 11, 28, tzinfo=timezone.utc)


def _ledger(tmp_path: Path) -> TreeLedger:
    return TreeLedger(tmp_path / "tree", index_path=DEFAULT_INDEX)


PAYLOAD = {
    "vulnerabilities": [
        {
            "cveID": "CVE-2026-0001",
            "vendorProject": "Acme",
            "product": "Portal",
            "vulnerabilityName": "SQL Injection in login",
            "shortDescription": "SQL injection allows remote code execution.",
            "dateAdded": "2026-11-20",
        },
        {
            "cveID": "CVE-2026-0002",
            "vendorProject": "Acme",
            "product": "Agent",
            "vulnerabilityName": "Prompt injection in tool layer",
            "shortDescription": "prompt injection via mcp tool descriptions",
            "dateAdded": "2026-11-27",
        },
        {
            "cveID": "CVE-2025-9999",
            "vendorProject": "Old",
            "product": "Thing",
            "vulnerabilityName": "SQL Injection",
            "shortDescription": "sql injection",
            "dateAdded": "2025-01-01",
        },
    ]
}


def test_parse_filters_old_and_keeps_strong(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    items = parse_kev(PAYLOAD, ledger, days=30, limit=10, now=NOW)
    ids = {i["id"] for i in items}
    assert "CVE-2025-9999" not in ids          # 日期过滤
    assert "CVE-2026-0001" in ids
    first = next(i for i in items if i["id"] == "CVE-2026-0001")
    assert first["level"] == "P"
    assert first["status"] == "pending"
    assert first["cats"][0]["no"] == 2


def test_parse_skips_weak_matches(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    payload = {"vulnerabilities": [{
        "cveID": "CVE-2026-0003",
        "vendorProject": "Foo",
        "product": "Bar",
        "vulnerabilityName": "Improper Check",
        "shortDescription": "nothing notable",
        "dateAdded": "2026-11-27",
    }]}
    assert parse_kev(payload, ledger, days=30, limit=10, now=NOW) == []


def test_ingest_dedupes_by_cve(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    fresh = Freshness(ledger)
    r1 = fresh.ingest(PAYLOAD, days=0, limit=10)
    r2 = fresh.ingest(PAYLOAD, days=0, limit=10)
    assert r1["added"] >= 2
    assert r2["added"] == 0
    assert len(fresh.pending()) == r1["total"]
