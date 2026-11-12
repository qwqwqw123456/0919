from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.runner import evaluate  # noqa: E402


async def test_golden_offline_cases_pass() -> None:
    summary = await evaluate(ROOT / "skills")

    failed = [row["id"] for row in summary["results"] if not row["ok"]]
    assert failed == [], f"离线 golden 样例退化：{[(r['id'], r['failures']) for r in summary['results'] if not r['ok']]}"
    assert summary["passed"] == summary["total"]
    assert summary["skipped"] == ["e2e-simple-answer"]


async def test_golden_model_cases_skipped_by_default() -> None:
    summary = await evaluate(ROOT / "skills", include_model=False)
    assert all(row["id"] != "e2e-simple-answer" for row in summary["results"])
