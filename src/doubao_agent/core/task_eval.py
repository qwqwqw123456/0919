"""任务级评测：把真实任务配上可判真假的验收标准，跑出分数而不是感觉。

判定只用确定性检查（文本断言 + 磁盘实物 + 调用预算），不看模型自评。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from .verify import check_file, check_text

TASKS_PATH = Path(__file__).resolve().parents[3] / "eval" / "tasks.json"


def load_tasks(path: Path | None = None) -> dict[str, Any]:
    payload = json.loads((path or TASKS_PATH).read_text(encoding="utf-8"))
    return payload


def prepare_workspace(raw: str) -> Path:
    """每次评测前清空工作区，并放一个种子文件供列目录类任务有东西可列。"""
    workspace = Path(raw)
    if workspace.exists():
        for item in workspace.rglob("*"):
            if item.is_file():
                item.unlink()
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "seed.txt").write_text("seed", encoding="utf-8")
    return workspace


def evaluate(case: dict[str, Any], result: dict[str, Any], workspace: Path) -> dict[str, Any]:
    """返回 {pass, checks, issues}；验收标准全部满足才算过。"""
    accept = case.get("accept") or {}
    answer = str(result.get("answer") or "")
    usage = result.get("usage") or {}
    calls = int(usage.get("calls") or 0)
    checks: list[dict[str, Any]] = []

    text_ok = check_text(
        answer,
        contains=accept.get("contains") or (),
        not_contains=accept.get("not_contains") or (),
        regex=accept.get("regex"),
    )
    checks.append({**text_ok, "name": "answer"})

    for spec in accept.get("files") or []:
        path = str(spec.get("path") or "").replace("{ws}", str(workspace))
        checks.append(
            check_file(
                path,
                min_bytes=int(spec.get("min_bytes", 1)),
                contains=spec.get("contains") or (),
                must_parse_json=bool(spec.get("json", False)),
            )
        )

    if accept.get("max_calls"):
        limit = int(accept["max_calls"])
        checks.append(
            {
                "name": "budget:calls",
                "ok": calls <= limit,
                "detail": f"{calls} 次调用（上限 {limit}）",
            }
        )

    if accept.get("require_completed", True) and result.get("status") != "completed":
        checks.append(
            {"name": "status", "ok": False, "detail": f"运行状态 {result.get('status')}"}
        )

    issues = [f"{c['name']}: {c['detail']}" for c in checks if not c["ok"]]
    return {"pass": not issues, "checks": checks, "issues": issues}


async def run_tasks(
    runtime: Any,
    cases: list[dict[str, Any]],
    workspace: Path,
    timeout: float = 240.0,
    on_case: Any = None,
    concurrency: int = 4,
) -> dict[str, Any]:
    """每条用例独立工作区、独立 Agent，按并发窗口跑，互不干扰。"""
    rows: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(max(1, concurrency))

    def sub_workspace(case: dict[str, Any]) -> Path:
        target = workspace / str(case["id"])
        prepare_workspace(str(target))
        return target

    async def one(case: dict[str, Any]) -> None:
        async with sem:
            case_ws = sub_workspace(case)
            task_text = str(case["task"]).replace("{ws}", str(case_ws))
            started = time.time()
            try:
                result = await asyncio.wait_for(runtime.start_run(task_text), timeout=timeout)
            except asyncio.TimeoutError:
                result = {"status": "timeout", "answer": "", "usage": {}, "route": {}}
            except Exception as exc:  # noqa: BLE001 - 单条崩溃不能带崩整批评测
                result = {"status": f"error: {type(exc).__name__}", "answer": "", "usage": {}, "route": {}}
            elapsed = round(time.time() - started, 1)
            verdict = evaluate(case, result, case_ws)
            route = result.get("route") or {}
            row = {
                "id": case["id"],
                "stratum": case.get("stratum"),
                "task": task_text,
                "status": result.get("status"),
                "route": route.get("decision"),
                "verified": route.get("verify_pass"),
                "calls": (result.get("usage") or {}).get("calls"),
                "tokens": (result.get("usage") or {}).get("total_tokens"),
                "seconds": elapsed,
                "pass": verdict["pass"],
                "issues": verdict["issues"],
                "answer": str(result.get("answer") or "")[:400],
            }
            rows.append(row)
            if on_case:
                on_case(row)

    await asyncio.gather(*(one(case) for case in cases))
    rows.sort(key=lambda r: str(r["id"]))
    by_stratum: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_stratum.setdefault(row["stratum"], {"total": 0, "pass": 0})
        bucket["total"] += 1
        bucket["pass"] += 1 if row["pass"] else 0

    total = len(rows)
    passed = sum(1 for row in rows if row["pass"])
    tokens = sum(int(row["tokens"] or 0) for row in rows)
    return {
        "total": total,
        "pass": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "tokens": tokens,
        "by_stratum": {
            key: {**val, "rate": round(val["pass"] / val["total"], 3)} for key, val in by_stratum.items()
        },
        "rows": rows,
    }


def save_report(report: dict[str, Any], folder: Path | None = None) -> Path:
    folder = folder or (Path(__file__).resolve().parents[3] / "eval" / "results")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
