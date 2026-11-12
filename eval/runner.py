"""Golden 评估集判定器（G2）。

离线样例直接调用仓库内置可执行技能，断言输出包含/不包含指定片段；
需要真实模型的端到端样例标记 ``needs_model``，离线闸门默认跳过。

用法（脚本）::

    python -m eval.runner --skills-dir ../skills [--include-model]

被测试调用时直接用 :func:`evaluate`。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from doubao_agent.skills import SkillManager
from doubao_agent.sources.skill import SkillSource
from doubao_agent.types import ToolCall

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def load_cases(name: str) -> list[dict[str, Any]]:
    payload = json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))
    return list(payload.get("cases", []))


def _check(case: dict[str, Any], output: str) -> list[str]:
    """返回不满足的断言列表；空列表表示通过。"""
    expect = case.get("expect") or {}
    failures: list[str] = []
    for token in expect.get("contains", []):
        if token not in output:
            failures.append(f"缺少片段 {token!r}")
    for token in expect.get("not_contains", []):
        if token in output:
            failures.append(f"出现了应排除片段 {token!r}")
    return failures


async def evaluate(
    skills_dir: str | Path, include_model: bool = False
) -> dict[str, Any]:
    """跑全部 golden 样例，返回汇总与逐条结果。"""
    manager = SkillManager(skills_dir)
    manager.refresh()
    source = SkillSource(manager)

    results: list[dict[str, Any]] = []
    offline_cases = [c for c in load_cases("skills.json")]
    model_cases = [c for c in load_cases("model.json") if c.get("needs_model")]

    for case in offline_cases:
        call = ToolCall(
            name=case["skill"], arguments=case.get("arguments") or {}, call_id=case["id"]
        )
        result = await source.call_tool(call)
        output = result.as_text()
        failures = _check(case, output) if result.ok else [f"技能执行失败：{result.error}"]
        results.append(
            {"id": case["id"], "ok": not failures, "failures": failures, "output": output}
        )

    skipped: list[str] = []
    if include_model:
        provider = _build_model_provider()
        if type(provider).__name__ == "MockProvider":
            # 未配置真实端点（或端点缺失退回 mock）：模型样例不参与离线判定，记跳过。
            skipped = [case["id"] for case in model_cases]
            await provider.aclose()
        else:
            for case in model_cases:
                messages = [{"role": "user", "content": case["task"]}]
                try:
                    resp = await provider.complete(messages, tools=[])
                    content = resp.get("content") or ""
                    failures = _check(case, content)
                    results.append(
                        {"id": case["id"], "ok": not failures, "failures": failures, "output": content}
                    )
                except Exception as exc:  # noqa: BLE001 - 单条样例失败不中断整轮
                    results.append(
                        {
                            "id": case["id"],
                            "ok": False,
                            "failures": [f"模型调用失败：{type(exc).__name__}: {exc}"],
                            "output": "",
                        }
                    )
            await provider.aclose()
    else:
        skipped = [case["id"] for case in model_cases]

    passed = sum(1 for row in results if row["ok"])
    return {"passed": passed, "total": len(results), "skipped": skipped, "results": results}


def _build_model_provider() -> Any:
    """按当前配置构造模型 provider；延迟导入避免离线闸门依赖配置链。"""
    from doubao_agent.config import load_config
    from doubao_agent.providers import build_provider

    return build_provider(load_config().model)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Golden 评估集")
    parser.add_argument("--skills-dir", default=str(_repo_root() / "skills"))
    parser.add_argument("--include-model", action="store_true")
    args = parser.parse_args()

    summary = asyncio.run(evaluate(args.skills_dir, include_model=args.include_model))
    for row in summary["results"]:
        mark = "PASS" if row["ok"] else "FAIL"
        print(f"[{mark}] {row['id']}" + ("" if row["ok"] else f" :: {row['failures']}"))
    print(f"\n通过 {summary['passed']}/{summary['total']}" + (f"，跳过模型样例 {summary['skipped']}" if summary["skipped"] else ""))
    if summary["passed"] != summary["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
