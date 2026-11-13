from __future__ import annotations

import asyncio
import inspect
import json
import re
from typing import Any, Callable

from ..providers import ModelProvider
from .agent import Agent

_DECOMPOSE_PROMPT = """把下面的任务拆分成 2~5 个可以并行执行、互不依赖的子任务。
要求：
1. 每个子任务自包含、可独立完成，不需要等待其他子任务的结果。
2. 子任务用一句话描述清楚目标，避免歧义。
3. 只输出 JSON 字符串数组，不要输出其他任何内容。

任务：{task}"""

_REVIEW_PROMPT = """你是审阅者（reviewer）。下面是原任务与多个执行者（executor）的结果。
请判断：结果是否完成任务、彼此是否冲突，并给出可直接交付的最终答复。
只输出 JSON，不要解释：
{{"answer": "最终答复", "conflicts": ["冲突点"], "verdict": "ok" 或 "needs_attention"}}

原任务：{task}

执行结果：
{results}"""


class Blackboard:
    """共享黑板：协作过程中各角色读写同一份状态（拆解、结果、审阅结论）。"""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._data)


def _parse_review(text: str) -> dict[str, Any] | None:
    """容错解析审阅 JSON；解析不出 answer 就返回 None，由调用方退回机械汇总。"""
    raw = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    answer = str(data.get("answer") or "").strip()
    if not answer:
        return None
    conflicts = [
        str(item).strip() for item in (data.get("conflicts") or []) if str(item).strip()
    ]
    verdict = str(data.get("verdict") or ("ok" if not conflicts else "needs_attention"))
    return {"answer": answer, "conflicts": conflicts, "verdict": verdict}


def _parse_string_list(text: str) -> list[str]:
    """从模型输出中容错解析字符串数组（兼容 markdown 代码块与裸 JSON）。"""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 兜底：按行提取双引号字符串
        return [m for m in re.findall(r'"([^"]+)"', text) if m.strip()]
    if isinstance(data, list):
        return [x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in data]
    if isinstance(data, dict) and isinstance(data.get("subtasks"), list):
        return [
            x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
            for x in data["subtasks"]
        ]
    return []


class Coordinator:
    """并行协同：拆解 → 并发执行子任务 → 汇总。

    `spawn` 每次调用返回一个独立 Agent 实例（共享工具/记忆/审计/计量，
    独立取消信号与会话），子任务并发执行互不干扰。
    """

    def __init__(self, spawn: Callable[[], Agent], provider: ModelProvider) -> None:
        self.spawn = spawn
        self.provider = provider

    async def decompose(self, task: str) -> list[str]:
        """一次模型调用，把复杂任务拆成可并行的子任务。拆解失败则整体作为一个子任务。"""
        reply = await self.provider.complete(
            [{"role": "user", "content": _DECOMPOSE_PROMPT.format(task=task)}], []
        )
        subtasks = _parse_string_list(reply.get("content") or "")
        return subtasks or [task]

    @staticmethod
    async def _emit(
        callback: Callable[[dict[str, Any]], Any] | None, event: dict[str, Any]
    ) -> None:
        if callback is None:
            return
        result = callback(event)
        if inspect.isawaitable(result):
            await result

    async def run_parallel(
        self,
        task: str,
        subtasks: list[str] | None = None,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
        confirm: Callable[[list[str]], Any] | None = None,
        review: bool = False,
    ) -> dict[str, Any]:
        if not subtasks:
            subtasks = await self.decompose(task)

        if confirm is not None:
            outcome = confirm(list(subtasks))
            if inspect.isawaitable(outcome):
                outcome = await outcome
            if isinstance(outcome, list):
                subtasks = [str(s) for s in outcome if str(s).strip()]

        board = Blackboard()
        board.set("task", task)
        board.set("subtasks", list(subtasks))
        await self._emit(
            on_event, {"type": "subtasks", "task": task, "subtasks": subtasks, "count": len(subtasks)}
        )

        async def _one(subtask: str) -> dict[str, Any]:
            agent = self.spawn()
            result = await agent.run(subtask, on_event=on_event)
            return {
                "task": subtask,
                "role": "executor",
                "status": result.get("status", "unknown"),
                "answer": result.get("answer", ""),
                "session_id": result.get("session_id"),
            }

        gathered = await asyncio.gather(
            *(_one(subtask) for subtask in subtasks), return_exceptions=True
        )
        results: list[dict[str, Any]] = []
        for subtask, item in zip(subtasks, gathered):
            if isinstance(item, Exception):
                results.append(
                    {
                        "task": subtask,
                        "status": "failed",
                        "answer": f"{type(item).__name__}: {item}",
                        "session_id": None,
                    }
                )
            else:
                results.append(item)

        board.set("results", results)
        ok_count = sum(1 for item in results if item["status"] == "completed")
        status = "completed" if ok_count == len(results) else ("incomplete" if ok_count else "failed")
        answer = self._summarize(task, results, ok_count)

        verdict: dict[str, Any] | None = None
        if review:
            verdict = await self._review(task, results)
            if verdict is not None:
                board.set("review", verdict)
                answer = verdict["answer"]
                if verdict["verdict"] != "ok" and status == "completed":
                    status = "incomplete"
                await self._emit(
                    on_event,
                    {
                        "type": "review",
                        "task": task,
                        "verdict": verdict["verdict"],
                        "conflicts": verdict["conflicts"],
                    },
                )

        await self._emit(
            on_event, {"type": "finish", "task": task, "answer": answer, "subtasks": results}
        )
        return {
            "status": status,
            "answer": answer,
            "subtasks": results,
            "completed": ok_count,
            "total": len(results),
            "review": verdict,
            "blackboard": board.snapshot(),
        }

    async def _review(self, task: str, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        """审阅者角色：合并执行者结果、指出冲突并给出最终答复。失败时返回 None。"""
        brief = [
            {"task": item["task"], "status": item["status"], "answer": item["answer"][:1200]}
            for item in results
        ]
        prompt = _REVIEW_PROMPT.format(
            task=task, results=json.dumps(brief, ensure_ascii=False, indent=2)
        )
        try:
            reply = await self.provider.complete(
                [{"role": "user", "content": prompt}], []
            )
        except Exception:  # noqa: BLE001 - 审阅失败不应抹掉已有结果
            return None
        return _parse_review(reply.get("content") or "")

    @staticmethod
    def _summarize(task: str, results: list[dict[str, Any]], ok_count: int) -> str:
        lines = [f"任务「{task}」拆分为 {len(results)} 个子任务并行执行，成功 {ok_count} 个："]
        for index, item in enumerate(results, 1):
            lines.append(f"\n{index}. [{item['status']}] {item['task']}\n   {item['answer'].strip()}")
        return "\n".join(lines)
