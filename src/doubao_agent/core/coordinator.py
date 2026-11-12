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
    ) -> dict[str, Any]:
        if not subtasks:
            subtasks = await self.decompose(task)

        if confirm is not None:
            outcome = confirm(list(subtasks))
            if inspect.isawaitable(outcome):
                outcome = await outcome
            if isinstance(outcome, list):
                subtasks = [str(s) for s in outcome if str(s).strip()]

        await self._emit(
            on_event, {"type": "subtasks", "task": task, "subtasks": subtasks, "count": len(subtasks)}
        )

        async def _one(subtask: str) -> dict[str, Any]:
            agent = self.spawn()
            result = await agent.run(subtask, on_event=on_event)
            return {
                "task": subtask,
                "status": result.get("status", "unknown"),
                "answer": result.get("answer", ""),
                "session_id": result.get("session_id"),
            }

        results = await asyncio.gather(*(_one(subtask) for subtask in subtasks))

        ok_count = sum(1 for item in results if item["status"] == "completed")
        status = "completed" if ok_count == len(results) else ("incomplete" if ok_count else "failed")
        answer = self._summarize(task, results, ok_count)
        await self._emit(
            on_event, {"type": "finish", "task": task, "answer": answer, "subtasks": results}
        )
        return {
            "status": status,
            "answer": answer,
            "subtasks": results,
            "completed": ok_count,
            "total": len(results),
        }

    @staticmethod
    def _summarize(task: str, results: list[dict[str, Any]], ok_count: int) -> str:
        lines = [f"任务「{task}」拆分为 {len(results)} 个子任务并行执行，成功 {ok_count} 个："]
        for index, item in enumerate(results, 1):
            lines.append(f"\n{index}. [{item['status']}] {item['task']}\n   {item['answer'].strip()}")
        return "\n".join(lines)
