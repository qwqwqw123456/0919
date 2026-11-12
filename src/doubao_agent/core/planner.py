from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..providers import ModelProvider
from ..skills import SkillManager
from ..types import ToolSpec
from .usage import UsageMeter

PLANNER_PROMPT = """你是任务规划器。你只做一次规划，不执行，也不与用户对话。

根据任务和可用工具，输出**严格 JSON**（不要 Markdown 代码块，不要任何解释文字）：
- 能用一段非交互脚本完成的多步操作：
  {"mode":"script","language":"python","code":"..."}
- 必须调用外部工具（MCP / API / 浏览器 / 原生能力）：
  {"mode":"tools","steps":[{"tool":"工具名","arguments":{...}}]}
- 无需任何操作，只需回答：
  {"mode":"answer","answer":"..."}

规则：
1. 优先 script 模式：把多步操作合并进一段脚本，一次执行完成。
2. 最多 8 步；参数不确定就不要编造，改用 answer 说明缺什么。
3. 脚本必须非交互，关键结果用 print 输出。
4. 工具名必须来自下面的清单，不得臆造。
5. 步骤有先后依赖时可加 "id" 与 "depends_on":["上游id"]；无依赖不要加，执行时会自动顺序跑。

可用工具：
{catalog}"""

@dataclass
class Plan:
    mode: str
    language: str = "python"
    code: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""


def parse_plan(text: str) -> Plan | None:
    """容错解析模型输出；任何不合规都返回 None，由调用方决定是否回退。"""
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    mode = str(data.get("mode", "")).strip().lower()
    if mode == "answer":
        return Plan(mode="answer", answer=str(data.get("answer", "")))

    if mode == "script":
        code = data.get("code")
        if not isinstance(code, str) or not code.strip():
            return None
        language = str(data.get("language", "python")).strip().lower()
        language = "shell" if language in ("shell", "bash", "sh") else "python"
        return Plan(mode="script", language=language, code=code)

    if mode == "tools":
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            return None
        steps: list[dict[str, Any]] = []
        for item in raw_steps[:8]:
            if not isinstance(item, dict):
                return None
            tool = item.get("tool")
            if not isinstance(tool, str) or not tool.strip():
                return None
            arguments = item.get("arguments") or {}
            if not isinstance(arguments, dict):
                return None
            step: dict[str, Any] = {"tool": tool.strip(), "arguments": arguments}
            depends = item.get("depends_on")
            if isinstance(depends, list) and depends:
                step["depends_on"] = [str(d) for d in depends]
            if item.get("id"):
                step["id"] = str(item["id"])
            steps.append(step)
        return Plan(mode="tools", steps=steps)

    return None


class Planner:
    """入口单次规划：一个任务只调用一次模型，之后执行阶段零推理。"""

    def __init__(
        self,
        provider: ModelProvider,
        skills: SkillManager | None = None,
        usage: UsageMeter | None = None,
    ) -> None:
        self.provider = provider
        self.skills = skills
        self.usage = usage or UsageMeter()

    @staticmethod
    def _catalog(tools: list[ToolSpec]) -> str:
        return "\n".join(
            f"- {spec.name}（{spec.risk.value}）：{spec.description}"
            for spec in tools
            if not spec.name.startswith("script.")
        )

    async def plan(
        self, task: str, tools: list[ToolSpec], memory_context: str = ""
    ) -> Plan | None:
        catalog = self._catalog(tools)
        system = PLANNER_PROMPT.replace("{catalog}", catalog or "（无可用工具）")
        if self.skills is not None:
            knowledge = self.skills.catalog()
            if knowledge:
                system += "\n\n知识型技能（先读说明，再按说明产出计划）：\n" + knowledge
            matched = self.skills.knowledge_context(task)
            if matched:
                system += "\n\n命中的技能说明：\n" + matched
        if memory_context:
            system += f"\n\n参考记忆：\n{memory_context}"

        reply = await self.provider.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": task},
            ],
            [],
        )
        self.usage.record(reply.get("usage"))
        return parse_plan(str(reply.get("content") or ""))
