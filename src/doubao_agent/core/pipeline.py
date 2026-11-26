"""任务流水线：拆解 → 逐原子步骤执行 → 独立核验 → 定点修复。

职责分离是本模块的唯一目的：规划器不下场执行，执行器只做当前一步且不自评，
核验器不给工具、只按验收标准判对错。任何阶段预算用尽即停，不硬跑。

失败回退：拆解不出合法计划时返回 None，由 Agent 回落到常规则 ReAct 循环。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

PLAN_PROMPT = """你是任务拆解器。只拆解，不执行，不回答问题。
输出严格 JSON（不要 Markdown 代码块，不要任何解释）：
{{"goal":"一句话目标","criteria":["可判真假的验收标准"],"steps":[{{"id":"s1","objective":"只做一件事的步骤"}}]}}

规则：
1. 每步只做一件事，步骤不超过 {max_atoms} 个；
2. 每步必须有可核验产物（文件、数据、结论），不许写"继续处理"这类空话；
2.1 不要拆"准备/确认/核对/汇报"这类不产生产物的步骤：核对由核验员负责；
2.2 简单任务（一步能做完）只给一步，不要为了显得完整硬拆；
2.3 不同步骤不许重复查同一事实，后一步只能基于前一步产物继续加工；
3. 缺信息写进 criteria 的"待确认"，不许编造；
4. 确实一步能做完时，steps 只给一步。

任务：{task}"""

EXEC_PROMPT = """你是执行器，本次只负责一个原子步骤。
不要评价整体任务，不要写总结，不要提前做后续步骤。

当前步骤：{objective}

已完成的前置产物：
{context}

要求：
- 需要工具就真实调用，拿到真实结果，不许假装完成；
- 完成后输出一行：STEP_DONE: <本步产物或结论>
- 做不了输出一行：STEP_BLOCKED: <缺什么>"""

VERIFY_PROMPT = """你是核验员。只核验，不修改，不重做任务，不给建议方案。
任务目标：{goal}

验收标准：
{criteria}

各步产物：
{context}

逐条对照标准判断，输出严格 JSON：
{{"pass": true, "issues": ["具体缺口或错误，指明哪一步哪一条"]}}
判定以最终产物是否满足标准为准：执行器已在步骤里给出的工具返回即视为证据，
不要要求它再证明"调用发生过"；某步标 STEP_BLOCKED 但整体产物已满足标准时判 pass。
没有确凿证据指出问题就 pass=true，不要因为"可能不够完美"判失败。"""

_JSON_HINT = re.compile(r"\{.*\}", re.S)


def _parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    match = _JSON_HINT.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


class Pipeline:
    """把一次任务拆成原子步骤逐段执行，再交给独立核验环节判定。

    依赖 Agent 的 `_complete` / `_execute` / sessions / audit / usage，
    不重复实现模型调用与工具权限链路。
    """

    def __init__(
        self,
        agent: Any,
        max_atoms: int = 6,
        atom_steps: int = 4,
        repair_rounds: int = 1,
        context_limit: int = 800,
    ) -> None:
        self.agent = agent
        self.max_atoms = max(1, int(max_atoms))
        self.atom_steps = max(1, int(atom_steps))
        self.repair_rounds = max(0, int(repair_rounds))
        self.context_limit = max(200, int(context_limit))

    # ---- 对外入口 ---------------------------------------------------------
    async def run(
        self,
        task: str,
        session_id: str | None = None,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any] | None:
        """返回结果字典；拆解失败返回 None，让调用方回落到常规则循环。"""
        agent = self.agent
        plan = await self._plan(task, session_id)
        if plan is None or not plan.get("steps"):
            return None

        if session_id is None:
            session_id = agent.sessions.create(task)
        agent._live_session_id = session_id
        agent.sessions.append_message(session_id, 0, {"role": "user", "content": task})
        agent.audit.append(
            session_id, "start", task=task, pipeline=True, atoms=len(plan["steps"])
        )
        await agent._emit(on_event, {"type": "start", "session": session_id, "task": task})

        criteria = [str(item) for item in plan.get("criteria") or []]
        goal = str(plan.get("goal") or task)
        outputs: list[dict[str, str]] = []
        steps_run = 1  # 拆解本身占一步
        blocked = False

        for index, step in enumerate(plan["steps"], 1):
            if agent.cancel_event.is_set():
                return await self._finish(
                    task, session_id, goal, outputs, "cancelled", "任务已被取消", steps_run, None, on_event
                )
            if agent.usage.exceeded():
                return await self._finish(
                    task, session_id, goal, outputs, "incomplete", agent.usage.exceeded() or "", steps_run, None, on_event
                )
            objective = str(step.get("objective") or "").strip()
            if not objective:
                continue
            await agent._emit(on_event, {"type": "atom_start", "atom": index, "objective": objective})
            output = await self._atom(task, objective, outputs, session_id, steps_run + 1)
            steps_run += 1
            if output.startswith("STEP_BLOCKED"):
                blocked = True
            outputs.append({"id": str(step.get("id") or f"s{index}"), "objective": objective, "output": output})
            agent.audit.append(
                session_id, "atom", step=index, objective=objective, blocked=blocked,
                output=_clip(output, 200),
            )
            await agent._emit(on_event, {"type": "atom_end", "atom": index, "blocked": blocked})

        verdict = await self._verify(goal, criteria, outputs, session_id)
        steps_run += 1
        agent.audit.append(session_id, "verify", pass_=verdict["pass"], issues=verdict["issues"])

        rounds = 0
        while not verdict["pass"] and rounds < self.repair_rounds:
            rounds += 1
            issues = "；".join(verdict["issues"]) or "核验未通过但未给出原因"
            await agent._emit(on_event, {"type": "repair_start", "round": rounds})
            output = await self._atom(
                task, f"只修复以下核验问题：{issues}", outputs, session_id, steps_run + 1
            )
            steps_run += 1
            outputs.append({"id": f"fix{rounds}", "objective": f"修复：{issues}", "output": output})
            verdict = await self._verify(goal, criteria, outputs, session_id)
            steps_run += 1
            agent.audit.append(
                session_id, "verify", pass_=verdict["pass"], issues=verdict["issues"], round=rounds
            )

        status = "completed" if verdict["pass"] and not blocked else "incomplete"
        answer = self._compose(goal, outputs, verdict)
        return await self._finish(
            task, session_id, goal, outputs, status, answer, steps_run, verdict, on_event
        )

    # ---- 各阶段 ----------------------------------------------------------
    async def _plan(self, task: str, session_id: str | None) -> dict[str, Any] | None:
        """阶段一：只拆解。工具清单不传入，保证规划器没有执行能力。"""
        messages = [
            {"role": "system", "content": PLAN_PROMPT.format(max_atoms=self.max_atoms, task=task)},
            {"role": "user", "content": "开始拆解。"},
        ]
        reply = await self.agent._complete(messages, [], session_id or "")
        data = _parse_json(str(reply.get("content") or ""))
        if not data:
            return None
        steps = data.get("steps")
        if not isinstance(steps, list):
            return None
        data["steps"] = [step for step in steps[: self.max_atoms] if isinstance(step, dict)]
        return data

    async def _atom(
        self,
        task: str,
        objective: str,
        outputs: list[dict[str, str]],
        session_id: str,
        step_no: int,
    ) -> str:
        """阶段二：单步执行。只看本步目标与前置产物，工具可用，判定权留给核验阶段。"""
        agent = self.agent
        messages = [
            {
                "role": "system",
                "content": EXEC_PROMPT.format(objective=objective, context=self._context(outputs)),
            },
            {"role": "user", "content": f"任务背景：{task}\n只执行当前步骤。"},
        ]
        tools = agent.registry.openai_tools()
        last = ""
        last_tool = ""
        for _ in range(self.atom_steps):
            if agent.cancel_event.is_set():
                return "STEP_BLOCKED: 任务已取消"
            if agent.usage.exceeded():
                return f"STEP_BLOCKED: {agent.usage.exceeded()}"
            reply = await agent._complete(messages, tools, session_id)
            content = str(reply.get("content") or "").strip()
            calls = reply.get("tool_calls") or []
            assistant: dict[str, Any] = {"role": "assistant"}
            if content:
                assistant["content"] = content
            if calls:
                assistant["tool_calls"] = [
                    {
                        "id": tc.get("id") or f"call_{step_no}_{i}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                        },
                    }
                    for i, tc in enumerate(calls)
                ]
            messages.append(assistant)
            if content:
                last = content
            if not calls:
                break
            for tc in calls:
                result = await agent._execute(
                    tc["name"],
                    dict(tc.get("arguments") or {}),
                    str(tc.get("id") or f"call_{step_no}"),
                    session_id,
                    step=step_no,
                )
                text = _clip(result.as_text(), 2000)
                if result.ok:
                    last_tool = text
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tc.get("id") or f"call_{step_no}"),
                        "content": text,
                    }
                )
        if last:
            return last
        if last_tool:
            # 只有工具返回没有结束语时，用工具结果兜底，并标记为已产出，
            # 不把"做完但没写总结"误判成阻塞。
            return f"STEP_DONE(工具返回): {last_tool}"
        return "STEP_BLOCKED: 未产出内容"

    async def _verify(
        self,
        goal: str,
        criteria: list[str],
        outputs: list[dict[str, str]],
        session_id: str,
    ) -> dict[str, Any]:
        """阶段三：独立核验。不给工具、不许修改，只输出 pass 与问题清单。"""
        criteria_text = "\n".join(f"- {item}" for item in criteria) or "- （未给出验收标准，按目标判断）"
        messages = [
            {
                "role": "system",
                "content": VERIFY_PROMPT.format(goal=goal, criteria=criteria_text, context=self._context(outputs)),
            },
            {"role": "user", "content": "开始核验。"},
        ]
        reply = await self.agent._complete(messages, [], session_id)
        data = _parse_json(str(reply.get("content") or "")) or {}
        issues = data.get("issues")
        return {
            "pass": bool(data.get("pass")),
            "issues": [str(item) for item in issues] if isinstance(issues, list) else [],
            "raw": str(reply.get("content") or ""),
        }

    # ---- 收尾 ------------------------------------------------------------
    def _context(self, outputs: list[dict[str, str]]) -> str:
        if not outputs:
            return "-（无）"
        lines = [
            f"[{item['id']}] {item['objective']}\n{_clip(item['output'], self.context_limit)}"
            for item in outputs
        ]
        return "\n\n".join(lines)

    def _compose(self, goal: str, outputs: list[dict[str, str]], verdict: dict[str, Any]) -> str:
        parts = [f"目标：{goal}"]
        for item in outputs:
            parts.append(f"[{item['id']}] {_clip(item['output'], 400)}")
        if verdict["pass"]:
            parts.append("核验：通过")
        else:
            issues = "；".join(verdict["issues"]) or "核验未通过（未给出具体原因）"
            parts.append(f"核验：未通过 — {issues}")
        return "\n\n".join(parts)

    async def _finish(
        self,
        task: str,
        session_id: str,
        goal: str,
        outputs: list[dict[str, str]],
        status: str,
        answer: str,
        steps: int,
        verdict: dict[str, Any] | None = None,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        agent = self.agent
        agent.sessions.finish(session_id, answer, status)
        agent.audit.append(
            session_id, "finish", step=steps, answer=answer, status=status,
            pipeline=True, usage=agent.usage.snapshot(),
        )
        agent._consolidate(task, answer, status, steps, session_id, None)
        if on_event is not None:
            await agent._emit(on_event, {"type": "finish", "session": session_id, "answer": answer})
        return {
            "session_id": session_id,
            "status": status,
            "answer": answer,
            "steps": steps,
            "usage": agent.usage.snapshot(),
            "pipeline": {
                "goal": goal,
                "atoms": len(outputs),
                "verified": bool(verdict and verdict["pass"]),
                "issues": (verdict or {}).get("issues", []),
            },
        }
