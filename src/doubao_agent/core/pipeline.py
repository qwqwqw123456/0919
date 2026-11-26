"""任务流水线：拆解 → 逐原子步骤执行 → 独立核验 → 定点修复。

职责分离是本模块的唯一目的：规划器不下场执行，执行器只做当前一步且不自评，
核验器不给工具、只按验收标准判对错。任何阶段预算用尽即停，不硬跑。

失败回退：拆解不出合法计划时返回 None，由 Agent 回落到常规则 ReAct 循环。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from .verify import check_claimed_paths, check_file, claims_write, format_evidence, paths_in, summarize

PLAN_PROMPT = """你是任务拆解器。只拆解，不执行，不回答问题。
输出严格 JSON（不要 Markdown 代码块，不要任何解释）：
{{"goal":"一句话目标","criteria":["可判真假的验收标准"],"steps":[{{"id":"s1","objective":"只做一件事的步骤","output":"产物文件的绝对路径，没有就写 null"}}]}}

规则：
1. 每步只做一件事，步骤不超过 {max_atoms} 个；
2. 每步必须有可核验产物（文件、数据、结论），不许写"继续处理"这类空话；
2.0 只有本步真的要写文件时才填 output 绝对路径，否则写 null；不许把"被评估/被检查的对象"当成 output（评估不改原文件）；
2.1 不要拆"准备/确认/核对/汇报"这类不产生产物的步骤：核对由核验员负责；
2.2 简单任务（一步能做完）只给一步，不要为了显得完整硬拆；
2.3 不同步骤不许重复查同一事实，后一步只能基于前一步产物继续加工；
3. 缺信息写进 criteria 的"待确认"，不许编造；
4. 确实一步能做完时，steps 只给一步；
5. 评估/判断/说明类任务最多 2 步：重点是给出结论和依据，不是把事实查个底朝天。

任务：{task}"""

EXEC_PROMPT = """你是执行器，本次只负责一个原子步骤。
不要评价整体任务，不要写总结，不要提前做后续步骤。

当前步骤：{objective}

已完成的前置产物：
{context}

要求：
- 需要工具就真实调用，拿到真实结果，不许假装完成；
- 检索限定目录与深度，或加 timeout，别做无界递归搜索；
- 本步最多 2 次工具调用；关键事实够了就停下写 STEP_DONE，不做全面取证；
- 完成后输出一行：STEP_DONE: <本步产物或结论>
- 做不了输出一行：STEP_BLOCKED: <缺什么>"""

VERIFY_PROMPT = """你是核验员。只核验，不修改，不重做任务，不给建议方案。
任务目标：{goal}

验收标准：
{criteria}

工具实测的确定性检查（比文字更可信，FAIL 就是 FAIL）：
{evidence}

各步产物：
{context}

逐条对照标准判断，输出严格 JSON：
{{"pass": true, "issues": ["具体缺口或错误，指明哪一步哪一条"]}}
判定以最终产物是否满足标准为准：执行器已在步骤里给出的工具返回即视为证据，
不要要求它再证明"调用发生过"；某步标 STEP_BLOCKED 但整体产物已满足标准时判 pass。
没有确凿证据指出问题就 pass=true，不要因为"可能不够完美"判失败。
确定性检查里出现 FAIL 时，pass 必须为 false，不许多解释。
只按上面的验收标准判断，不许自行添加任务里没有的路径、目录或格式要求；
证据不足时判 pass=false 并写明缺哪条标准，不要编造缺口。"""

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
        atom_tool_max: int = 3,  # 实测：列目录→落盘→回读这类一步要 3 次工具，2 次会截断
        deadline_seconds: float = 240.0,
        repair_rounds: int = 1,
        context_limit: int = 800,
    ) -> None:
        self.agent = agent
        self.max_atoms = max(1, int(max_atoms))
        self.atom_steps = max(1, int(atom_steps))
        # 单步工具调用硬上限：不靠模型自觉收敛，超过就带着已有结果收步
        self.atom_tool_max = max(1, int(atom_tool_max))
        self.deadline_seconds = float(deadline_seconds)
        self.repair_rounds = max(0, int(repair_rounds))
        self.context_limit = max(200, int(context_limit))
        # 本步执行中实际调用了多少次工具，用于判断"拆了但没干活"的降级情形
        self._atom_tool_calls = 0

    # ---- 对外入口 ---------------------------------------------------------
    async def run(
        self,
        task: str,
        session_id: str | None = None,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any] | None:
        """返回结果字典；拆解失败返回 None，让调用方回落到常规则循环。"""
        agent = self.agent
        started = time.monotonic()
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
        deescalated = False

        for index, step in enumerate(plan["steps"], 1):
            if agent.cancel_event.is_set():
                return await self._finish(
                    task, session_id, goal, outputs, "cancelled", "任务已被取消", steps_run, None, on_event
                )
            if agent.usage.exceeded():
                return await self._finish(
                    task, session_id, goal, outputs, "incomplete", agent.usage.exceeded() or "", steps_run, None, on_event
                )
            if time.monotonic() - started > self.deadline_seconds:
                return await self._finish(
                    task, session_id, goal, outputs, "incomplete",
                    f"超过整体时限 {self.deadline_seconds:.0f}s，已停止", steps_run, None, on_event,
                )
            objective = str(step.get("objective") or "").strip()
            if not objective:
                continue
            await agent._emit(on_event, {"type": "atom_start", "atom": index, "objective": objective})
            self._atom_tool_calls = 0
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

            # 时限预算：本步跑完已经吃掉大半时限，就不再核验/修复，直接如实收尾
            if time.monotonic() - started > self.deadline_seconds * 0.75:
                agent.audit.append(session_id, "deadline_stop", atoms=len(outputs))
                return await self._finish(
                    task, session_id, goal, outputs, "incomplete",
                    f"本步已用掉 {time.monotonic() - started:.0f}s，超过时限预算的 3/4，停止以免继续烧额度",
                    steps_run, None, on_event,
                )

            # 降级出口：只拆出一步、没动工具、验收标准极简 —— 核验价值为零，直接收尾。
            if (
                len(plan["steps"]) == 1
                and self._atom_tool_calls == 0
                and len(criteria) <= 1
            ):
                deescalated = True
                break

        if deescalated:
            agent.audit.append(session_id, "pipeline_deescalated", atoms=len(outputs))
            answer = self._compose(goal, outputs, {"pass": True, "issues": []})
            return await self._finish(
                task, session_id, goal, outputs, "completed", answer, steps_run,
                {"pass": True, "issues": [], "deescalated": True}, on_event,
            )

        checks = self._artifact_checks(plan, outputs)
        agent.audit.append(
            session_id, "artifact_checks",
            pass_=summarize(checks)["pass"], checks=[c["name"] for c in checks if not c["ok"]],
        )
        verdict = await self._verify(goal, criteria, outputs, session_id, checks)
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
            checks = self._artifact_checks(plan, outputs)
            verdict = await self._verify(goal, criteria, outputs, session_id, checks)
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
        tools = agent._available_tools()
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
            if self._atom_tool_calls >= self.atom_tool_max:
                # 本步工具额度用完，不再让模型继续取证，用已有结果收步
                last = last or last_tool
                break
            for tc in calls:
                if self._atom_tool_calls >= self.atom_tool_max:
                    break
                self._atom_tool_calls += 1
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
        checks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """阶段三：独立核验。确定性检查先于模型判断，FAIL 不许被模型推翻。"""
        checks = checks or []
        hard = summarize(checks)
        criteria_text = "\n".join(f"- {item}" for item in criteria) or "- （未给出验收标准，按目标判断）"
        messages = [
            {
                "role": "system",
                "content": VERIFY_PROMPT.format(
                    goal=goal,
                    criteria=criteria_text,
                    evidence=format_evidence(checks),
                    context=self._context(outputs),
                ),
            },
            {"role": "user", "content": "开始核验。"},
        ]
        reply = await self.agent._complete(messages, [], session_id)
        data = _parse_json(str(reply.get("content") or "")) or {}
        if not data:
            # 核验器没按 JSON 输出时再问一次，别把"解析失败"当成"核验通过/无问题"
            messages.append({"role": "assistant", "content": str(reply.get("content") or "")})
            messages.append({"role": "user", "content": "只输出 JSON：{\"pass\": true|false, \"issues\": [...]}"})
            reply = await self.agent._complete(messages, [], session_id)
            data = _parse_json(str(reply.get("content") or "")) or {}
        if not data:
            hard = summarize(checks)
            return {
                "pass": False,
                "issues": hard["issues"] + ["核验器输出无法解析（两次），按未通过处理"],
                "checks": checks,
                "raw": str(reply.get("content") or ""),
            }
        issues = data.get("issues")
        model_issues = [str(item) for item in issues] if isinstance(issues, list) else []
        passed = bool(data.get("pass")) and hard["pass"]
        merged = list(hard["issues"]) + model_issues
        return {
            "pass": passed,
            "issues": merged,
            "checks": checks,
            "raw": str(reply.get("content") or ""),
        }

    def _artifact_checks(self, plan: dict[str, Any], outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """拆解里声明了产物路径就查文件；执行器说写了文件也回查磁盘。"""
        checks: list[dict[str, Any]] = []
        claimed = set()
        for item in outputs:
            output = str(item.get("output") or "")
            if claims_write(output):
                claimed.update(paths_in(output))
        for step in plan.get("steps") or []:
            if not isinstance(step, dict):
                continue
            declared = step.get("output")
            if isinstance(declared, str) and declared.strip() and declared.strip().lower() != "null":
                for path in paths_in(declared) or [declared.strip()]:
                    target = Path(path).expanduser()
                    # 只查落在本机、且父目录存在的路径，避免把编造路径当失败原因
                    if str(target) in claimed:
                        continue
                    if target.is_absolute() and target.parent.is_dir():
                        # 拆解器的声明只是线索，文件没落地不一定说明执行错，交给核验员看
                        checks.append(check_file(target, hard=False))
                    elif target.is_absolute():
                        checks.append(
                            {
                                "name": f"file:{target.name}",
                                "ok": False,
                                "detail": f"父目录不存在：{target.parent}",
                                "hard": False,
                            }
                        )
        checks.extend(check_claimed_paths(outputs))
        return checks

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
                "checks": (verdict or {}).get("checks", []),
                "deescalated": bool((verdict or {}).get("deescalated")),
            },
        }
