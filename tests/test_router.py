from __future__ import annotations

from conftest import FakeSource, _registry

from doubao_agent.core.agent import Agent
from doubao_agent.core.audit import AuditLog
from doubao_agent.core.memory import MemoryStore
from doubao_agent.core.permissions import PermissionPolicy
from doubao_agent.core.pipeline import Pipeline, VERIFY_PROMPT
from doubao_agent.core.router import decide_route, is_sampled, stratum_of
from doubao_agent.core.session import SessionStore
from doubao_agent.providers.mock import MockProvider

# ---- 规则分流：不调模型，纯字符串判定 ------------------------------------


def test_route_marks_short_fact_question_as_single() -> None:
    for task in ("1+1 等于几", "内存总量是多少", "TCP 和 UDP 的区别"):
        assert decide_route(task)["route"] == "single", task


def test_route_sends_multistep_and_deliverables_to_pipeline() -> None:
    cases = {
        "帮我写一份周报": "交付物",
        "先查内存，然后给出结论": "多步",
        "把这三份表格汇总成一份报告": "交付物",
        "删除旧归档并核对残留": "高风险",
        "给出结论并附证据链接": "要求核验",
    }
    for task, expected in cases.items():
        decision = decide_route(task)
        assert decision["route"] == "pipeline", task
        assert expected in decision["reason"], (task, decision["reason"])


def test_route_long_task_is_pipeline() -> None:
    task = "帮我看一下这个项目的结构和依赖，再整理一份可以照着做的说明，最后检查有没有漏项"
    assert decide_route(task)["route"] == "pipeline"


def test_route_empty_task_is_single() -> None:
    assert decide_route("  ")["route"] == "single"


# ---- 分流落地：单循环、升级、降级 ---------------------------------------


def _reply(content: str, calls: list[dict] | None = None) -> dict:
    return {"content": content, "tool_calls": calls or []}


async def _agent(tmp_path, script: list[dict], max_steps: int = 6) -> Agent:
    registry = await _registry(FakeSource())
    agent = Agent(
        registry=registry,
        provider=MockProvider(script=script),
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(tmp_path / "agent.db"),
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(tmp_path / "agent.db"),
        max_steps=max_steps,
    )
    agent.pipeline = Pipeline(agent, max_atoms=3, atom_steps=2)
    return agent


async def test_simple_task_stays_in_single_loop(tmp_path) -> None:
    agent = await _agent(tmp_path, [_reply("4")])

    result = await agent.run("1+1 等于几")

    assert result["status"] == "completed"
    assert result["route"]["decision"] == "single"
    assert result["route"]["escalated"] is False
    assert "pipeline" not in result
    assert result["usage"]["calls"] == 1  # 硬闸内，没有多余核验调用


async def test_incomplete_single_loop_escalates_to_pipeline(tmp_path) -> None:
    agent = await _agent(
        tmp_path,
        [
            _reply(None, [{"id": "c1", "name": "fake.ping", "arguments": {}}]),  # 单循环只准跑一步
            _reply('{"goal":"查一下","criteria":["有结论","有数字"],"steps":'
                   '[{"id":"s1","objective":"调工具拿数据"}]}'),
            _reply("STEP_DONE: 结果 4"),
            _reply('{"pass": true, "issues": []}'),
        ],
        max_steps=1,
    )

    result = await agent.run("查一下 4 是多少")

    assert result["route"]["escalated"] is True
    assert result["status"] == "completed"


async def test_single_atom_without_tools_deescalates(tmp_path) -> None:
    agent = await _agent(
        tmp_path,
        [
            _reply('{"goal":"算 2+2","criteria":["结果是 4"],"steps":'
                   '[{"id":"s1","objective":"直接算出 2+2"}]}'),
            _reply("STEP_DONE: 4"),
        ],
    )

    result = await agent.run("先算 2+2，然后给结论")

    assert result["pipeline"]["deescalated"] is True
    assert result["usage"]["calls"] == 2  # 跳过了核验调用
    assert result["status"] == "completed"


# ---- 两个必须留住的回归坑 ----------------------------------------------


async def test_tool_success_without_final_text_is_not_blocked(tmp_path) -> None:
    """坑 1：工具成功但模型没写结论，不能误判 STEP_BLOCKED。"""
    agent = await _agent(
        tmp_path,
        [
            _reply('{"goal":"ping","criteria":["拿到 pong","写明结果"],"steps":'
                   '[{"id":"s1","objective":"调用 ping"}]}'),
            _reply(None, [{"id": "c1", "name": "fake.ping", "arguments": {}}]),
            _reply(""),  # 没有任何结束语
            _reply('{"pass": true, "issues": []}'),
        ],
    )

    result = await agent.run("先 ping 一下，然后核对返回")

    assert result["status"] == "completed"
    first_atom = result["answer"].split("[s1]")[1]
    assert "STEP_BLOCKED" not in first_atom
    assert "STEP_DONE(工具返回)" in first_atom


def test_verifier_must_not_demand_proof_of_invocation() -> None:
    """坑 2：核验器不得要求执行者"再证明调用发生过"。"""
    assert "不要要求它再证明" in VERIFY_PROMPT


def test_risk_always_pipeline_even_if_short():
    decision = decide_route("删除临时文件")
    assert decision["route"] == "pipeline"
    assert decision["risk"] is True


def test_image_task_goes_pipeline_sidebut_keeps_image_stratum():
    decision = decide_route("这张图里写了什么", has_image=True)
    assert decision["route"] == "pipeline"
    assert "图片" in decision["reason"]
    assert stratum_of(decision, 1, True) == "image"


def test_stratum_split_between_qa_and_tool():
    assert stratum_of({"decision": "single", "risk": False}, 1) == "simple_qa"
    assert stratum_of({"decision": "single", "risk": False}, 2) == "simple_tool"
    assert stratum_of({"decision": "pipeline", "risk": False}, 3) == "multi_step"
    assert stratum_of({"decision": "pipeline", "risk": True}, 3) == "high_risk"


def test_sampling_is_deterministic_and_roughly_ten_percent():
    assert is_sampled("abc") == is_sampled("abc")
    hits = sum(is_sampled(f"s{i}") for i in range(1000))
    assert 60 <= hits <= 140
