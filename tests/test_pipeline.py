from __future__ import annotations

from conftest import FakeSource, _registry

from doubao_agent.core.agent import Agent
from doubao_agent.core.audit import AuditLog
from doubao_agent.core.memory import MemoryStore
from doubao_agent.core.permissions import PermissionPolicy
from doubao_agent.core.pipeline import Pipeline, _parse_json
from doubao_agent.core.session import SessionStore
from doubao_agent.providers.mock import MockProvider

# 两条验收标准：确保不会走"单步 + 无工具 + 标准极简"的降级出口
PLAN = (
    '{"goal":"算 2+2","criteria":["结果是 4","结论里写明 4"],'
    '"steps":[{"id":"s1","objective":"只计算 2+2 的值"}]}'
)


def _reply(content: str) -> dict:
    return {"content": content, "tool_calls": []}


async def _agent(tmp_path, script: list[dict], **pipeline_kwargs) -> Agent:
    registry = await _registry(FakeSource())
    agent = Agent(
        registry=registry,
        provider=MockProvider(script=script),
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(tmp_path / "agent.db"),
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(tmp_path / "agent.db"),
        max_steps=6,
    )
    agent.pipeline = Pipeline(agent, **pipeline_kwargs)
    return agent


def test_parse_json_tolerates_code_fence() -> None:
    assert _parse_json('```json\n{"pass": true}\n```') == {"pass": True}
    assert _parse_json("这不是 JSON") is None


async def test_pipeline_runs_atom_then_verifies(tmp_path) -> None:
    agent = await _agent(
        tmp_path,
        [_reply(PLAN), _reply("STEP_DONE: 4"), _reply('{"pass": true, "issues": []}')],
    )

    result = await agent.run("先算 2+2，然后核对结果并给结论")

    assert result["status"] == "completed"
    assert result["pipeline"]["verified"] is True
    assert result["pipeline"]["atoms"] == 1
    assert "核验：通过" in result["answer"]


async def test_pipeline_repairs_once_then_passes(tmp_path) -> None:
    agent = await _agent(
        tmp_path,
        [
            _reply(PLAN),
            _reply("STEP_DONE: 3"),
            _reply('{"pass": false, "issues": ["s1 结果错误：应为 4"]}'),
            _reply("STEP_DONE: 4"),
            _reply('{"pass": true, "issues": []}'),
        ],
    )

    result = await agent.run("先算 2+2，然后核对结果并给结论")

    assert result["status"] == "completed"
    assert result["pipeline"]["verified"] is True
    assert result["pipeline"]["atoms"] == 2  # 原步骤 + 一次修复


async def test_pipeline_reports_incomplete_when_verification_fails(tmp_path) -> None:
    agent = await _agent(
        tmp_path,
        [
            _reply(PLAN),
            _reply("STEP_DONE: 5"),
            _reply('{"pass": false, "issues": ["答案是 5，与 4 不符"]}'),
            _reply("STEP_DONE: 5"),
            _reply('{"pass": false, "issues": ["仍然是 5"]}'),
        ],
        repair_rounds=1,
    )

    result = await agent.run("先算 2+2，然后核对结果并给结论")

    assert result["status"] == "incomplete"  # 不许把没通过核验的结果标成完成
    assert result["pipeline"]["issues"] == ["仍然是 5"]
    assert "核验：未通过" in result["answer"]


async def test_invalid_plan_falls_back_to_react_loop(tmp_path) -> None:
    agent = await _agent(tmp_path, [_reply("我不会拆解"), _reply("4")])

    result = await agent.run("先算 2+2，然后核对结果并给结论")

    assert result["status"] == "completed"
    assert "pipeline" not in result


PLAN_WITH_OUTPUT = (
    '{"goal":"写结论文件","criteria":["结论文件存在且写有 4","回答里写明 4"],'
    '"steps":[{"id":"s1","objective":"写出结论文件","output":"%s"}]}'
)


async def test_missing_declared_artifact_fails_even_if_model_says_pass(tmp_path) -> None:
    missing = tmp_path / "never" / "结论.md"
    missing.parent.mkdir(parents=True, exist_ok=True)
    plan = PLAN_WITH_OUTPUT % missing
    agent = await _agent(
        tmp_path,
        [_reply(plan), _reply("STEP_DONE: 已经写完文件"), _reply('{"pass": true, "issues": []}')],
    )

    result = await agent.run("写结论文件，然后核对文件在不在")

    assert result["status"] == "incomplete"
    assert result["pipeline"]["verified"] is False
    assert any("不存在" in issue for issue in result["pipeline"]["issues"])


async def test_existing_declared_artifact_passes(tmp_path) -> None:
    target = tmp_path / "结论.md"
    target.write_text("结论：4", encoding="utf-8")
    plan = PLAN_WITH_OUTPUT % target
    agent = await _agent(
        tmp_path,
        [_reply(plan), _reply("STEP_DONE: 已经写完文件"), _reply('{"pass": true, "issues": []}')],
    )

    result = await agent.run("写结论文件，然后核对文件在不在")

    assert result["status"] == "completed"
    assert result["pipeline"]["checks"][0]["ok"] is True
