from pathlib import Path

import pytest

from hostagent.executor import Executor
from hostagent.gateway import HostRuntimeGateway
from hostagent.governance import Gate, Ledger, PolicyStore
from hostagent.models import Plan, PlanNode


@pytest.fixture()
def runtime(tmp_path, monkeypatch):
    monkeypatch.delenv("HOST_RUNTIME_URL", raising=False)
    policy = PolicyStore(tmp_path / "policy.db")
    ledger = Ledger(tmp_path / "ledger.jsonl")

    def build(dry_run: bool = True, mode: str = "auto") -> Executor:
        return Executor(
            gateway=HostRuntimeGateway(),
            gate=Gate(policy, ledger, mode=mode),
            ledger=ledger,
            output_dir=tmp_path / "output",
            dry_run=dry_run,
        )

    return build, policy, ledger, tmp_path


def test_plan_executes_and_resolves_placeholder(runtime):
    build, _, _, _ = runtime
    plan = Plan("写文档", [
        PlanNode("select_best", "select_best", {"candidates": ["正文内容"], "criteria": "正文"}),
        PlanNode("create_doc", "create_doc", {"title": "测试", "content": "{pick}"},
                 needs=["select_best"]),
    ])
    report = build().run(plan)
    assert report["status"] == "completed"
    assert report["nodes"]["create_doc"]["status"] == "completed"
    assert report["nodes"]["create_doc"]["output"]["dry_run"] is True


def test_external_runtime_never_fakes_success(runtime):
    build, _, _, _ = runtime
    report = build().run(Plan("查法条", [PlanNode("legal_search", "legal_search", {"query": "劳动法"})]))
    node = report["nodes"]["legal_search"]
    assert node["status"] == "unavailable"
    assert "宿主" in node["error"]
    assert report["status"] == "failed"


def test_denied_node_is_not_executed(runtime):
    build, _, _, _ = runtime
    report = build().run(Plan("发消息", [PlanNode("send_message", "send_message", {"text": "hi"})]))
    node = report["nodes"]["send_message"]
    assert node["status"] == "denied"
    assert node["verdict"] == "denied"


def test_preauth_allows_send(runtime):
    build, policy, _, _ = runtime
    policy.add("send_message", "any", "*")
    report = build().run(Plan("发消息", [
        PlanNode("send_message", "send_message", {"text": "hi", "target_id": "oc_1"}),
    ]))
    assert report["nodes"]["send_message"]["status"] == "completed"
    assert report["nodes"]["send_message"]["verdict"] == "preauthorized"


def test_upstream_failure_blocks_downstream(runtime):
    build, _, _, _ = runtime
    plan = Plan("x", [
        PlanNode("select_best", "select_best", {"candidates": [], "criteria": "x"}),
        PlanNode("create_doc", "create_doc", {"title": "t", "content": "{pick}"},
                 needs=["select_best"]),
    ])
    report = build().run(plan)
    assert report["nodes"]["select_best"]["status"] == "failed"
    assert report["nodes"]["create_doc"]["status"] == "blocked"
    assert report["status"] == "failed"


def test_ledger_records_plan_and_finish(runtime):
    build, _, ledger, _ = runtime
    build().run(Plan("x", [PlanNode("format_text", "format_text", {"text": "ok"})]))
    events = [r["event"] for r in ledger.read()]
    assert events[0] == "plan" and events[-1] == "finish"
    ok, _ = ledger.verify()
    assert ok
