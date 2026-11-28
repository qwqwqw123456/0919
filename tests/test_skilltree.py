"""技能树台账：证据折算 A/P/B/C、共现自动加边、导出。"""
from pathlib import Path

from doubao_agent.core.skilltree import DEFAULT_INDEX, TreeLedger

EXEC_OK = {"event": "tool_call", "tool": "host.sandbox_execute_bash", "ok": True}
EXEC_FAIL = {"event": "tool_call", "tool": "host.sandbox_execute_bash", "ok": False}
DENIED = {"event": "permission", "tool": "script.run_shell", "allowed": False, "reason": "红线"}
FINISH = {"event": "finish", "answer": "done"}


def _ledger(tmp_path: Path) -> TreeLedger:
    return TreeLedger(tmp_path / "tree", index_path=DEFAULT_INDEX)


def test_classify_hits_packer_category(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    top = ledger.classify("判断 /tmp/ls_packed 是否加壳，直接用工具箱执行")
    assert top and top[0]["no"] == 4
    assert top[0]["score"] >= 3


def test_ingest_exec_success_upgrades_b_category(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    rec = ledger.ingest(
        "s1",
        "统计 agent 的上下文长度与成本开销",
        [EXEC_OK, FINISH],
        status="completed",
    )
    assert rec and rec["level"] == "A"
    assert rec["cats"][0]["no"] == 28
    state = ledger.state()
    assert state["cats"]["28"]["upgraded"] is True
    assert state["cats"]["28"]["tools"]["host.sandbox_execute_bash"] == 1
    assert state["levels"]["A"] == 24  # 基线 23 + 第 28 类升级


def test_ingest_failure_marks_regressed_without_downgrade(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.ingest("s2", "判断 /tmp/a 是否加壳", [EXEC_OK, FINISH], status="completed")
    rec = ledger.ingest("s3", "判断 /tmp/b 是否加壳", [EXEC_FAIL], status="failed")
    assert rec and rec["level"] == "B"
    state = ledger.state()
    assert state["cats"]["4"]["regressed"] is True
    assert state["cats"]["4"]["level"] == "A"


def test_ingest_denied_only_is_boundary(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    rec = ledger.ingest("s4", "用 sqlmap 打这个站点", [DENIED], status="completed")
    assert rec and rec["level"] == "C"


def test_empty_hits_are_not_recorded(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    assert ledger.ingest("s5", "现在几点？只回答 HH:MM。", [FINISH], status="completed") is None
    assert ledger.evidence() == []


def test_duplicate_session_skipped(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    assert ledger.ingest("s6", "判断 /tmp/a 是否加壳", [EXEC_OK], status="completed")
    assert ledger.ingest("s6", "判断 /tmp/a 是否加壳", [EXEC_OK], status="completed") is None
    assert len(ledger.evidence()) == 1


def test_cooccurrence_creates_edge_outside_manual(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.ingest(
        "s7",
        "分析 UPX 加壳类型，给出逆向脱壳与 IDA 静态分析思路",
        [EXEC_OK, FINISH],
        status="completed",
    )
    state = ledger.state()
    pairs = {(e["from"], e["to"]) for e in state["auto_edges"]}
    assert (4, 5) in pairs
    assert (1, 2) not in pairs  # 手绘边不重复生成


def test_export_matches_page_shape(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.ingest("s8", "判断 /tmp/a 是否加壳", [EXEC_OK], status="completed")
    path = ledger.export(tmp_path / "tree" / "out.json")
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert set(data) >= {"meta", "D", "EV", "edges", "auto_edges"}
    assert data["EV"]["4"]["lv"] == "A"
    assert len(data["EV"]) == 29
