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


def test_tool_fail_counted_without_becoming_edge(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.ingest("s9", "判断 /tmp/a 是否加壳", [EXEC_FAIL], status="failed")
    state = ledger.state()
    cat = state["cats"]["4"]
    assert cat["tools_fail"]["host.sandbox_execute_bash"] == 1
    assert cat["tools"] == {}                      # 失败不计成功工具
    assert state["auto_edges"] == []               # 工具不进边


def test_cooccurrence_creates_edge_outside_manual(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    for sid in ("s7", "s7b"):
        ledger.ingest(
            sid,
            "分析 UPX 加壳类型，给出逆向脱壳与 IDA 静态分析思路",
            [EXEC_OK, FINISH],
            status="completed",
        )
    state = ledger.state()
    pairs = {(e["from"], e["to"]) for e in state["auto_edges"]}
    assert (4, 5) in pairs
    assert (1, 2) not in pairs  # 手绘边不重复生成
    edge = next(e for e in state["auto_edges"] if (e["from"], e["to"]) == (4, 5))
    assert edge["kind"] == "cooccur" and edge["reason"] and edge["explain"]


def test_single_session_cooccurrence_does_not_draw_edge(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.ingest("s10", "分析 UPX 加壳与逆向脱壳", [EXEC_OK], status="completed")
    assert ledger.state()["auto_edges"] == []


def test_command_template_normalization() -> None:
    from doubao_agent.core.skilltree import command_template

    def t(cmd: str):
        return command_template("script.run_shell", {"arguments": "{'command': '" + cmd + "'}"})

    assert t("nmap -sV 1.2.3.4") == "nmap"
    assert t("cd /tmp && upx -d a.bin") == "upx"
    assert t("docker run alpine") == "docker:run"
    assert t("ls -la /tmp") is None            # 通用底座不进模板
    assert t("curl https://x") is None


def test_tool_edges_gated_by_threshold(tmp_path) -> None:
    ev = {"event": "tool_call", "tool": "script.run_shell", "ok": True,
          "arguments": "{'command': 'nmap -sV 10.0.0.1'}"}

    two = _ledger(tmp_path / "two")
    for sid in ("s11", "s12"):
        two.ingest(sid, "内网渗透 端口扫描 nmap", [ev, FINISH], status="completed")
    assert two.state()["tool_edges"] == []          # 未到 3 次

    three = _ledger(tmp_path / "three")
    for sid in ("s13", "s14", "s15"):
        three.ingest(sid, "内网渗透 端口扫描 nmap", [ev, FINISH], status="completed")
    edges = three.state()["tool_edges"]
    assert any(e["tool"] == "nmap" and e["cat"] == 1 and e["count"] == 3 for e in edges)
    assert {e["tool"] for e in edges} == {"nmap"}


def test_snapshot_and_diff(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.ingest("s20", "判断 /tmp/a 是否加壳", [EXEC_OK], status="completed")
    ledger.export(tmp_path / "tree" / "a.json")
    ledger.ingest("s21", "判断 /tmp/b 是否加壳", [EXEC_OK], status="completed")
    ledger.export(tmp_path / "tree" / "b.json")
    d = ledger.diff()
    assert d["ok"] is True
    assert d["evidence_delta"] == 1
    assert d["new_edges"] == [] and d["gone_edges"] == []


def test_export_matches_page_shape(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.ingest("s8", "判断 /tmp/a 是否加壳", [EXEC_OK], status="completed")
    path = ledger.export(tmp_path / "tree" / "out.json")
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert set(data) >= {"meta", "D", "EV", "edges", "auto_edges"}
    assert data["EV"]["4"]["lv"] == "A"
    assert len(data["EV"]) == 29


def test_wordindex_lookup() -> None:
    from doubao_agent.core.wordindex import lookup

    res = lookup("提权", limit=5)
    assert res["ok"] and res["total"] > 20
    assert len(res["hits"]) == 5
    assert res["hits"][0]["no"] in range(1, 30)
    assert lookup("", limit=5)["ok"] is False
