import argparse
import json

import pytest

from hostagent import cli
from hostagent.skills import QuotaStore, SkillError, SovereignSkillClient, route_task


def test_quota_accounting(tmp_path):
    quota = QuotaStore(tmp_path / "q.db")
    quota.set_limit("code-review", 2)
    assert quota.remaining("code-review") == 2
    quota.incr("code-review")
    assert quota.used("code-review") == 1
    assert quota.remaining("code-review") == 1


def test_unlimited_quota_is_minus_one(tmp_path):
    quota = QuotaStore(tmp_path / "q.db")
    assert quota.remaining("code-review") == -1


def test_unknown_skill_is_rejected(tmp_path):
    client = SovereignSkillClient(quota=QuotaStore(tmp_path / "q.db"))
    with pytest.raises(SkillError, match="无此能力授权"):
        client.invoke("rm-rf-everything", "x")


def test_missing_binary_is_honest(tmp_path):
    client = SovereignSkillClient(binary="definitely-not-installed", quota=QuotaStore(tmp_path / "q.db"))
    with pytest.raises(SkillError, match="未安装"):
        client.invoke("code-review", "审查这段代码")


def test_exhausted_quota_blocks_invoke(tmp_path):
    quota = QuotaStore(tmp_path / "q.db")
    quota.set_limit("code-review", 1)
    quota.incr("code-review")
    client = SovereignSkillClient(quota=quota)
    with pytest.raises(SkillError, match="额度已用尽"):
        client.invoke("code-review", "审查")


def test_route_without_mapping_is_none(tmp_path):
    assert route_task("写周报", tmp_path / "missing.yaml") is None


def test_cli_run_spec_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path / ".hostagent")
    monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path / "output")
    spec = {
        "intent": "测试执行",
        "nodes": [
            {"id": "select_best", "tool": "select_best",
             "params": {"candidates": ["正文"], "criteria": "正文"}},
            {"id": "create_doc", "tool": "create_doc",
             "params": {"title": "t", "content": "{pick}"}, "needs": ["select_best"]},
        ],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    code = cli.main(["run", "--spec", str(spec_path), "--dry-run", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    assert json.loads(out)["status"] == "completed"


def test_cli_ledger_verify(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path / ".hostagent")
    assert cli.main(["ledger", "verify"]) == 0
    assert "完好" in capsys.readouterr().out


def test_cli_contract_lists_21(capsys):
    assert cli.main(["contract"]) == 0
    out = capsys.readouterr().out
    assert "契约节点 21 个" in out
