from pathlib import Path

from hostagent.contract import NODES
from hostagent.governance import DENIED, PREAUTH, Gate, Ledger, PolicyStore


def _gate(tmp_path: Path, mode: str = "auto") -> tuple[Gate, PolicyStore, Ledger]:
    policy = PolicyStore(tmp_path / "policy.db")
    ledger = Ledger(tmp_path / "ledger.jsonl")
    return Gate(policy, ledger, mode=mode), policy, ledger


def test_read_only_is_auto_approved(tmp_path):
    gate, _, _ = _gate(tmp_path)
    assert gate.decide(NODES["search_web"], {}).verdict == "auto_approved"
    assert gate.decide(NODES["create_sheet"], {}).allowed


def test_irreversible_denied_without_preauth(tmp_path):
    gate, _, _ = _gate(tmp_path)
    decision = gate.decide(NODES["send_message"], {"text": "hi"})
    assert decision.verdict == DENIED
    assert not decision.allowed


def test_whitelist_preauth_allows_without_human(tmp_path):
    gate, policy, _ = _gate(tmp_path)
    policy.add("send_message", "any", "*")
    decision = gate.decide(NODES["send_message"], {"text": "hi"})
    assert decision.verdict == PREAUTH
    assert decision.allowed and decision.approval_id


def test_target_scoped_preauth_only_matches_target(tmp_path):
    gate, policy, _ = _gate(tmp_path)
    policy.add("send_file", "target", "ou_me")
    assert gate.decide(NODES["send_file"], {"target_user_id": "ou_me"}).allowed
    assert not gate.decide(NODES["send_file"], {"target_user_id": "ou_other"}).allowed


def test_strict_mode_denies_doc(tmp_path):
    gate, _, _ = _gate(tmp_path, mode="strict")
    assert not gate.decide(NODES["create_doc"], {}).allowed
    assert gate.decide(NODES["search_web"], {}).allowed


def test_ledger_hash_chain_verifies(tmp_path):
    gate, _, ledger = _gate(tmp_path)
    gate.decide(NODES["search_web"], {})
    gate.decide(NODES["send_message"], {})
    ok, count = ledger.verify()
    assert ok and count == 2


def test_ledger_detects_tampering(tmp_path):
    gate, _, ledger = _gate(tmp_path)
    gate.decide(NODES["search_web"], {})
    path = tmp_path / "ledger.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("search_web", "send_message")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, _ = ledger.verify()
    assert not ok
