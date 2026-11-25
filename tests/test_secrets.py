from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from doubao_agent.core import secrets as secret_lib


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("OPENAI_API_KEY", "AGENT_VAULT_FILE", "AGENT_VAULT_CMD",
                "AGENT_VAULT_PASSPHRASE", "AGENT_VAULT_ENC"):
        monkeypatch.delenv(key, raising=False)


def test_env_wins(monkeypatch, tmp_path):
    vault = tmp_path / "v.json"
    vault.write_text(json.dumps({"OPENAI_API_KEY": "from-vault"}), "utf-8")
    monkeypatch.setenv("AGENT_VAULT_FILE", str(vault))
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert secret_lib.resolve_secret("OPENAI_API_KEY") == ("from-env", "env")


def test_vault_file_fallback(monkeypatch, tmp_path):
    vault = tmp_path / "v.json"
    vault.write_text(json.dumps({"OPENAI_API_KEY": "sk-vault-demo"}), "utf-8")
    monkeypatch.setenv("AGENT_VAULT_FILE", str(vault))
    assert secret_lib.resolve_secret("OPENAI_API_KEY") == ("sk-vault-demo", "vault-file")


def test_vault_file_accepts_value_objects(monkeypatch, tmp_path):
    vault = tmp_path / "v.json"
    vault.write_text(json.dumps({"OPENAI_API_KEY": {"value": "sk-obj"}}), "utf-8")
    monkeypatch.setenv("AGENT_VAULT_FILE", str(vault))
    assert secret_lib.resolve_secret("OPENAI_API_KEY") == ("sk-obj", "vault-file")


def test_encrypted_vault_roundtrip(monkeypatch, tmp_path):
    target = tmp_path / "agent-secrets.enc"
    secret_lib.store_encrypted({"OPENAI_API_KEY": "sk-enc-demo"}, "pw-123", target)
    assert oct(target.stat().st_mode)[-3:] == "600"
    raw = target.read_bytes()
    assert b"sk-enc-demo" not in raw  # 落盘不是明文

    monkeypatch.setenv("AGENT_VAULT_ENC", str(target))
    monkeypatch.setenv("AGENT_VAULT_PASSPHRASE", "pw-123")
    assert secret_lib.resolve_secret("OPENAI_API_KEY") == ("sk-enc-demo", "vault-enc")


def test_encrypted_vault_wrong_or_missing_passphrase(monkeypatch, tmp_path):
    target = tmp_path / "agent-secrets.enc"
    secret_lib.store_encrypted({"OPENAI_API_KEY": "sk-enc-demo"}, "right", target)
    monkeypatch.setenv("AGENT_VAULT_ENC", str(target))
    assert secret_lib.resolve_secret("OPENAI_API_KEY") == (None, None)
    monkeypatch.setenv("AGENT_VAULT_PASSPHRASE", "wrong")
    assert secret_lib.resolve_secret("OPENAI_API_KEY") == (None, None)


def test_apply_fallback_only_fills_missing(monkeypatch, tmp_path):
    vault = tmp_path / "v.json"
    vault.write_text(json.dumps({"A_KEY": "a", "B_KEY": "b"}), "utf-8")
    monkeypatch.setenv("AGENT_VAULT_FILE", str(vault))
    monkeypatch.setenv("A_KEY", "explicit")
    filled = secret_lib.apply_fallback(["A_KEY", "B_KEY", "C_KEY"])
    assert filled == {"B_KEY": "vault-file"}
    assert os.environ["A_KEY"] == "explicit"
    assert os.environ["B_KEY"] == "b"
    assert "C_KEY" not in os.environ


def test_status_reports_without_values(monkeypatch, tmp_path):
    vault = tmp_path / "v.json"
    vault.write_text(json.dumps({"X": "secret-value"}), "utf-8")
    monkeypatch.setenv("AGENT_VAULT_FILE", str(vault))
    rows = {row["name"]: row for row in secret_lib.status()}
    assert rows["env"]["available"] is True
    assert rows["vault-file"]["available"] is True
    assert "secret-value" not in json.dumps(rows)


def test_config_fills_key_from_vault(monkeypatch, tmp_path):
    from doubao_agent.config import load_config

    vault = tmp_path / "v.json"
    vault.write_text(json.dumps({"DEMO_API_KEY": "sk-demo"}), "utf-8")
    cfg_file = tmp_path / "agent.yaml"
    cfg_file.write_text(
        f"workspace: {tmp_path}\nmodel:\n  provider: openai\n  api_key_env: DEMO_API_KEY\n",
        "utf-8",
    )
    monkeypatch.setenv("AGENT_VAULT_FILE", str(vault))
    cfg = load_config(cfg_file)
    assert cfg.secret_sources == {"DEMO_API_KEY": "vault-file"}
    assert os.environ["DEMO_API_KEY"] == "sk-demo"
