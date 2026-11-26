"""模型通道：切通道只改配置、探活如实报错、凭证不外显。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import yaml

from doubao_agent.core.model_channels import (
    channels,
    find_channel,
    mask_key,
    probe,
    resolve_key,
    switch_channel,
)


def _cfg(**kw) -> SimpleNamespace:
    base = dict(
        model={"provider": "openai", "base_url": "https://a/v1", "model": "m1", "api_key_env": "K1"},
        model_fallback={},
        models=[],
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_mask_key_keeps_only_head_and_tail() -> None:
    assert mask_key("") == "(未配置)"
    assert mask_key("short") == "****"
    assert mask_key("ak_2abcdefghijklOv7L") == "ak_2********Ov7L"


def test_channels_marks_current_and_includes_registry() -> None:
    cfg = _cfg(models=[{"name": "kimi", "base_url": "https://b/v1", "model": "k2"}])
    rows = channels(cfg)
    assert [r["name"] for r in rows] == ["main", "kimi"]
    assert rows[0]["current"] is True


def test_switch_channel_moves_old_main_to_fallback(tmp_path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"base_url": "https://a/v1", "model": "m1"},
                "models": [{"name": "kimi", "base_url": "https://b/v1", "model": "k2"}],
                "max_steps": 40,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    info = switch_channel(str(path), "kimi")
    assert info["model"] == "k2"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["model"]["base_url"] == "https://b/v1"
    assert data["model_fallback"]["model"] == "m1"
    assert data["max_steps"] == 40  # 其余配置不动


def test_switch_unknown_channel_raises(tmp_path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump({"model": {"model": "m1"}}), encoding="utf-8")
    try:
        switch_channel(str(path), "nope")
    except KeyError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当报错")


def test_probe_without_key_reports_missing_env(monkeypatch) -> None:
    monkeypatch.delenv("K_MISSING", raising=False)
    channel = {"base_url": "https://a/v1", "model": "m1", "api_key_env": "K_MISSING"}
    result = asyncio.run(probe(channel))
    assert result["ok"] is False
    assert "K_MISSING" in result["error"]


def test_find_channel_and_resolve_key(monkeypatch) -> None:
    monkeypatch.setenv("K2", "sk-test-1234")
    cfg = _cfg(models=[{"name": "fast", "base_url": "https://b/v1", "model": "m2", "api_key_env": "K2"}])
    row = find_channel(cfg, "fast")
    assert row is not None and resolve_key(row) == "sk-test-1234"
    assert find_channel(cfg, "none") is None
