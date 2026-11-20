import importlib.util
import json
import sys
from pathlib import Path

import pytest

SKILL_MAIN = Path(__file__).resolve().parents[1] / "skills" / "trae-daily-checkin" / "main.py"

# 用与客户端同源算法生成的密文样本（明文见下方断言），仅用于测试解密链路。
STORAGE_FIXTURE = (
    "dGMFEAAAAAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8q2YcLbi+K+4B3MNR4RYSm6y5GS3RAt2FSKfY0"
    "1OUUX3CjLTG6h3jR8jH2ANhJ4mMbVksIB7O+enkN8K28hmrV2bAT7/7xB7rJCO0pvKXECejkZ/woOBuZtkVAvOf8S"
    "M2lOWJZvylgsPq7BeAIjjZJwXyLnX0h7Msy7sDjjzOKpVMmm4XCOQKUMHFsKSPrr74="
)


def _load_skill():
    spec = importlib.util.spec_from_file_location("trae_daily_checkin_skill", SKILL_MAIN)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tdc = _load_skill()


def _creds() -> "tdc.Credentials":
    return tdc.Credentials(token="token", device_id="device", source="test")


def test_aes128_decrypt_matches_fips197_vector():
    key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    ciphertext = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")
    plain = tdc._decrypt_block(ciphertext, tdc._expand_key(key))
    assert plain.hex() == "00112233445566778899aabbccddeeff"


def test_decrypt_storage_value():
    auth = json.loads(tdc.decrypt_storage_value(STORAGE_FIXTURE))
    assert auth["token"] == "sample-token"
    assert auth["userId"] == "10001"
    assert auth["userRegion"]["region"] == "CN"


def test_decrypt_rejects_tampered_ciphertext():
    raw = bytearray(STORAGE_FIXTURE.encode())
    raw[-4] = ord("A") if raw[-4] != ord("A") else ord("B")
    with pytest.raises(ValueError):
        tdc.decrypt_storage_value(raw.decode())


def test_extract_device_id_prefers_device_center_key(tmp_path):
    storage = {"iCubeAuthInfo://icube-dc:real-device": "x", "telemetry.devDeviceId": "telemetry"}
    assert tdc._extract_device_id(tmp_path, storage) == "real-device"


def test_extract_device_id_from_tiny_storage(tmp_path):
    aha = tmp_path / "aha"
    aha.mkdir()
    (aha / "TinyStorage").write_text(
        json.dumps({"tiny_storage_data": {"aha.device.device_id": '{"device_id_str":"123456"}'}}),
        encoding="utf-8",
    )
    assert tdc._extract_device_id(tmp_path, {"telemetry.devDeviceId": "fallback"}) == "123456"


def test_extract_device_id_telemetry_fallback(tmp_path):
    assert tdc._extract_device_id(tmp_path, {"telemetry.devDeviceId": "dev-1"}) == "dev-1"


def test_resolve_credentials_from_env(monkeypatch):
    monkeypatch.setenv("TRAE_TOKEN", "env-token")
    monkeypatch.setenv("TRAE_DEVICE_ID", "env-device")
    creds = tdc.resolve_credentials()
    assert creds.token == "env-token"
    assert creds.device_id == "env-device"
    assert creds.source == "环境变量"


def test_status_action_only_queries(monkeypatch):
    calls = []

    def fake_post(path, creds, with_device, retries=tdc.MAX_RETRIES):
        calls.append(path)
        return {"code": 0, "checked_in": False, "enable": True, "credits": 150, "extra_credits": 50}

    monkeypatch.setattr(tdc, "resolve_credentials", lambda storage=None: _creds())
    monkeypatch.setattr(tdc, "_post", fake_post)

    result = tdc.run("status")
    assert result["result"] == "not_checked_in"
    assert calls == [tdc.STATUS_PATH]


def test_checkin_claims_and_refreshes(monkeypatch):
    responses = [
        {"code": 0, "checked_in": False, "enable": True, "credits": 150, "extra_credits": 50},
        {"code": 0, "message": "success"},
        {"code": 0, "checked_in": True, "enable": True, "credits": 150, "extra_credits": 50},
    ]

    def fake_post(path, creds, with_device, retries=tdc.MAX_RETRIES):
        assert with_device is (path == tdc.CLAIM_PATH)
        return responses.pop(0)

    monkeypatch.setattr(tdc, "resolve_credentials", lambda storage=None: _creds())
    monkeypatch.setattr(tdc, "_post", fake_post)

    result = tdc.run("checkin")
    assert result["result"] == "claimed"
    assert result["checked_in"] is True
    assert result["credits"] == 150
    assert responses == []


def test_checkin_is_idempotent(monkeypatch):
    calls = []

    def fake_post(path, creds, with_device, retries=tdc.MAX_RETRIES):
        calls.append(path)
        return {"code": 0, "checked_in": True, "enable": True, "credits": 150}

    monkeypatch.setattr(tdc, "resolve_credentials", lambda storage=None: _creds())
    monkeypatch.setattr(tdc, "_post", fake_post)

    result = tdc.run("checkin")
    assert result["result"] == "already_checked_in"
    assert calls == [tdc.STATUS_PATH]


def test_claim_error_is_reported(monkeypatch):
    responses = [
        {"code": 0, "checked_in": False, "enable": True},
        {"code": 9004, "message": "bad request"},
    ]

    def fake_post(path, creds, with_device, retries=tdc.MAX_RETRIES):
        return responses.pop(0)

    monkeypatch.setattr(tdc, "resolve_credentials", lambda storage=None: _creds())
    monkeypatch.setattr(tdc, "_post", fake_post)

    with pytest.raises(tdc.TraeApiError) as excinfo:
        tdc.run("checkin")
    assert excinfo.value.code == 9004
    assert "9004" in str(excinfo.value)
