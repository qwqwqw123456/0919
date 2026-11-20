#!/usr/bin/env python3
"""trae-daily-checkin：读取本机 Trae 登录态，领取每日积分。

流程：定位 storage.json → 解密登录态 → 查签到状态 → 未签到则领取。
只用标准库；令牌仅在内存中使用，不打印、不落盘。

输入（可选，stdin JSON）：{"action": "checkin"|"status", "dry_run": true, "storage": "..."}
命令行等价参数：--status / --dry-run / --storage PATH
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

API_BASE = "https://api.trae.cn"
STATUS_PATH = "/trae/api/v2/ug/checkin_credits/status"
CLAIM_PATH = "/trae/api/v2/ug/checkin_credits/claim"

AUTH_KEY = "iCubeAuthInfo://icube.cloudide"
DEVICE_KEY_PREFIX = "iCubeAuthInfo://icube-dc:"
APP_NAMES = ("TRAE SOLO CN", "Trae CN", "TRAE SOLO", "Trae")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
RETRY_BASE = 1.5
RETRY_JITTER = 0.8
RATE_LIMIT_CODES = {9074, 429}
MSG_INVALID_DEVICE = 9004
MSG_AUTH_FAILED = 1001

SALT_A = bytes([
    82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251,
    124, 227, 57, 130, 155, 47, 255, 135, 52, 142, 67, 68, 196, 222, 233, 203,
    84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66, 250, 195, 78,
    8, 46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37,
])
SALT_B = bytes([
    31, 221, 168, 51, 136, 7, 199, 49, 177, 18, 16, 89, 39, 128, 236, 95,
    96, 81, 127, 169, 25, 181, 74, 13, 45, 229, 122, 159, 147, 201, 156, 239,
    160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97,
    23, 43, 4, 126, 186, 119, 214, 38, 225, 105, 20, 99, 85, 33, 12, 125,
])


# ---------------------------------------------------------------------------
# 最小 AES-128-CBC 解密（仅解密，避免引入第三方依赖）
# ---------------------------------------------------------------------------


def _rotl8(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (8 - shift))) & 0xFF


def _build_tables() -> tuple[list[int], list[int]]:
    sbox = [0] * 256
    p = q = 1
    while True:
        p = (p ^ ((p << 1) & 0xFF) ^ (0x1B if p & 0x80 else 0)) & 0xFF
        q ^= (q << 1) & 0xFF
        q ^= (q << 2) & 0xFF
        q ^= (q << 4) & 0xFF
        q &= 0xFF
        if q & 0x80:
            q ^= 0x09
        sbox[p] = (
            q ^ _rotl8(q, 1) ^ _rotl8(q, 2) ^ _rotl8(q, 3) ^ _rotl8(q, 4) ^ 0x63
        )
        if p == 1:
            break
    sbox[0] = 0x63
    inv = [0] * 256
    for index, value in enumerate(sbox):
        inv[value] = index
    return sbox, inv


SBOX, INV_SBOX = _build_tables()


def _xtime(value: int) -> int:
    return ((value << 1) ^ 0x1B) & 0xFF if value & 0x80 else (value << 1) & 0xFF


def _gmul(a: int, b: int) -> int:
    product = 0
    for _ in range(8):
        if b & 1:
            product ^= a
        a = _xtime(a)
        b >>= 1
    return product


def _expand_key(key: bytes) -> list[list[int]]:
    words = [list(key[index * 4 : index * 4 + 4]) for index in range(4)]
    rcon = 1
    for index in range(4, 44):
        temp = words[index - 1][:]
        if index % 4 == 0:
            temp = [SBOX[value] for value in temp[1:] + temp[:1]]
            temp[0] ^= rcon
            rcon = _xtime(rcon)
        words.append([words[index - 4][j] ^ temp[j] for j in range(4)])
    return [sum(words[round * 4 : round * 4 + 4], []) for round in range(11)]


def _add_round_key(state: list[int], round_key: list[int]) -> list[int]:
    return [value ^ round_key[index] for index, value in enumerate(state)]


def _inv_shift_rows(state: list[int]) -> list[int]:
    out = state[:]
    for row in range(1, 4):
        for column in range(4):
            out[4 * column + row] = state[4 * ((column - row) % 4) + row]
    return out


def _inv_mix_columns(state: list[int]) -> list[int]:
    out = state[:]
    for column in range(4):
        a0, a1, a2, a3 = state[4 * column : 4 * column + 4]
        out[4 * column] = _gmul(a0, 14) ^ _gmul(a1, 11) ^ _gmul(a2, 13) ^ _gmul(a3, 9)
        out[4 * column + 1] = _gmul(a0, 9) ^ _gmul(a1, 14) ^ _gmul(a2, 11) ^ _gmul(a3, 13)
        out[4 * column + 2] = _gmul(a0, 13) ^ _gmul(a1, 9) ^ _gmul(a2, 14) ^ _gmul(a3, 11)
        out[4 * column + 3] = _gmul(a0, 11) ^ _gmul(a1, 13) ^ _gmul(a2, 9) ^ _gmul(a3, 14)
    return out


def _decrypt_block(block: bytes, round_keys: list[list[int]]) -> bytes:
    state = _add_round_key(list(block), round_keys[10])
    for round in range(9, 0, -1):
        state = _inv_shift_rows(state)
        state = [INV_SBOX[value] for value in state]
        state = _add_round_key(state, round_keys[round])
        state = _inv_mix_columns(state)
    state = _inv_shift_rows(state)
    state = [INV_SBOX[value] for value in state]
    return bytes(_add_round_key(state, round_keys[0]))


def aes128_cbc_decrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    if len(key) != 16 or len(iv) != 16 or len(data) % 16:
        raise ValueError("AES-128-CBC 参数长度不合法")
    round_keys = _expand_key(key)
    out = bytearray()
    previous = iv
    for offset in range(0, len(data), 16):
        block = data[offset : offset + 16]
        plain = _decrypt_block(block, round_keys)
        out.extend(a ^ b for a, b in zip(plain, previous))
        previous = block
    return bytes(out)


def _pkcs7_unpad(data: bytes) -> bytes:
    if not data:
        raise ValueError("解密结果为空")
    pad = data[-1]
    if pad < 1 or pad > 16 or data[-pad:] != bytes([pad]) * pad:
        raise ValueError("解密结果填充不合法")
    return data[:-pad]


# ---------------------------------------------------------------------------
# Trae 本地登录态
# ---------------------------------------------------------------------------


@dataclass
class Credentials:
    token: str
    device_id: str = ""
    user_id: str = ""
    region: str = ""
    source: str = ""
    app_root: Path | None = None


class CredentialError(RuntimeError):
    pass


class TraeApiError(RuntimeError):
    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


def decrypt_storage_value(encoded: str) -> str:
    """解密 storage.json 中的 iCube 加密串，返回明文字符串。"""
    raw = base64.b64decode(encoded)
    if len(raw) <= 102:
        raise ValueError("密文过短，不是有效的 Trae 登录态")
    random_part = raw[6:38]
    ciphertext = raw[38:]
    salt = bytes(a ^ b for a, b in zip(SALT_A, SALT_B))
    first = hashlib.sha512(random_part).digest()
    derived = hashlib.sha512(first + salt).digest()
    decrypted = _pkcs7_unpad(aes128_cbc_decrypt(derived[:16], derived[16:32], ciphertext))
    if len(decrypted) < 64:
        raise ValueError("解密内容过短，完整性校验失败")
    expected, plaintext = decrypted[:64], decrypted[64:]
    if hashlib.sha512(plaintext).digest() != expected:
        raise ValueError("完整性校验失败（SHA-512 不匹配）")
    return plaintext.decode("utf-8")


def _decode_maybe_encrypted(value: str) -> str:
    text = value.strip()
    if text.startswith("{"):
        return text
    return decrypt_storage_value(text)


def _app_roots() -> list[tuple[str, Path]]:
    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
    return [(name, base / name) for name in APP_NAMES]


def _storage_candidates(override: str | None = None) -> list[Path]:
    if override:
        return [Path(override).expanduser()]
    env_path = os.environ.get("TRAE_STORAGE")
    if env_path:
        return [Path(env_path).expanduser()]
    return [root / "User" / "globalStorage" / "storage.json" for _, root in _app_roots()]


def _extract_device_id(app_root: Path, storage: dict[str, Any]) -> str:
    for key in storage:
        if isinstance(key, str) and key.startswith(DEVICE_KEY_PREFIX):
            value = key[len(DEVICE_KEY_PREFIX) :].strip()
            if value:
                return value

    tiny = app_root / "aha" / "TinyStorage"
    if tiny.is_file():
        try:
            data = json.loads(tiny.read_text("utf-8"))
            raw = (data.get("tiny_storage_data") or {}).get("aha.device.device_id")
            if raw:
                info = json.loads(_decode_maybe_encrypted(raw))
                for field_name in ("device_id_str", "device_id", "uuid"):
                    value = str(info.get(field_name) or "").strip()
                    if value:
                        return value
        except (OSError, ValueError, TypeError):
            pass

    for key in ("telemetry.devDeviceId", "telemetry.deviceId"):
        value = str(storage.get(key) or "").strip()
        if value:
            return value

    machine_id = str(storage.get("telemetry.machineId") or "").strip()
    if not machine_id:
        machine_file = app_root / "machineid"
        if machine_file.is_file():
            try:
                machine_id = machine_file.read_text("utf-8").strip()
            except OSError:
                machine_id = ""
    if machine_id:
        return hashlib.sha256(machine_id.encode("utf-8")).hexdigest()[:32]
    return ""


def resolve_credentials(storage_override: str | None = None) -> Credentials:
    token = os.environ.get("TRAE_TOKEN", "").strip()
    if token:
        return Credentials(
            token=token,
            device_id=os.environ.get("TRAE_DEVICE_ID", "").strip(),
            user_id=os.environ.get("TRAE_USER_ID", "").strip(),
            region=os.environ.get("TRAE_USER_REGION", "").strip(),
            source="环境变量",
        )

    errors: list[str] = []
    for storage_path in _storage_candidates(storage_override):
        if not storage_path.is_file():
            continue
        try:
            storage = json.loads(storage_path.read_text("utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{storage_path}: {exc}")
            continue
        encoded = storage.get(AUTH_KEY)
        if not isinstance(encoded, str) or not encoded:
            errors.append(f"{storage_path}: 缺少 {AUTH_KEY}")
            continue
        try:
            auth = json.loads(_decode_maybe_encrypted(encoded))
        except (ValueError, TypeError) as exc:
            errors.append(f"{storage_path}: {exc}")
            continue
        if not isinstance(auth, dict) or not (auth.get("token") or auth.get("accessToken")):
            errors.append(f"{storage_path}: 解密后没有 token")
            continue
        app_root = storage_path.parents[2]
        region = ""
        user_region = auth.get("userRegion")
        if isinstance(user_region, dict):
            region = str(user_region.get("region") or "")
        return Credentials(
            token=str(auth.get("token") or auth.get("accessToken")),
            device_id=_extract_device_id(app_root, storage),
            user_id=str(auth.get("userId") or auth.get("uid") or ""),
            region=region,
            source=str(storage_path),
            app_root=app_root,
        )

    detail = f"（{errors[0]}）" if errors else ""
    raise CredentialError(
        "未找到 Trae 本地登录态" + detail + "。请确认本机已安装并登录 Trae 客户端，"
        "或设置环境变量 TRAE_TOKEN / TRAE_DEVICE_ID。"
    )


# ---------------------------------------------------------------------------
# 签到接口
# ---------------------------------------------------------------------------


def _headers(creds: Credentials, with_device: bool) -> dict[str, str]:
    headers = {
        "Authorization": f"Cloud-IDE-JWT {creds.token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Origin": "https://www.trae.cn",
        "Referer": "https://www.trae.cn/",
        "User-Agent": USER_AGENT,
    }
    if with_device and creds.device_id:
        headers["x-device-id"] = creds.device_id
    if creds.user_id:
        headers["x-user-id"] = creds.user_id
    if creds.region:
        headers["X-User-Region"] = creds.region
    return headers


def _read_json(payload: bytes) -> dict[str, Any]:
    if not payload:
        return {}
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _post(path: str, creds: Credentials, with_device: bool, retries: int = MAX_RETRIES) -> dict[str, Any]:
    url = API_BASE + path
    body = json.dumps({"req_source": 1}).encode("utf-8")
    last_error = "未知错误"

    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url, data=body, method="POST", headers=_headers(creds, with_device)
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                parsed = _read_json(response.read())
            if parsed.get("code") in RATE_LIMIT_CODES:
                last_error = f"服务端限流（code={parsed.get('code')}）"
            else:
                return parsed
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise TraeApiError("登录态已失效（HTTP %d），请打开 Trae 客户端重新登录" % exc.code)
            parsed = _read_json(exc.read() if exc.fp else b"")
            if parsed.get("code") in RATE_LIMIT_CODES or exc.code == 429 or exc.code >= 500:
                last_error = f"HTTP {exc.code}"
            elif parsed:
                return parsed
            else:
                last_error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            last_error = f"网络异常：{reason}"

        if attempt < retries:
            time.sleep(RETRY_BASE * attempt + random.uniform(0, RETRY_JITTER))

    raise TraeApiError(f"请求失败（{last_error}），已重试 {retries} 次")


def _claim_error(parsed: dict[str, Any]) -> str:
    code = parsed.get("code")
    if code == MSG_INVALID_DEVICE:
        return "签到失败（code=9004）：设备标识无效，请确认本机 Trae 客户端已正常登录"
    if code == MSG_AUTH_FAILED:
        return "签到失败（code=1001）：登录态已过期，请打开 Trae 客户端重新登录"
    message = str(parsed.get("message") or parsed)
    return f"签到未成功（code={code}）：{message}"


def _status_summary(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "checked_in": bool(parsed.get("checked_in")),
        "credits": int(parsed.get("credits") or 0),
        "extra_credits": int(parsed.get("extra_credits") or 0),
        "enable": bool(parsed.get("enable")),
    }


def run(action: str = "checkin", dry_run: bool = False, storage: str | None = None) -> dict[str, Any]:
    creds = resolve_credentials(storage)
    result: dict[str, Any] = {
        "ok": True,
        "action": action,
        "date": date.today().isoformat(),
        "source": creds.source,
        "device_id_present": bool(creds.device_id),
    }

    status = _post(STATUS_PATH, creds, with_device=False)
    if status.get("code") not in (None, 0):
        raise TraeApiError(f"查询签到状态失败：{status.get('message') or status}")
    summary = _status_summary(status)
    result.update(summary)

    if action == "status" or dry_run:
        result["result"] = "already_checked_in" if summary["checked_in"] else "not_checked_in"
        return result

    if summary["checked_in"]:
        result["result"] = "already_checked_in"
        return result
    if not summary["enable"]:
        result["result"] = "disabled"
        result["message"] = "当前账号的签到活动未启用"
        return result

    claim = _post(CLAIM_PATH, creds, with_device=True)
    if claim.get("code") != 0:
        raise TraeApiError(_claim_error(claim), code=claim.get("code"))

    refreshed = _post(STATUS_PATH, creds, with_device=False, retries=min(2, MAX_RETRIES))
    if refreshed.get("code") in (None, 0):
        result.update(_status_summary(refreshed))
    result["result"] = "claimed"
    result["message"] = "签到成功"
    return result


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def _read_payload() -> dict[str, Any]:
    if sys.stdin is None or sys.stdin.isatty():
        return {}
    try:
        raw = sys.stdin.read()
    except OSError:
        return {}
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def main(argv: list[str] | None = None) -> int:
    payload = _read_payload()
    parser = argparse.ArgumentParser(description="Trae 每日积分自动领取")
    parser.add_argument("--action", choices=["checkin", "status"], default=None)
    parser.add_argument("--status", action="store_true", help="只查询签到状态")
    parser.add_argument("--dry-run", action="store_true", help="只查状态，不领取")
    parser.add_argument("--storage", default=None, help="手动指定 storage.json 路径")
    args = parser.parse_args(argv)

    action = args.action or ("status" if args.status else payload.get("action")) or "checkin"
    if action not in ("checkin", "status"):
        action = "checkin"
    dry_run = args.dry_run or bool(payload.get("dry_run"))
    storage = args.storage or payload.get("storage") or None

    try:
        result = run(action=action, dry_run=dry_run, storage=storage)
    except CredentialError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    except TraeApiError as exc:
        output: dict[str, Any] = {"ok": False, "error": str(exc)}
        if exc.code is not None:
            output["code"] = exc.code
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
