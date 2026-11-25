"""凭据兜底来源：系统 keyring / 平台 vault。

优先级（环境变量由 `config._load_local_env` 先行注入，最高）：
1. 进程环境（含 `.env.local`）
2. 系统 keyring（`keyring` 包；macOS 走 `security`，Linux 走 `secret-tool`）
3. 平台 vault：`AGENT_VAULT_FILE` 指向的 JSON（600 权限），或 `AGENT_VAULT_CMD` 命令行输出
4. 本机加密库：`AGENT_VAULT_PASSPHRASE` + `cryptography` 解 `~/.dw/vault/agent-secrets.enc`

任何一层不可用都静默跳过，不阻断启动；不伪造"加密"，不可用时如实报告缺失。
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_VAULT_DIR = Path.home() / ".dw" / "vault"
ENC_FILE = "agent-secrets.enc"
KEYRING_SERVICE = "doubao-agent"


def _vault_file() -> Path | None:
    raw = os.environ.get("AGENT_VAULT_FILE")
    return Path(raw).expanduser() if raw else None


def _enc_path() -> Path:
    raw = os.environ.get("AGENT_VAULT_ENC")
    return Path(raw).expanduser() if raw else DEFAULT_VAULT_DIR / ENC_FILE


def _from_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value or None


def _from_vault_file(name: str) -> str | None:
    path = _vault_file()
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entry = data.get(name) if isinstance(data, dict) else None
    if isinstance(entry, dict):  # {"NAME": {"value": "..."} } 也接受
        entry = entry.get("value")
    return str(entry) if entry else None


def _from_vault_cmd(name: str) -> str | None:
    cmd = os.environ.get("AGENT_VAULT_CMD")
    if not cmd:
        return None
    try:
        out = subprocess.run(
            shlex.split(cmd) + [name], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip() if out.returncode == 0 else ""
    return value or None


def _from_keyring(name: str) -> str | None:
    try:
        import keyring  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return keyring.get_password(KEYRING_SERVICE, name) or None
    except Exception:  # 后端未配置/不可达时按不可用处理
        return None


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt  # 延迟导入，非硬依赖

    return Scrypt(salt=salt, length=32, n=2**14, r=8, p=1).derive(passphrase.encode("utf-8"))


def _from_encrypted_vault(name: str) -> str | None:
    """只认显式给了 passphrase 的加密库；没 passphrase 就当没有，不做假解密。"""
    passphrase = os.environ.get("AGENT_VAULT_PASSPHRASE")
    path = _enc_path()
    if not passphrase or not path.is_file():
        return None
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        blob = path.read_bytes()
        salt, nonce, cipher = blob[:16], blob[16:28], blob[28:]
        key = _derive_key(passphrase, salt)
        data = json.loads(AESGCM(key).decrypt(nonce, cipher, None).decode("utf-8"))
    except Exception:
        return None
    bucket = data.get("keys", data) if isinstance(data, dict) else {}
    entry = bucket.get(name) if isinstance(bucket, dict) else None
    if isinstance(entry, dict):
        entry = entry.get("value")
    return str(entry) if entry else None


def resolve_secret(name: str) -> tuple[str | None, str | None]:
    """返回 (值, 来源)；环境变量已经有时直接返回 env，不触碰其他后端。"""
    value = _from_env(name)
    if value:
        return value, "env"
    for source, getter in (
        ("vault-file", _from_vault_file),
        ("keyring", _from_keyring),
        ("vault-cmd", _from_vault_cmd),
        ("vault-enc", _from_encrypted_vault),
    ):
        value = getter(name)
        if value:
            return value, source
    return None, None


def apply_fallback(names: list[str]) -> dict[str, str]:
    """把缺失的变量从兜底层补进环境；返回 {变量: 来源}，绝不回显值。"""
    filled: dict[str, str] = {}
    for name in names:
        if os.environ.get(name):
            continue
        value, source = resolve_secret(name)
        if value and source and source != "env":
            os.environ[name] = value
            filled[name] = source
    return filled


def status() -> list[dict[str, Any]]:
    """各后端的可用性快照，用于 CLI 自查（不读取任何凭据值）。"""
    rows: list[dict[str, Any]] = []
    rows.append({"name": "env", "available": True, "note": "进程环境 + .env.local（最高优先）"})

    try:
        import keyring  # type: ignore[import-not-found]

        backend = getattr(keyring, "get_keyring", lambda: None)()
        rows.append(
            {
                "name": "keyring",
                "available": backend is not None,
                "note": f"service={KEYRING_SERVICE} backend={type(backend).__name__ if backend else '-'}",
            }
        )
    except ImportError:
        rows.append({"name": "keyring", "available": False, "note": "未安装 keyring 包"})

    vf = _vault_file()
    rows.append(
        {
            "name": "vault-file",
            "available": bool(vf and vf.is_file()),
            "note": f"AGENT_VAULT_FILE={vf}" if vf else "未设置 AGENT_VAULT_FILE",
        }
    )
    rows.append(
        {
            "name": "vault-cmd",
            "available": bool(os.environ.get("AGENT_VAULT_CMD")),
            "note": "AGENT_VAULT_CMD 已设置" if os.environ.get("AGENT_VAULT_CMD") else "未设置",
        }
    )
    enc = _enc_path()
    try:
        import cryptography  # noqa: F401

        has_crypto = True
    except ImportError:
        has_crypto = False
    rows.append(
        {
            "name": "vault-enc",
            "available": bool(has_crypto and enc.is_file() and os.environ.get("AGENT_VAULT_PASSPHRASE")),
            "note": (
                f"{enc}（需要 AGENT_VAULT_PASSPHRASE）"
                if has_crypto
                else "cryptography 未安装"
            ),
        }
    )
    return rows


def store_encrypted(entries: dict[str, str], passphrase: str, path: Path | None = None) -> Path:
    """把一组凭据写成 passphrase 加密文件（600）；供初始化/迁移用。"""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    target = Path(path) if path else _enc_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    salt, nonce = os.urandom(16), os.urandom(12)
    key = _derive_key(passphrase, salt)
    payload = json.dumps({"version": 1, "keys": entries}, ensure_ascii=False).encode("utf-8")
    blob = salt + nonce + AESGCM(key).encrypt(nonce, payload, None)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, blob)
    finally:
        os.close(fd)
    os.chmod(target, 0o600)
    return target
