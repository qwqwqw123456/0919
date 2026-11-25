from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils import dumps

try:  # POSIX 跨进程锁；非 POSIX 平台退回进程内锁
    import fcntl
except ImportError:  # pragma: no cover - 仅非 POSIX
    fcntl = None  # type: ignore[assignment]

import threading

_SECRET_KEYS = {
    "token",
    "authorization",
    "password",
    "passwd",
    "secret",
    "cookie",
    "credential",
    "api_key",
    "apikey",
    "private_key",
}
_SECRET_SUFFIXES = ("_token", "_secret", "_password", "_api_key", "_apikey", "_private_key")

# 字段名兜不住的情况（密钥写在 code/command/content 正文里）：按值模式打码
_VALUE_PATTERNS = (
    (re.compile(r"(?i)(bearer)\s+[A-Za-z0-9._\-+/=]{8,}"), r"\1 ***"),
    (re.compile(r"(?i)\b(sk|rk|pk)-[A-Za-z0-9]{8,}"), "sk-***"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA***"),
    (re.compile(r"\bgh[pous]_[A-Za-z0-9]{16,}\b"), "gh*_***"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{4,}"), "jwt-***"),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?key|secret|token|password|passwd|authorization)"
            r"""(\s*[=:]\s*)(["']?)([^\s,;"'}]{4,})"""
        ),
        r"\1\2\3***",
    ),
)


_PLACEHOLDERS = {"***", "bearer", "basic", "token", "undefined", "null"}


def _kv_repl(match: re.Match[str]) -> str:
    """key=value 里只打码真正的值；Bearer/Basic 这类前缀留给上一条规则处理。"""
    value = match.group(4)
    if value.strip("\"'").lower() in _PLACEHOLDERS:
        return match.group(0)
    return f"{match.group(1)}{match.group(2)}{match.group(3)}***"


def _scrub_text(text: str) -> str:
    """把字符串里内嵌的凭据模式替换成 ***；非字符串原样返回。"""
    for pattern, repl in _VALUE_PATTERNS[:-1]:
        text = pattern.sub(repl, text)
    return _VALUE_PATTERNS[-1][0].sub(_kv_repl, text)

# 链上允许的摘要长度：64 = 完整 sha256（新记录），16 = 历史截断记录（兼容旧链）
_HASH_LENGTHS = (64, 16)


def redact(value: Any) -> Any:
    """把疑似凭据的字段替换成 ***，避免 token / 密钥落进审计日志。"""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _SECRET_KEYS or lowered.endswith(_SECRET_SUFFIXES):
                out[key] = "***"
            else:
                out[key] = redact(item)
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _scrub_text(value)
    return value


def _digest(prev: str, record: dict[str, Any], length: int = 64) -> str:
    full = hashlib.sha256((prev + dumps(record, sort_keys=True)).encode("utf-8")).hexdigest()
    return full[:length]


def _hash_ok(stored: str | None) -> bool:
    """摘要必须是合法 hex，且长度只能是 64（完整）或 16（历史）。"""
    if not isinstance(stored, str) or len(stored) not in _HASH_LENGTHS:
        return False
    return all(ch in "0123456789abcdef" for ch in stored)


class AuditLog:
    """Append-only, hash-chained JSONL. Supports replay and tamper detection.

    写入用跨进程文件锁串行化，摘录用完整 sha256（64 hex）；
    校验同时接受历史 16 hex 截断记录，旧链不需要重写。
    """

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self._thread_lock = threading.Lock()

    def _path(self, session_id: str) -> Path:
        return self.runs_dir / session_id / "audit.jsonl"

    def _lock_path(self, path: Path) -> Path:
        return path.with_name(path.name + ".lock")

    def _tail_state(self, path: Path) -> tuple[str, int]:
        """读文件最后一条的 (hash, seq)；文件不存在/为空返回 ("genesis", 0)。"""
        if not path.exists():
            return "genesis", 0
        try:
            size = path.stat().st_size
        except OSError:
            return "genesis"
        if size == 0:
            return "genesis", 0
        with path.open("rb") as handle:
            handle.seek(max(0, size - 4096))
            chunk = handle.read()
        lines = [line for line in chunk.splitlines() if line.strip()]
        while lines:
            try:
                row = json.loads(lines[-1])
                return row.get("hash", "genesis"), int(row.get("seq") or 0)
            except (json.JSONDecodeError, TypeError, ValueError):
                # 定位点切断了首行；丢掉这半行继续往前找
                lines = lines[1:]
        # 单条记录超过 4KB 时回退整文件读取
        last = ""
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = line
        if not last:
            return "genesis", 0
        try:
            row = json.loads(last)
            return row.get("hash", "genesis"), int(row.get("seq") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            return "genesis", 0

    def append(self, session_id: str, event: str, **payload: Any) -> dict[str, Any]:
        path = self._path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._lock_path(path)
        with self._thread_lock, lock_path.open("a+") as lock_handle:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                prev, prev_seq = self._tail_state(path)
                record: dict[str, Any] = redact(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                        "session": session_id,
                        "event": event,
                        **payload,
                    }
                )
                record["seq"] = prev_seq + 1
                digest = _digest(prev, record, 64)  # 与 verify 同口径：摘要不含 prev/hash
                record["prev"] = prev
                record["hash"] = digest
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(dumps(record) + "\n")
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return record

    def read(self, session_id: str, limit: int = 500) -> list[dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows[-limit:]

    def sessions(self) -> list[str]:
        """列出有审计记录的会话，按最近修改时间倒序。"""
        if not self.runs_dir.exists():
            return []
        paths = [p for p in self.runs_dir.iterdir() if (p / "audit.jsonl").is_file()]
        paths.sort(key=lambda p: (p / "audit.jsonl").stat().st_mtime, reverse=True)
        return [p.name for p in paths]

    def export(self, session_id: str, out: Path | str | None = None) -> tuple[Path, int]:
        """校验通过后导出该会话审计副本；链损坏时拒绝导出。"""
        ok, count = self.verify(session_id)
        if not ok:
            raise ValueError(f"审计链校验失败，拒绝导出：{session_id}")
        source = self._path(session_id)
        target = Path(out) if out else self.runs_dir / session_id / "audit.export.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes() if source.exists() else b"")
        return target, count

    def verify(self, session_id: str) -> tuple[bool, int]:
        """校验哈希链：逐条重算，任何一条被改过、断链或摘要不合法都返回 False。

        兼容两种摘要长度：64（完整）与 16（历史截断）；同一条链内不要求长度一致。
        """
        path = self._path(session_id)
        if not path.exists():
            return True, 0
        prev = "genesis"
        seq = 0
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    return False, count
                if not isinstance(record, dict):
                    return False, count
                stored_prev = record.pop("prev", None)
                stored_hash = record.pop("hash", None)
                if not _hash_ok(stored_hash) or not isinstance(stored_prev, str):
                    return False, count
                if "seq" in record:
                    stored_seq = record.get("seq")
                    if not isinstance(stored_seq, int) or stored_seq != seq + 1:
                        return False, count
                    seq = stored_seq
                expected = _digest(prev, record, len(stored_hash))
                if stored_prev != prev or stored_hash != expected:
                    return False, count
                prev = expected
                count += 1
        return True, count
