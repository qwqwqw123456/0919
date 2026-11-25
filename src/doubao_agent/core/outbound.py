"""出站通道：让 Agent 能主动把结果推到人身上（控制台 / 飞书 webhook / SMTP）。

三条纪律写死在代码里，免得"主动"变成骚扰：
1. 同一内容在去重窗口内只发一次（fingerprint 命中即跳过）；
2. 每个通道每天有硬上限（默认 3 条），超了只记账不外发；
3. 发送前统一脱敏（Bearer / sk- / AK·SK 前缀 / api_key= 值），避免把凭证送出去。
"""

from __future__ import annotations

import hashlib
import json
import re
import smtplib
import sqlite3
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Callable
from .db import connect as connect_db

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outbox_channel_time ON outbox(channel, created_at);
"""

_SECRET_PATTERNS = (
    (re.compile(r"(Bearer\s+)([A-Za-z0-9._\-]{8,})"), r"\1********"),
    (re.compile(r"(sk-[A-Za-z0-9]{6})[A-Za-z0-9]+"), r"\1********"),
    (re.compile(r"(AK[A-Za-z0-9]{2})[A-Za-z0-9]{4,}"), r"\1********"),
    (re.compile(r"(api_key=)[^&\s]+"), r"\1********"),
    (re.compile(r"((?:token|secret|password)[\"']?\s*[:=]\s*)[\"']?[A-Za-z0-9._\-]{6,}"), r"\1********"),
)


def redact(text: str) -> str:
    out = text
    for pattern, repl in _SECRET_PATTERNS:
        out = pattern.sub(repl, out)
    return out


@dataclass
class OutboundConfig:
    channel: str = "console"  # console | feishu | smtp
    daily_limit: int = 3
    dedupe_hours: float = 24.0
    feishu_webhook_url: str = ""
    smtp_host: str = ""
    smtp_port: int = 465
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_to: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _http_post_json(url: str, payload: dict[str, Any], timeout: float = 6.0) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - 仅用户配置的 webhook
        resp.read()


def feishu_transport(cfg: OutboundConfig) -> Callable[[str], None]:
    def send(text: str) -> None:
        if not cfg.feishu_webhook_url:
            raise RuntimeError("未配置飞书 webhook（outbound.feishu_webhook_url 或 FEISHU_WEBHOOK_URL）")
        _http_post_json(cfg.feishu_webhook_url, {"msg_type": "text", "content": {"text": text}})

    return send


def smtp_transport(cfg: OutboundConfig) -> Callable[[str], None]:
    def send(text: str) -> None:
        if not (cfg.smtp_host and cfg.smtp_to):
            raise RuntimeError("未配置 SMTP（host / to）")
        msg = MIMEText(text, "plain", "utf-8")
        msg["Subject"] = "Agent 通知"
        msg["From"] = cfg.smtp_user or cfg.smtp_to
        msg["To"] = cfg.smtp_to
        with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, timeout=10) as server:
            if cfg.smtp_user:
                server.login(cfg.smtp_user, cfg.smtp_password)
            server.sendmail(msg["From"], [cfg.smtp_to], msg.as_string())

    return send


class Outbox:
    """发送台账：上限、去重、失败原因都落库，不靠内存。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = connect_db(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def count_since(self, channel: str, since: float) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM outbox WHERE channel = ? AND created_at >= ? "
                "AND status = 'sent'",
                (channel, since),
            ).fetchone()
        return int(row["n"])

    def seen(self, channel: str, fingerprint: str, since: float) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM outbox WHERE channel = ? AND fingerprint = ? AND created_at >= ? "
                "AND status = 'sent' LIMIT 1",
                (channel, fingerprint, since),
            ).fetchone()
        return row is not None

    def record(self, channel: str, fingerprint: str, text: str, status: str, error: str = "") -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO outbox(channel, fingerprint, text, status, error, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (channel, fingerprint, text, status, error, time.time()),
            )
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM outbox ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class Notifier:
    def __init__(
        self,
        config: OutboundConfig,
        db_path: str | Path,
        transport: Callable[[str], None] | None = None,
        *,
        outbox: Outbox | None = None,
    ) -> None:
        self.config = config
        self._own_outbox = outbox is None
        self.outbox = outbox or Outbox(db_path)
        if transport is not None:
            self._transport = transport
        elif config.channel == "feishu":
            self._transport = feishu_transport(config)
        elif config.channel == "smtp":
            self._transport = smtp_transport(config)
        elif config.channel == "console":
            self._transport = print
        else:
            raise ValueError(f"未知出站通道: {config.channel}")

    def send(self, text: str, *, fingerprint: str | None = None, now: float | None = None) -> dict[str, Any]:
        body = redact(str(text).strip())
        if not body:
            return {"ok": False, "status": "empty", "reason": "空内容不发送"}
        key = fingerprint or hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        now = time.time() if now is None else now
        since = now - self.config.dedupe_hours * 3600
        channel = self.config.channel

        if self.outbox.seen(channel, key, since):
            self.outbox.record(channel, key, body, "duplicate")
            return {"ok": False, "status": "duplicate", "reason": "去重窗口内已发过同样内容"}
        if self.outbox.count_since(channel, since=now - 86400) >= self.config.daily_limit:
            self.outbox.record(channel, key, body, "limited")
            return {"ok": False, "status": "limited", "reason": f"今日已发 {self.config.daily_limit} 条，到上限"}

        try:
            self._transport(body)
        except Exception as exc:  # noqa: BLE001 - 发送失败要留痕，不吞
            self.outbox.record(channel, key, body, "failed", str(exc)[:300])
            return {"ok": False, "status": "failed", "reason": str(exc)[:300]}

        self.outbox.record(channel, key, body, "sent")
        return {"ok": True, "status": "sent", "fingerprint": key}

    def close(self) -> None:
        if self._own_outbox:
            self.outbox.close()


def config_from_dict(data: dict[str, Any] | None, env: dict[str, str] | None = None) -> OutboundConfig:
    data = dict(data or {})
    env = env or {}
    cfg = OutboundConfig(
        channel=str(data.get("channel") or env.get("AGENT_OUTBOUND_CHANNEL") or "console"),
        daily_limit=int(data.get("daily_limit", 3)),
        dedupe_hours=float(data.get("dedupe_hours", 24.0)),
        feishu_webhook_url=str(data.get("feishu_webhook_url") or env.get("FEISHU_WEBHOOK_URL") or ""),
        smtp_host=str(data.get("smtp_host") or env.get("SMTP_HOST") or ""),
        smtp_port=int(data.get("smtp_port", env.get("SMTP_PORT", 465))),
        smtp_user=str(data.get("smtp_user") or env.get("SMTP_USER") or ""),
        smtp_password=str(data.get("smtp_password") or env.get("SMTP_PASSWORD") or ""),
        smtp_to=str(data.get("smtp_to") or env.get("SMTP_TO") or ""),
    )
    return cfg
