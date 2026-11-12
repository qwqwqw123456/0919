from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

# 第一层：会话状态文件（工作记忆）。字段固定，开工前必须先读。
FIELDS = (
    "知识库",
    "远程调用",
    "我的名字",
    "用户称呼",
    "记忆点",
    "上次会话日期",
    "新点主题",
    "离线时间线",
    "工作流节点",
    "初始化阶段",
    "待回顾",
    "暂停状态",
    "写入锁",
    "会话计数",
    "时间戳",
    "焦点线",
)

DEFAULTS: dict[str, Any] = {name: "" for name in FIELDS}
DEFAULTS.update({"会话计数": 0, "写入锁": "unlocked", "初始化阶段": "cold"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS working_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    data TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class WorkingMemory:
    """短期记忆：单条会话状态 + 写入锁，每次会话开头先读、结束前回写。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---- 内部读写（调用方负责持锁）----
    def _read(self) -> dict[str, Any]:
        row = self._conn.execute("SELECT data FROM working_state WHERE id = 1").fetchone()
        data: dict[str, Any] = dict(DEFAULTS)
        if row is not None:
            try:
                stored = json.loads(row["data"])
            except (TypeError, json.JSONDecodeError):
                stored = {}
            if isinstance(stored, dict):
                for name in FIELDS:
                    if name in stored:
                        data[name] = stored[name]
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO working_state(id, data, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
            (json.dumps(data, ensure_ascii=False), time.time()),
        )
        self._conn.commit()

    # ---- 公开接口 ----
    def load(self) -> dict[str, Any]:
        with self._mutex:
            return self._read()

    def update(self, **fields: Any) -> dict[str, Any]:
        unknown = set(fields) - set(FIELDS)
        if unknown:
            raise ValueError(f"未知的状态字段：{', '.join(sorted(unknown))}")
        with self._mutex:
            data = self._read()
            if data.get("写入锁") == "locked":
                raise RuntimeError("会话状态正被写入锁占用")
            data.update(fields)
            self._write(data)
            return data

    @contextmanager
    def locked(self) -> Iterator[dict[str, Any]]:
        """进入写入锁：期间禁止其它 update，退出时自动解锁。"""
        with self._mutex:
            data = self._read()
            data["写入锁"] = "locked"
            self._write(data)
            try:
                yield data
            finally:
                data = self._read()
                data["写入锁"] = "unlocked"
                self._write(data)

    def begin_session(self, task: str) -> dict[str, Any]:
        """会话开头调用：读状态、计数 +1、更新日期与焦点线，然后再开口。"""
        with self._mutex:
            data = self._read()
            data["上次会话日期"] = str(data.get("时间戳") or "")
            data["会话计数"] = int(data.get("会话计数") or 0) + 1
            data["时间戳"] = _now()
            data["焦点线"] = _clip(task, 120)
            data["工作流节点"] = "运行中"
            data["初始化阶段"] = "ready"
            data["暂停状态"] = "none"
            self._write(data)
            return data

    def end_session(self, status: str, summary: str = "") -> dict[str, Any]:
        """会话收尾：记录工作流节点、待回顾与暂停状态。"""
        with self._mutex:
            data = self._read()
            data["工作流节点"] = status
            data["初始化阶段"] = "idle"
            data["时间戳"] = _now()
            if status == "completed":
                data["待回顾"] = _clip(summary, 120)
            elif status in ("cancelled", "paused"):
                data["暂停状态"] = status
            self._write(data)
            return data

    def context_text(self) -> str:
        """渲染成可直接注入系统提示的片段。"""
        data = self.load()
        lines = [
            f"- {name}：{value}"
            for name, value in data.items()
            if name != "写入锁" and value not in ("", None, 0)
        ]
        if not lines:
            return ""
        return "会话状态（开工前已读取）：\n" + "\n".join(lines)

    def close(self) -> None:
        with self._mutex:
            self._conn.close()
