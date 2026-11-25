"""SQLite 统一入口：一处设 WAL 与忙等超时，避免每个 Store 各写一套连接参数。

并发口径：daemon / 定时任务 / CLI 会同时打开同一个 agent.db，
默认回滚日志下读写互阻；统一开 WAL 让读写并行，busy_timeout 给足排队时间。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT = 30.0


def connect(
    path: Path | str,
    *,
    check_same_thread: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    row_factory: Any = None,
) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread, timeout=timeout)
    if row_factory is not None:
        conn.row_factory = row_factory
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.DatabaseError:
        # 内存库或只读介质不支持 WAL 时保持默认，不阻断启动
        pass
    return conn
