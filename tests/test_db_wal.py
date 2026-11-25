from __future__ import annotations

import multiprocessing as mp
import sqlite3

from doubao_agent.core import db
from doubao_agent.core.profile import ProfileStore


def _writes(path: str, tag: str, count: int, queue) -> None:
    from doubao_agent.core import db as db_mod

    conn = db_mod.connect(path)
    errors = 0
    for i in range(count):
        try:
            conn.execute("INSERT INTO t(v) VALUES (?)", (f"{tag}-{i}",))
            conn.commit()
        except sqlite3.OperationalError:
            errors += 1
    conn.close()
    queue.put((tag, errors))


def test_wal_and_busy_timeout_defaults(tmp_path):
    path = tmp_path / "a.db"
    conn = db.connect(path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 25000
    conn.close()


def test_store_uses_wal(tmp_path):
    store = ProfileStore(tmp_path / "agent.db")
    store.set("称呼", "老板")
    store.close()
    check = sqlite3.connect(str(tmp_path / "agent.db"))
    assert check.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    check.close()


def test_concurrent_writers_do_not_lock(tmp_path):
    path = tmp_path / "c.db"
    init = db.connect(path)
    init.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)")
    init.commit()
    init.close()

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_writes, args=(str(path), f"p{i}", 200, queue)) for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    results = [queue.get() for _ in procs]
    assert all(errors == 0 for _, errors in results)
    final = db.connect(path)
    assert final.execute("SELECT count(*) FROM t").fetchone()[0] == 800
    final.close()
