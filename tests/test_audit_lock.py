from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

from doubao_agent.core.audit import AuditLog, _digest


def _writer(runs_dir: str, session: str, count: int, tag: str) -> None:
    log = AuditLog(Path(runs_dir))
    for i in range(count):
        log.append(session, "step", tag=tag, index=i)


def test_hash_is_full_sha256(tmp_path):
    log = AuditLog(tmp_path)
    record = log.append("s1", "run_start", task="t")
    assert len(record["hash"]) == 64
    assert record["prev"] == "genesis"
    assert log.verify("s1") == (True, 1)


def test_legacy_16_hex_chain_still_verifies(tmp_path):
    log = AuditLog(tmp_path)
    path = log._path("legacy")
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = "genesis"
    with path.open("w", encoding="utf-8") as handle:
        for i in range(3):
            record = {"ts": f"t{i}", "session": "legacy", "event": "step", "i": i}
            digest = _digest(prev, record, 16)
            record["prev"], record["hash"] = prev, digest
            handle.write(json.dumps(record) + "\n")
            prev = digest
    assert log.verify("legacy") == (True, 3)


def test_mixed_chain_verifies(tmp_path):
    log = AuditLog(tmp_path)
    path = log._path("mixed")
    path.parent.mkdir(parents=True, exist_ok=True)
    first = {"ts": "t0", "session": "mixed", "event": "a"}
    first_hash = _digest("genesis", first, 16)
    first["prev"], first["hash"] = "genesis", first_hash
    path.write_text(json.dumps(first) + "\n", "utf-8")

    record = log.append("mixed", "b")
    assert len(record["hash"]) == 64
    assert log.verify("mixed") == (True, 2)


def test_bad_hash_length_or_non_hex_rejected(tmp_path):
    log = AuditLog(tmp_path)
    log.append("bad", "a")
    path = log._path("bad")
    record = json.loads(path.read_text("utf-8").strip())
    record["hash"] = "z" * 64  # 合法长度但非 hex
    path.write_text(json.dumps(record) + "\n", "utf-8")
    assert log.verify("bad") == (False, 0)

    record["hash"] = "abc123"  # 非 16/64 长度
    path.write_text(json.dumps(record) + "\n", "utf-8")
    assert log.verify("bad") == (False, 0)


def test_tamper_still_detected(tmp_path):
    log = AuditLog(tmp_path)
    log.append("t", "a", n=1)
    log.append("t", "b", n=2)
    path = log._path("t")
    lines = path.read_text("utf-8").splitlines()
    forged = json.loads(lines[0])
    forged["n"] = 9
    lines[0] = json.dumps(forged)
    path.write_text("\n".join(lines) + "\n", "utf-8")
    assert log.verify("t") == (False, 0)


def test_cross_process_append_keeps_chain(tmp_path):
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_writer, args=(str(tmp_path), "multi", 25, f"p{i}"))
        for i in range(4)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0

    log = AuditLog(tmp_path)
    ok, count = log.verify("multi")
    assert ok is True
    assert count == 100
    hashes = [row["hash"] for row in log.read("multi", limit=100)]
    assert len(set(hashes)) == 100  # 没有两个进程拿到同一个 prev
