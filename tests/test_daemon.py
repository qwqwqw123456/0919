from __future__ import annotations

import os

from doubao_agent.daemon import alive, paths, read_pid


def test_read_pid_parses_int_and_handles_missing_or_bad(tmp_path) -> None:
    pid_file, _ = paths(tmp_path)
    pid_file.write_text("12345\n", encoding="utf-8")
    assert read_pid(pid_file) == 12345

    missing = tmp_path / "nope.pid"
    assert read_pid(missing) is None

    pid_file.write_text("not-a-number", encoding="utf-8")
    assert read_pid(pid_file) is None


def test_alive_reports_current_process_and_missing_pid() -> None:
    assert alive(os.getpid()) is True
    assert alive(99999999) is False


def test_paths_creates_agent_dir(tmp_path) -> None:
    pid_file, log_file = paths(tmp_path)
    assert pid_file.name == "daemon.pid"
    assert log_file.name == "daemon.log"
    assert pid_file.parent.is_dir()
