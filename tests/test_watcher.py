from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from doubao_agent.core.watcher import _run_watcher, _snapshot, diff_snapshot


def test_snapshot_filters_by_suffix(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.log").write_text("y", encoding="utf-8")

    snap_all = _snapshot([tmp_path], [])
    assert len(snap_all) == 2

    snap_txt = _snapshot([tmp_path], [".txt"])
    assert len(snap_txt) == 1
    assert str(tmp_path / "a.txt") in snap_txt


def test_diff_snapshot_detects_created_modified_deleted() -> None:
    prev = {"f1": 100.0, "f2": 200.0, "f3": 300.0}
    curr = {"f1": 100.0, "f2": 201.0, "f4": 400.0}
    events = diff_snapshot(prev, curr)
    by_path = {e["path"]: e["event"] for e in events}
    assert by_path["f2"] == "modified"
    assert by_path["f4"] == "created"
    assert by_path["f3"] == "deleted"
    assert "f1" not in by_path


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish(self, event: dict) -> None:
        self.events.append(event)


class _FakeRuntime:
    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        self.runs: list[str] = []

    async def start_run(self, task: str) -> None:
        self.runs.append(task)


async def test_watcher_publishes_event_and_dispatches_template(tmp_path: Path) -> None:
    (tmp_path / "base.txt").write_text("baseline", encoding="utf-8")
    config = SimpleNamespace(
        workspace=tmp_path,
        watch_dirs=[],
        watch_interval=0.05,
        watch_task="处理 {event}: {path}",
        watch_suffixes=[],
    )
    runtime = _FakeRuntime(config)
    bus = _FakeBus()

    task = asyncio.create_task(_run_watcher(runtime, bus))
    try:
        # 先等首轮基线快照建立，再新增文件，避免新文件被算进基线。
        await asyncio.sleep(0.15)
        (tmp_path / "new.txt").write_text("new", encoding="utf-8")
        deadline = asyncio.get_event_loop().time() + 3.0
        while not bus.events and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)

        assert bus.events, "未捕获文件变更事件"
        event = bus.events[0]
        assert event["type"] == "file_change"
        assert event["event"] == "created"
        assert event["path"].endswith("new.txt")
        await asyncio.sleep(0.1)
        assert runtime.runs, "模板任务未触发"
        assert runtime.runs[0].startswith("处理 created:")
    finally:
        task.cancel()
