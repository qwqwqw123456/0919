from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any


def _watch_dirs(config: Any) -> list[Path]:
    raw = config.watch_dirs or [config.workspace]
    base = Path(config.workspace)
    dirs: list[Path] = []
    for item in raw:
        path = Path(item)
        if not path.is_absolute():
            path = base / path
        dirs.append(path)
    return dirs


def _snapshot(dirs: list[Path], suffixes: list[str]) -> dict[str, float]:
    """目录文件 mtime 快照；不可读文件跳过，不抛异常。"""
    snap: dict[str, float] = {}
    wanted = tuple(s.lower() for s in suffixes)
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if wanted and path.suffix.lower() not in wanted:
                continue
            try:
                snap[str(path)] = path.stat().st_mtime
            except OSError:
                continue
    return snap


def diff_snapshot(prev: dict[str, float], curr: dict[str, float]) -> list[dict[str, str]]:
    """对比两次快照，返回新增/修改/删除事件。"""
    events: list[dict[str, str]] = []
    for path, mtime in curr.items():
        if path not in prev:
            events.append({"event": "created", "path": path})
        elif prev[path] != mtime:
            events.append({"event": "modified", "path": path})
    for path in prev:
        if path not in curr:
            events.append({"event": "deleted", "path": path})
    return events


async def _run_watcher(runtime: Any, bus: Any) -> None:
    """轮询目录快照，变更即广播；配置了任务模板才触发一次运行。

    首批快照只建基线、不上报；循环内异常不得杀死后台任务。
    """
    config = runtime.config
    dirs = _watch_dirs(config)
    suffixes = list(config.watch_suffixes)
    prev = _snapshot(dirs, suffixes)
    inflight: set[str] = set()
    recent: dict[str, float] = {}
    cooldown = max(0.0, float(config.watch_interval))
    tasks: set[asyncio.Task[Any]] = set()

    async def dispatch(event: dict[str, str]) -> None:
        path, event_type = event["path"], event["event"]
        try:
            bus.publish({"type": "file_change", "event": event_type, "path": path})
            if config.watch_task:
                task_text = config.watch_task.replace("{path}", path).replace("{event}", event_type)
                await runtime.start_run(task_text)
        except Exception:  # noqa: BLE001 - 单条变更不得终止监听循环
            pass
        finally:
            inflight.discard(path)

    try:
        while True:
            await asyncio.sleep(config.watch_interval)
            try:
                curr = _snapshot(dirs, suffixes)
                now = time.time()
                for event in diff_snapshot(prev, curr):
                    key = f"{event['event']}:{event['path']}"
                    if key in inflight:
                        continue
                    # 时间窗内同一路径同类事件合并，避免连续写入重复触发。
                    last = recent.get(key, 0.0)
                    if event["event"] != "deleted" and now - last < cooldown:
                        continue
                    recent[key] = now
                    inflight.add(event["path"])
                    task = asyncio.create_task(dispatch(event))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                prev = curr
            except Exception:  # noqa: BLE001 - 快照失败保留下限快照，下轮再试
                pass
    finally:
        for task in list(tasks):
            task.cancel()
