from __future__ import annotations

import asyncio
import time

from doubao_agent.skills import SkillManager
from doubao_agent.sources.skill import SkillSource
from doubao_agent.types import ToolCall


def _write_skill(root, name: str, script: str, timeout: float = 60) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: 测试\nentry: main.py\ntimeout: {timeout}\n---\n\n正文",
        "utf-8",
    )
    (d / "main.py").write_text(script, "utf-8")


def test_cancel_before_spawn(tmp_path):
    _write_skill(tmp_path, "slow", "import time; time.sleep(30)")
    manager = SkillManager(tmp_path)
    manager.refresh()
    event = asyncio.Event()
    event.set()
    source = SkillSource(manager, cancel_event=event)

    started = time.perf_counter()
    result = asyncio.run(
        source.call_tool(ToolCall(name="slow", arguments={}, call_id="c1"))
    )
    assert result.ok is False
    assert result.error == "cancelled"
    assert time.perf_counter() - started < 1.0


def test_timeout_kills_child_process_group(tmp_path):
    marker = tmp_path / "marker.txt"
    _write_skill(
        tmp_path,
        "spawner",
        "import subprocess, time\n"
        f"subprocess.Popen(['/bin/bash','-c','sleep 2; echo leaked > {marker}'])\n"
        "time.sleep(10)\n",
        timeout=0.3,
    )
    manager = SkillManager(tmp_path)
    manager.refresh()
    source = SkillSource(manager)

    started = time.perf_counter()
    result = asyncio.run(
        source.call_tool(ToolCall(name="spawner", arguments={}, call_id="c1"))
    )
    assert result.error and "timeout" in result.error
    assert time.perf_counter() - started < 5

    time.sleep(2.5)
    assert not marker.exists()  # 子进程被整组回收，不会在超时后继续落盘
