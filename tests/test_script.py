from __future__ import annotations

from doubao_agent.types import ToolCall


async def test_script_mode_runs_python_in_one_call(tmp_path) -> None:
    from doubao_agent.sources.script import ScriptSource

    source = ScriptSource(tmp_path)
    specs = await source.list_tools()
    assert {spec.name for spec in specs} == {"run_python", "run_shell"}

    result = await source.call_tool(
        ToolCall(
            name="run_python",
            arguments={"code": "from pathlib import Path\nPath('out.txt').write_text('hi')\nprint((2 + 3) * 7)"},
            call_id="s1",
        )
    )
    assert result.ok
    assert "35" in result.content
    assert (tmp_path / "out.txt").read_text() == "hi"


async def test_script_mode_reports_nonzero_exit(tmp_path) -> None:
    from doubao_agent.sources.script import ScriptSource

    source = ScriptSource(tmp_path)
    result = await source.call_tool(
        ToolCall(name="run_shell", arguments={"command": "exit 3"}, call_id="s2")
    )
    assert result.ok is False
    assert "exit code 3" in (result.error or "")


async def test_script_rejects_cwd_outside_workspace(tmp_path) -> None:
    from doubao_agent.sources.script import ScriptSource

    source = ScriptSource(tmp_path)
    for bad in ("/", "..", "../other"):
        result = await source.call_tool(
            ToolCall(name="run_shell", arguments={"command": "pwd", "cwd": bad}, call_id="s3")
        )
        assert result.ok is False
        assert "outside workspace" in (result.error or "")


async def test_script_runs_in_workspace_subdir(tmp_path) -> None:
    from doubao_agent.sources.script import ScriptSource

    sub = tmp_path / "sub"
    sub.mkdir()
    source = ScriptSource(tmp_path)
    result = await source.call_tool(
        ToolCall(name="run_shell", arguments={"command": "pwd", "cwd": "sub"}, call_id="s4")
    )
    assert result.ok is True
    assert str(sub.resolve()) in result.content


async def test_script_shell_is_not_login_shell(tmp_path) -> None:
    from doubao_agent.sources.script import ScriptSource

    source = ScriptSource(tmp_path)
    result = await source.call_tool(
        ToolCall(
            name="run_shell",
            arguments={"command": "shopt -q login_shell && echo login || echo plain"},
            call_id="s5",
        )
    )
    assert result.ok is True
    assert "plain" in result.content


async def test_script_timeout_kills_process_group(tmp_path) -> None:
    import asyncio
    import os
    import time

    from doubao_agent.sources.script import ScriptSource

    pidfile = tmp_path / "child.pid"
    source = ScriptSource(tmp_path)
    result = await source.call_tool(
        ToolCall(
            name="run_shell",
            arguments={"command": f"echo $$ > {pidfile}; sleep 30", "timeout": 0.5},
            call_id="s6",
        )
    )
    assert result.ok is False
    assert "timeout" in (result.error or "")

    pid = int(pidfile.read_text().strip())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"shell process {pid} survived timeout")


async def test_script_reports_cancelled_when_event_set(tmp_path) -> None:
    import asyncio

    from doubao_agent.sources.script import ScriptSource

    event = asyncio.Event()
    event.set()
    source = ScriptSource(tmp_path, cancel_event=event)

    result = await source.call_tool(
        ToolCall(name="run_python", arguments={"code": "print(1)"}, call_id="s1")
    )

    assert result.ok is False
    assert result.error == "cancelled"


async def test_script_cancel_is_per_call(tmp_path) -> None:
    import asyncio

    from doubao_agent.sources.script import ScriptSource

    source = ScriptSource(tmp_path)
    slow_event = asyncio.Event()

    slow = asyncio.create_task(
        source.call_tool(
            ToolCall(
                name="run_python",
                arguments={"code": "import time; time.sleep(10)"},
                call_id="slow",
                cancel_event=slow_event,
            )
        )
    )
    await asyncio.sleep(0.3)

    fast = await source.call_tool(
        ToolCall(
            name="run_python",
            arguments={"code": "print(42)"},
            call_id="fast",
            cancel_event=asyncio.Event(),
        )
    )
    assert fast.ok is True
    assert "42" in fast.content

    slow_event.set()
    cancelled = await slow
    assert cancelled.ok is False
    assert cancelled.error == "cancelled"
