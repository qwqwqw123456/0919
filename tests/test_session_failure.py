from __future__ import annotations

import pytest

from doubao_agent.config import Config
from doubao_agent.runtime import Runtime


class BoomProvider:
    async def complete(self, *args, **kwargs):
        raise RuntimeError("provider 炸了")

    async def aclose(self) -> None:
        return None


def _cfg(tmp_path) -> Config:
    return Config(
        workspace=tmp_path,
        runs_dir=tmp_path / "runs",
        db_path=tmp_path / "agent.db",
        skills_dir=tmp_path / "skills",
        scheduler_enabled=False,
        planning_enabled=False,
        model={"provider": "mock"},
    )


async def test_exception_marks_session_failed(tmp_path) -> None:
    runtime = await Runtime.build(_cfg(tmp_path))
    try:
        runtime.agent.provider = BoomProvider()
        runtime.provider = BoomProvider()
        with pytest.raises(RuntimeError):
            await runtime.start_run("跑个任务")
        rows = runtime.sessions.list()
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert runtime.running_sessions() == []
        assert "failed" in (rows[0]["result"] or "")
    finally:
        await runtime.aclose()


async def test_normal_run_still_completed(tmp_path) -> None:
    runtime = await Runtime.build(_cfg(tmp_path))
    try:
        result = await runtime.start_run("随便聊聊")
        assert result["status"] == "completed"
        assert runtime.sessions.list()[0]["status"] == "completed"
    finally:
        await runtime.aclose()
