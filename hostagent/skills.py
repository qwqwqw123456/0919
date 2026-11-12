"""主权 Skill 客户端（manifest 第 D 层）。

唯一业务入口 `invoke` 扣额度；额度用尽走 `quota-apply`，不绕过。
本地不伪造结果：二进制缺失或调用失败都如实报错。
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

from .contract import SOVEREIGN_SKILLS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quotas (
    skill TEXT PRIMARY KEY,
    daily_limit INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS usage (
    skill TEXT NOT NULL,
    day TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (skill, day)
);
"""


class SkillError(RuntimeError):
    pass


class QuotaStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @staticmethod
    def _day() -> str:
        return date.today().isoformat()

    def set_limit(self, skill: str, daily_limit: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO quotas(skill, daily_limit) VALUES (?, ?) "
                "ON CONFLICT(skill) DO UPDATE SET daily_limit = excluded.daily_limit",
                (skill, daily_limit),
            )
            self._conn.commit()

    def used(self, skill: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT used FROM usage WHERE skill = ? AND day = ?", (skill, self._day())
            ).fetchone()
        return int(row["used"]) if row else 0

    def limit(self, skill: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT daily_limit FROM quotas WHERE skill = ?", (skill,)
            ).fetchone()
        return int(row["daily_limit"]) if row else 0

    def remaining(self, skill: str) -> int:
        limit = self.limit(skill)
        if limit <= 0:
            return -1  # 未设限
        return max(0, limit - self.used(skill))

    def incr(self, skill: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage(skill, day, used) VALUES (?, ?, 1) "
                "ON CONFLICT(skill, day) DO UPDATE SET used = used + 1",
                (skill, self._day()),
            )
            self._conn.commit()

    def all(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for skill in SOVEREIGN_SKILLS:
            rows.append({
                "skill": skill,
                "limit": self.limit(skill),
                "used": self.used(skill),
                "remaining": self.remaining(skill),
            })
        return rows

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class SovereignSkillClient:
    def __init__(self, *, binary: str = "sovereign-skill", quota: QuotaStore | None = None) -> None:
        self.binary = binary
        self.quota = quota

    @property
    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def invoke(self, skill: str, task: str, attachment: str | None = None) -> dict[str, Any]:
        if skill not in SOVEREIGN_SKILLS:
            raise SkillError(f"无此能力授权: {skill}（不在 13 个已授权 Skill 内）")
        if self.quota is not None:
            remaining = self.quota.remaining(skill)
            if remaining == 0:
                raise SkillError(f"{skill} 今日额度已用尽，需先 quota-apply")
        args = [self.binary, "invoke", "--skill", skill, "--task", task]
        if attachment:
            args.extend(["--attachment", attachment])
        stdout = self._run(args, action="invoke")
        if self.quota is not None:
            self.quota.incr(skill)
        return self._parse(stdout)

    def quota_apply(self, skill: str, reason: str) -> dict[str, Any]:
        stdout = self._run(
            [self.binary, "quota-apply", "--skill", skill, "--reason", reason], action="quota-apply"
        )
        return self._parse(stdout)

    def _run(self, args: list[str], action: str) -> str:
        if not self.available:
            raise SkillError(f"{self.binary} 未安装：{action} 需在云端 TEE 运行，本地不伪造结果")
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=180, check=False)
        except subprocess.TimeoutExpired as exc:
            raise SkillError(f"{self.binary} {action} 超时") from exc
        if proc.returncode != 0:
            raise SkillError(f"{self.binary} {action} 失败({proc.returncode}): "
                             f"{(proc.stderr or proc.stdout)[:300]}")
        return proc.stdout

    @staticmethod
    def _parse(stdout: str) -> dict[str, Any]:
        stdout = stdout.strip()
        if not stdout:
            return {}
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return {"raw": stdout[:1000]}
        return data if isinstance(data, dict) else {"result": data}


def route_task(task: str, mapping_path: str | Path = "assets/capability-mapping.yaml") -> str | None:
    """按 capability-mapping.yaml 的 routing_keywords 路由；未命中返回 None（诚实无授权）。"""
    path = Path(mapping_path)
    if not path.exists():
        return None
    try:
        import yaml
    except ImportError:
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("skills") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        keywords = entry.get("routing_keywords") or []
        name = entry.get("skill") or entry.get("name")
        if name in SOVEREIGN_SKILLS and any(str(k) in task for k in keywords):
            return str(name)
    return None


def _now() -> float:
    return time.time()
