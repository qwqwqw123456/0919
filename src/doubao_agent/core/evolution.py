from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

# 自进化第三块：技能用量遥测 + 生命周期 + 配方蒸馏（机制取自 Hermes curator）
# 边界：只记账与改状态位，不删除任何技能文件；蒸馏只产出草案，不自动安装。

STALE_DAYS = 14
ARCHIVE_DAYS = 30
EVENTS = ("use", "view", "patch")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_usage (
    name TEXT PRIMARY KEY,
    use_count INTEGER NOT NULL DEFAULT 0,
    view_count INTEGER NOT NULL DEFAULT 0,
    patch_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    last_used_at REAL,
    state TEXT NOT NULL DEFAULT 'active',
    pinned INTEGER NOT NULL DEFAULT 0,
    archived_at REAL
);
"""

_SLUG_UNSAFE = re.compile(r"[^0-9a-zA-Z_\u4e00-\u9fff-]+")


def _slug(text: str, fallback: str = "skill") -> str:
    cleaned = _SLUG_UNSAFE.sub("-", text.strip()).strip("-").lower()
    return (cleaned or fallback)[:24]


class EvolutionStore:
    """技能级用量台账：次数、最后使用时间、生命周期状态、pin 保护。

    与配方表互补：配方记"一串工具步骤能回放"，这里记"哪个技能真的被用过、
    多久没用、该不该退役"。两者共用同一个 agent.db，但表彼此独立。
    """

    def __init__(
        self,
        db_path: str | Path,
        stale_days: int = STALE_DAYS,
        archive_days: int = ARCHIVE_DAYS,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.stale_days = max(1, int(stale_days))
        self.archive_days = max(self.stale_days, int(archive_days))
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---- 记账 -------------------------------------------------------------
    def record(self, name: str, event: str = "use", now: float | None = None) -> dict[str, Any]:
        """记一次使用/查看/修改。复用或修改已归档技能时自动复活为 active。"""
        event = event if event in EVENTS else "use"
        stamp = time.time() if now is None else now
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM skill_usage WHERE name = ?", (name,)
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO skill_usage(name, created_at, state) VALUES (?, ?, 'active')",
                    (name, stamp),
                )
                row = self._conn.execute(
                    "SELECT * FROM skill_usage WHERE name = ?", (name,)
                ).fetchone()

            revived = False
            if event == "use":
                use_count, view_count, patch_count = (
                    row["use_count"] + 1,
                    row["view_count"],
                    row["patch_count"],
                )
            elif event == "view":
                use_count, view_count, patch_count = (
                    row["use_count"],
                    row["view_count"] + 1,
                    row["patch_count"],
                )
            else:
                use_count, view_count, patch_count = (
                    row["use_count"],
                    row["view_count"],
                    row["patch_count"] + 1,
                )

            state = row["state"]
            last_used = row["last_used_at"]
            archived_at = row["archived_at"]
            # 与 Hermes 一致：真的被用了就复活；光被查看不复活，避免"看一眼就续命"。
            if event in ("use", "patch"):
                last_used = stamp
                if state == "archived" and not row["pinned"]:
                    state, archived_at, revived = "active", None, True

            self._conn.execute(
                "UPDATE skill_usage SET use_count = ?, view_count = ?, patch_count = ?, "
                "last_used_at = ?, state = ?, archived_at = ? WHERE name = ?",
                (use_count, view_count, patch_count, last_used, state, archived_at, name),
            )
            self._conn.commit()
        return {
            "name": name,
            "event": event,
            "use_count": use_count,
            "view_count": view_count,
            "patch_count": patch_count,
            "state": state,
            "revived": revived,
        }

    # ---- 读数 -------------------------------------------------------------
    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            raw = self._conn.execute("SELECT * FROM skill_usage ORDER BY name").fetchall()
        return [self._row_to_dict(row) for row in raw]

    def _row_to_dict(self, row: sqlite3.Row, now: float | None = None) -> dict[str, Any]:
        stamp = time.time() if now is None else now
        anchor = row["last_used_at"] or row["created_at"]
        return {
            "name": row["name"],
            "use_count": row["use_count"],
            "view_count": row["view_count"],
            "patch_count": row["patch_count"],
            "created_at": row["created_at"],
            "last_used_at": row["last_used_at"],
            "idle_days": int((stamp - anchor) // 86400) if anchor else None,
            "state": row["state"],
            "pinned": bool(row["pinned"]),
            "archived_at": row["archived_at"],
        }

    def status(self, top: int = 10) -> dict[str, Any]:
        rows = self.rows()
        counts = {"active": 0, "stale": 0, "archived": 0}
        for row in rows:
            counts[row["state"]] = counts.get(row["state"], 0) + 1
        used = sorted((r for r in rows if r["use_count"] > 0), key=lambda r: -r["use_count"])
        cold = sorted((r for r in rows if r["use_count"] == 0), key=lambda r: -(r["idle_days"] or 0))
        return {
            "total": len(rows),
            "state_counts": counts,
            "pinned": sum(1 for r in rows if r["pinned"]),
            "top": used[:top],
            "cold": cold[:top],
            "thresholds": {"stale_days": self.stale_days, "archive_days": self.archive_days},
        }

    # ---- 生命周期 ----------------------------------------------------------
    def _target_state(self, row: dict[str, Any]) -> str:
        if row["pinned"]:
            return row["state"]
        idle = row["idle_days"]
        if idle is None:
            return row["state"]
        if idle >= self.archive_days:
            return "archived"
        if idle >= self.stale_days and row["state"] == "active":
            return "stale"
        return row["state"]

    def audit(self, apply: bool = False, now: float | None = None) -> list[dict[str, Any]]:
        """算出应流转的技能；apply=False 只预演，不改任何状态。"""
        stamp = time.time() if now is None else now
        with self._lock:
            raw = self._conn.execute("SELECT * FROM skill_usage ORDER BY name").fetchall()
        plan: list[dict[str, Any]] = []
        for row in raw:
            item = self._row_to_dict(row, stamp)
            target = self._target_state(item)
            if target != item["state"]:
                plan.append({**item, "from": item["state"], "to": target})
        if apply and plan:
            with self._lock:
                for item in plan:
                    self._conn.execute(
                        "UPDATE skill_usage SET state = ?, archived_at = ? WHERE name = ?",
                        (item["to"], stamp if item["to"] == "archived" else None, item["name"]),
                    )
                self._conn.commit()
        return plan

    def pin(self, name: str, pinned: bool = True) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT name FROM skill_usage WHERE name = ?", (name,)
            ).fetchone()
            if row is None:
                return False
            self._conn.execute(
                "UPDATE skill_usage SET pinned = ? WHERE name = ?", (1 if pinned else 0, name)
            )
            self._conn.commit()
        return True

    # ---- 配方蒸馏 ----------------------------------------------------------
    def distill(self, recipe: dict[str, Any], skills_dir: str | Path, force: bool = False) -> Path:
        """把一条成功配方蒸馏成知识型技能草案（写进 _drafts，不自动安装）。"""
        rid = int(recipe.get("id") or 0)
        task = str(recipe.get("task") or "").strip()
        steps = recipe.get("steps") or []
        if not task or not steps:
            raise ValueError("配方缺少任务或步骤，无法蒸馏")

        name = _slug(task)
        draft_dir = Path(skills_dir) / "_drafts" / name
        target = draft_dir / "SKILL.md"
        if target.exists() and not force:
            raise FileExistsError(f"草案已存在：{target}")

        lines = [
            "---",
            f"name: {name}",
            f"description: 已验证流程（配方 #{rid}）：{task[:60]}",
            'version: "0.1"',
            "tags: [distilled, recipe]",
            "risk: L1",
            f"source: recipe#{rid}",
            "---",
            "",
            "## 适用场景",
            "",
            f"与「{task}」同型的任务。该流程由一次成功执行录制，可零模型回放。",
            "",
            "## 已验证步骤",
            "",
        ]
        for index, step in enumerate(steps, 1):
            tool = step.get("tool") or "?"
            args = step.get("arguments") or {}
            mark = "" if step.get("ok", True) else "（该步曾失败，需人工复核）"
            lines.append(f"{index}. `{tool}` 参数 `{args}`{mark}")
        lines += [
            "",
            "## 验收",
            "",
            f"- 回放 `doubao-agent replay {rid}` 无失败步；",
            "- 首次正式使用前人工过一遍参数，确认没有把一次性参数写死。",
            "",
            "## 安装",
            "",
            f"确认可用后：`doubao-agent skills install {draft_dir}`",
        ]
        draft_dir.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", "utf-8")
        return target

    def close(self) -> None:
        with self._lock:
            self._conn.close()
