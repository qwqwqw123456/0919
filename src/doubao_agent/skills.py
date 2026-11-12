from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)
MAX_KNOWLEDGE_CHARS = 1500


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff-]+", "-", name.strip()).strip("-").lower()
    return cleaned or "skill"


@dataclass
class Skill:
    name: str
    description: str
    path: Path                    # SKILL.md 路径
    version: str = "0.1"
    tags: tuple[str, ...] = ()
    runtime: str = "python"       # python | shell
    entry: str | None = None      # 相对技能目录的脚本入口；为空表示知识型技能
    input_schema: dict[str, Any] = field(default_factory=dict)
    risk: str = "L1"
    timeout: float = 60.0
    _body: str | None = field(default=None, repr=False, compare=False)

    @property
    def executable(self) -> bool:
        return bool(self.entry)

    @property
    def directory(self) -> Path:
        return self.path.parent

    @property
    def body(self) -> str:
        """渐进式加载：只有真正被阅读或命中时才读取正文。"""
        if self._body is None:
            text = self.path.read_text("utf-8")
            match = FRONTMATTER.match(text)
            self._body = (match.group(2) if match else text).strip()
        return self._body

    def resources(self) -> list[str]:
        return sorted(
            str(item.relative_to(self.directory))
            for item in self.directory.rglob("*")
            if item.is_file() and item.name != "SKILL.md"
        )


def parse_skill(path: Path) -> Skill | None:
    """解析 SKILL.md；无 frontmatter 或缺少描述的文件不算技能。"""
    try:
        text = path.read_text("utf-8")
    except OSError:
        return None
    match = FRONTMATTER.match(text)
    if match is None:
        return None
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        return None
    description = str(meta.get("description") or "").strip()
    if not description:
        return None

    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    runtime = str(meta.get("runtime") or "python").lower()
    if runtime not in ("python", "shell"):
        runtime = "python"
    schema = meta.get("input_schema")

    return Skill(
        name=_clean_name(str(meta.get("name") or path.parent.name)),
        description=description,
        path=path,
        version=str(meta.get("version") or "0.1"),
        tags=tuple(str(tag) for tag in tags),
        runtime=runtime,
        entry=str(meta["entry"]) if meta.get("entry") else None,
        input_schema=schema if isinstance(schema, dict) else {},
        risk=str(meta.get("risk") or "L1").upper(),
        timeout=float(meta.get("timeout") or 60),
    )


def _terms(text: str) -> set[str]:
    lowered = text.lower()
    words = {word for word in re.split(r"[^\w\u4e00-\u9fff]+", lowered) if len(word) >= 2}
    grams: set[str] = set()
    for word in words:
        grams.update(word[index : index + 2] for index in range(len(word) - 1))
    return words | grams


class SkillManager:
    """扫描技能目录；元数据常驻，正文与资源按需加载。"""

    def __init__(self, root: str | Path, usage: Any = None) -> None:
        self.root = Path(root)
        self._skills: dict[str, Skill] = {}
        # 用量记录器（core.evolution.EvolutionStore）；可缺省，缺省时技能库照常工作。
        self.usage = usage

    def _meter(self, name: str, event: str) -> None:
        """记账失败不能拖垮技能加载：遥测是旁路，不是主链路。"""
        if self.usage is None:
            return
        try:
            self.usage.record(name, event)
        except Exception:  # noqa: BLE001 - best-effort
            pass

    def refresh(self) -> int:
        skills: dict[str, Skill] = {}
        if self.root.is_dir():
            for path in sorted(self.root.glob("*/SKILL.md")):
                skill = parse_skill(path)
                if skill is not None:
                    skills[skill.name] = skill
        self._skills = skills
        return len(skills)

    def list(self) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda skill: skill.name)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(_clean_name(name))

    def install(self, source: str | Path) -> Skill:
        """把外部技能目录安装进技能根目录；已存在同名技能时拒绝覆盖。"""
        src = Path(source)
        if not (src / "SKILL.md").is_file():
            raise ValueError(f"不是有效技能目录（缺少 SKILL.md）：{src}")
        skill = parse_skill(src / "SKILL.md")
        if skill is None:
            raise ValueError(f"SKILL.md 解析失败：{src}")
        dest = self.root / skill.name
        if dest.exists():
            raise ValueError(f"技能已存在：{skill.name}")
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)
        self.refresh()
        return self._skills[skill.name]

    def uninstall(self, name: str) -> bool:
        """删除已安装技能；只允许删除技能根目录内的目录。"""
        skill = self.get(name)
        if skill is None:
            return False
        target = skill.directory.resolve()
        if not target.is_relative_to(self.root.resolve()) or target == self.root.resolve():
            raise ValueError(f"拒绝删除技能目录之外的路径：{target}")
        shutil.rmtree(target)
        self.refresh()
        return True

    def catalog(self) -> str:
        """只列知识型技能；可执行技能已经作为工具进入注册表。"""
        rows = [
            f"- {skill.name}：{skill.description}"
            + (f"（标签：{', '.join(skill.tags)}）" if skill.tags else "")
            for skill in self.list()
            if not skill.executable
        ]
        return "\n".join(rows)

    def match(self, query: str, limit: int = 2) -> list[Skill]:
        terms = _terms(query)
        if not terms:
            return []
        scored: list[tuple[int, Skill]] = []
        for skill in self.list():
            haystack = " ".join([skill.name, skill.description, *skill.tags]).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, skill))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [skill for _, skill in scored[:limit]]

    def knowledge_context(self, query: str, limit: int = 2) -> str:
        """把命中的知识型技能正文注入规划提示，可执行技能不注入。"""
        sections = []
        for skill in self.match(query, limit):
            if skill.executable:
                continue
            self._meter(skill.name, "view")
            sections.append(f"### {skill.name}\n{skill.body[:MAX_KNOWLEDGE_CHARS]}")
        return "\n\n".join(sections)
