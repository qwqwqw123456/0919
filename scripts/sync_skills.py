"""把散落各处的技能统一梳理并装进 Agent 技能目录（默认符号链接，不复制大文件）。

用法：
    python scripts/sync_skills.py               # 预览：只打印会装什么
    python scripts/sync_skills.py --apply       # 实际建立/更新链接
    python scripts/sync_skills.py --apply --repair   # 额外把坏链接/空链接重建

规则：
- 按 ROOTS 顺序扫描，同名技能只保留**第一个**来源（宿主平台版优先，其次是个人技能、
  主权技能、hermes、仓库示例），避免同名覆盖导致"到底用的是哪一份"说不清；
- 每个技能一条绝对路径符号链接，宿主 79MB 只读目录不会被复制；
- 链接名用 frontmatter 的 name（没有则用目录名），清洗成合法技能名；
- 已存在且指向正确目标的链接跳过；存在但不是链接的目录（手写技能）默认不动。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doubao_agent.skills import parse_skill  # noqa: E402

ROOTS: list[tuple[str, Path]] = [
    ("host", Path("/runtime/skills")),
    ("user", Path.home() / ".doubao/agent_mode/workspace/.user_skills"),
    ("sovereign", Path.home() / ".config/sovereign-skill/skills"),
    ("hermes", Path.home() / ".hermes/skills"),
    ("repo", Path(__file__).resolve().parents[1] / "skills"),
]

SKILLS_DIR = Path(os.environ.get("AGENT_SKILLS_DIR") or (Path.home() / "agent-state/skills"))


def discover() -> tuple[list[dict], list[dict], list[dict]]:
    """返回 (待装项, 同名被跳过项, 解析失败项)。"""
    found: dict[str, dict] = {}
    skipped: list[dict] = []
    broken: list[dict] = []
    for root_name, root in ROOTS:
        if not root.exists():
            continue
        for md in sorted(root.rglob("SKILL.md")):
            skill = parse_skill(md)
            if skill is None:
                broken.append({"root": root_name, "path": str(md)})
                continue
            item = {
                "name": skill.name,
                "root": root_name,
                "dir": str(md.parent),
                "executable": skill.executable,
                "entry": skill.entry,
                "description": skill.description[:120],
            }
            if skill.name in found:
                skipped.append({"name": skill.name, "root": root_name, "kept": found[skill.name]["root"]})
                continue
            found[skill.name] = item
    return list(found.values()), skipped, broken


def apply(items: list[dict], repair: bool = False) -> dict:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    created = updated = kept = conflicts = 0
    for item in items:
        dest = SKILLS_DIR / item["name"]
        src = Path(item["dir"])
        if dest.is_symlink():
            if dest.resolve() == src.resolve():
                kept += 1
                continue
            if not repair:
                conflicts += 1
                continue
            dest.unlink()
            dest.symlink_to(src)
            updated += 1
            continue
        if dest.exists():
            # 已有真实目录（手写技能或早期复制版）：不覆盖，算冲突留给人看
            conflicts += 1
            continue
        dest.symlink_to(src)
        created += 1
    return {"created": created, "updated": updated, "kept": kept, "conflicts": conflicts}


def main() -> int:
    parser = argparse.ArgumentParser(description="技能全量梳理与装配")
    parser.add_argument("--apply", action="store_true", help="实际建立链接（默认只预览）")
    parser.add_argument("--repair", action="store_true", help="重建指向不对的链接")
    parser.add_argument("--manifest", default=None, help="把清单写到指定 JSON 路径")
    args = parser.parse_args()

    items, skipped, broken = discover()
    print(f"发现可装技能 {len(items)} 个｜同名跳过 {len(skipped)}｜解析失败 {len(broken)}")
    for name, count in _by_root(items).items():
        print(f"  {name}: {count}")
    print(f"可执行技能 {sum(1 for i in items if i['executable'])} 个（会出现在工具列表里）")

    result = {"discovered": len(items), "skipped": len(skipped), "broken": len(broken)}
    if args.apply:
        result.update(apply(items, repair=args.repair))
        print(f"装配结果：新建 {result['created']}｜重建 {result['updated']}｜已存在 {result['kept']}｜冲突 {result['conflicts']}")

    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(
            json.dumps({"items": items, "skipped": skipped, "broken": broken, "result": result}, ensure_ascii=False, indent=2),
            "utf-8",
        )
        print(f"清单已写：{args.manifest}")
    return 0


def _by_root(items: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item["root"]] = out.get(item["root"], 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
