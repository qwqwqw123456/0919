"""把当前台账打成自包含 HTML（数据内联），用于发布到 CDN 查看。

用法：PYTHONPATH=src python3 scripts/publish_tree.py --out /tmp/tree-published.html
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doubao_agent.core.skilltree import TreeLedger  # noqa: E402

PAGE = ROOT / "src" / "doubao_agent" / "server" / "static" / "tree.html"


def build(out: Path) -> Path:
    ledger = TreeLedger(Path.home() / "agent-state" / "tree")
    with tempfile.TemporaryDirectory() as tmp:
        data = ledger.export(Path(tmp) / "export.json").read_text(encoding="utf-8")
    payload = json.dumps(json.loads(data), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = PAGE.read_text(encoding="utf-8")
    block = f"window.__TREE_DATA__={payload};\n"
    marker = "var LOCAL_MODE=false"
    if marker in html:
        html = html.replace(marker, block + marker, 1)
    else:
        raise SystemExit("找不到主脚本插入点，页面结构变了")
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/tree-published.html")
    args = ap.parse_args()
    path = build(Path(args.out))
    print(f"已生成：{path}（{path.stat().st_size / 1024:.0f} KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
