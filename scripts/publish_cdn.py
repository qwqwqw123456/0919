"""生成可上传的独立页：SHADOW 指向指定 meta，并把壳版本写成 s15x。

用法：
  PYTHONPATH=src python3 scripts/publish_cdn.py --meta https://aka.doubaocdn.com/s/xxx --out /tmp/tree-shell.html
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "src" / "doubao_agent" / "server" / "static" / "tree.html"
SHELL_V = "s15x"


def build(meta_url: str, out: Path, note_url: str | None = None) -> Path:
    html = PAGE.read_text(encoding="utf-8")
    m = re.search(r"var SHADOW=\{[^}]*\};", html)
    if not m:
        raise SystemExit("SHADOW 段没找到")
    old = m.group(0)
    note = note_url or (re.search(r"note:'([^']+)'", old).group(1) if "note:'" in old else "")
    stamp = time.strftime("%Y-%m-%d %H:%M")
    new = ("var SHADOW={v:'v55+evidence',sv:'%s',t:'%s',meta:'%s',note:'%s'};"
           % (SHELL_V, stamp, meta_url, note))
    html = html.replace(old, new, 1)
    # 独立页没有本地 /api，下载按钮改回原始 CDN 直链，避免点了 404
    html = html.replace('href="/api/tree/download?which=words"',
                        'href="https://aka.doubaocdn.com/s/8q2GxAJw6B"', 1)
    html = html.replace('href="/api/tree/download?which=agent"',
                        'href="https://aka.doubaocdn.com/s/4ubRVqvWeh"', 1)
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="最新导出 JSON 的 CDN 地址")
    ap.add_argument("--note", default="", help="note 地址（默认沿用原值）")
    ap.add_argument("--out", default="/tmp/tree-shell.html")
    args = ap.parse_args()
    path = build(args.meta, Path(args.out), args.note or None)
    print(f"已生成 {path}（{path.stat().st_size/1024:.0f} KB，shellV={SHELL_V}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
