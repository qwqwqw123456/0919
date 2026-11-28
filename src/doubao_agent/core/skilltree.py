"""技能树台账：把每次运行的证据回写到 29 类图谱的 A/P/B/C 状态，并按共现自动加边。

口径（与图谱页一致）：
- A 已验：本机跑出可复现的结果（有执行类工具成功 + 会话收敛）
- P 部分验：只做到分析/推理，没有本机执行证据
- B 待验：一手来源、尚未自验；运行失败也只记反馈，不升级
- C 边界：被权限/红线拦下，没有产生结论

只做证据记账，不替人下结论：等级只升不降，失败只记 `fail` 计数与 `regressed` 标志。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

LEVELS = {"C": 0, "B": 1, "P": 2, "A": 3}
LEVEL_DESC = {
    "A": "已验（本机可复现）",
    "P": "部分验（有分析无执行）",
    "B": "待验（未自验）",
    "C": "边界（被拦/无结论）",
}

EXEC_TOOLS = (
    "script.run_shell",
    "script.run_python",
    "host.sandbox_execute_bash",
    "host.sandbox_execute_code",
    "host.browser_gui_execute_action",
    "toolbox.run",
)

DEFAULT_INDEX = Path(__file__).resolve().parents[3] / "data" / "sec-tree-v55.json"
DEFAULT_ALIAS = Path(__file__).resolve().parents[3] / "data" / "sec-tree-alias.json"
MIN_SCORE = 3.0
EDGE_MIN_SCORE = 5.0
_LEX_SPLIT = re.compile(r"[^\u4e00-\u9fa5A-Za-z0-9+#./\-]+")
_EXPLICIT = re.compile(r"#\s?([0-9]{1,2})(?![0-9])")


@dataclass
class Cat:
    no: int
    name: str
    short: str
    total: int
    subs: list[dict[str, Any]] = field(default_factory=list)
    items: list[str] = field(default_factory=list)
    base_level: str = "B"

    def public(self) -> dict[str, Any]:
        return {
            "no": self.no,
            "name": self.name,
            "short": self.short,
            "total": self.total,
            "subs": [{"k": s.get("k", ""), "n": s.get("n", 0)} for s in self.subs],
            "base_level": self.base_level,
        }


class TreeLedger:
    """一份可增量写入的证据台账；所有等级都由证据重算，不手改。"""

    def __init__(
        self,
        root: str | Path,
        index_path: str | Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = Path(index_path).expanduser() if index_path else DEFAULT_INDEX
        self.alias_path = self.index_path.parent / "sec-tree-alias.json"
        self.evidence_path = self.root / "evidence.jsonl"
        self._clock = clock
        self._index: dict[str, Any] = {}
        self._cats: dict[int, Cat] = {}
        self._lex: list[tuple[int, float, str]] = []
        self._tiers: list[dict[str, Any]] = []
        self._edges: list[list[int]] = []
        self._load_index()

    # ---------- 索引 ----------
    def _load_index(self) -> None:
        if not self.index_path.is_file():
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._index = data
        ev = data.get("EV", {}) or {}
        d = data.get("D", {}) or {}
        short = data.get("SHORT", {}) or {}
        for raw_no, raw in (d.get("cats") or {}).items():
            no = int(raw_no)
            subs = list(raw.get("subs") or [])
            items: list[str] = []
            for sub in subs:
                items.extend(str(x) for x in (sub.get("items") or []))
            cat = Cat(
                no=no,
                name=str(raw.get("nm") or ""),
                short=str(short.get(raw_no, raw.get("nm") or ""))[:8],
                total=int(raw.get("n") or len(items)),
                subs=subs,
                items=items,
                base_level=str((ev.get(raw_no) or {}).get("lv") or "B"),
            )
            self._cats[no] = cat
        self._tiers = list(d.get("tiers") or [])
        self._edges = [[int(a), int(b)] for a, b in (d.get("edges") or [])]
        self._build_lexicon()

    def _build_lexicon(self) -> None:
        lex: list[tuple[int, float, str]] = []
        for cat in self._cats.values():
            for tok in _LEX_SPLIT.split(cat.name):
                tok = tok.strip()
                if len(tok) >= 2:
                    lex.append((cat.no, 3.0, tok.lower()))
            if cat.short and len(cat.short) >= 2:
                lex.append((cat.no, 2.0, cat.short.lower()))
            for sub in cat.subs:
                key = str(sub.get("k") or "").strip()
                if len(key) >= 2:
                    lex.append((cat.no, 2.0, key.lower()))
            for item in cat.items:
                item = str(item).strip()
                if len(item) >= 3 and not item.isdigit():
                    lex.append((cat.no, 1.0, item.lower()))
        alias: dict[str, list[str]] = {}
        if self.alias_path.is_file():
            try:
                alias = json.loads(self.alias_path.read_text(encoding="utf-8")) or {}
            except (OSError, json.JSONDecodeError):
                alias = {}
        for no_str, words in alias.items():
            try:
                no = int(no_str)
            except (TypeError, ValueError):
                continue
            if no not in self._cats:
                continue
            for word in words or []:
                word = str(word).strip().lower()
                if len(word) >= 2 and not word.isdigit():
                    lex.append((no, 3.0, word))
        compiled: list[tuple[int, float, str, re.Pattern[str] | None]] = []
        for no, weight, tok in lex:
            pat = None
            if tok.isascii() and re.search(r"[a-z0-9]", tok):
                pat = re.compile(r"(?<![a-z0-9_])" + re.escape(tok) + r"(?![a-z0-9_])")
            compiled.append((no, weight, tok, pat))
        self._lex = compiled

    @property
    def ready(self) -> bool:
        return bool(self._cats)

    # ---------- 分类 ----------
    def classify(self, text: str, limit: int = 3) -> list[dict[str, Any]]:
        if not self.ready or not text:
            return []
        low = text.lower()
        scores: dict[int, float] = {}
        hits: dict[int, list[str]] = {}
        for no, weight, tok, pat in self._lex:
            hit = bool(pat.search(low)) if pat else (tok in low)
            if hit:
                scores[no] = scores.get(no, 0.0) + weight
                bucket = hits.setdefault(no, [])
                if len(bucket) < 6 and tok not in bucket:
                    bucket.append(tok)
        for match in _EXPLICIT.finditer(text):
            no = int(match.group(1))
            if no in self._cats:
                scores[no] = scores.get(no, 0.0) + 6.0
                hits.setdefault(no, []).insert(0, f"#{no}")
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        out = []
        for no, score in ranked[:limit]:
            if score < MIN_SCORE:
                continue
            out.append({
                "no": no,
                "name": self._cats[no].name,
                "score": round(score, 1),
                "hits": hits.get(no, []),
            })
        return out

    # ---------- 证据 ----------
    def ingest(
        self,
        session_id: str,
        task: str,
        events: Iterable[dict[str, Any]],
        spans: Iterable[dict[str, Any]] | None = None,
        status: str = "",
    ) -> dict[str, Any] | None:
        """把一次会话折算成一条证据；分类不中就不写，避免污染图谱。"""
        events = list(events or [])
        if session_id in self._seen():
            return None
        tool_ok = [
            str(e.get("tool")) for e in events
            if e.get("event") == "tool_call" and e.get("ok")
        ]
        tool_fail = [
            str(e.get("tool")) for e in events
            if e.get("event") == "tool_call" and not e.get("ok")
        ]
        denied = [
            e for e in events
            if e.get("event") == "permission" and e.get("allowed") is False
        ]
        finish = next((e for e in events if e.get("event") == "finish"), {})
        answer = str(finish.get("answer") or "")[:600]
        route = next((e for e in events if e.get("event") == "route"), {})
        final = str(status or route.get("status") or "")
        text = "\n".join([str(task or ""), " ".join(tool_ok), answer[:300]])
        cats = self.classify(text)
        if not cats:
            return None

        exec_ok = [t for t in tool_ok if t in EXEC_TOOLS]
        exec_fail = [t for t in tool_fail if t in EXEC_TOOLS]
        if denied and not exec_ok:
            level = "C"
        elif final in ("failed", "incomplete") or (exec_fail and not exec_ok):
            level = "B"
        elif exec_ok:
            level = "A"
        else:
            level = "P"

        record = {
            "ts": round(self._clock(), 3),
            "session": session_id,
            "task": str(task or "")[:200],
            "level": level,
            "status": final or "unknown",
            "cats": cats,
            "tools_ok": sorted({t for t in tool_ok})[:12],
            "tools_fail": sorted({t for t in tool_fail})[:12],
            "denied": len(denied),
            "spans": len(list(spans or [])),
            "note": self._note(level, exec_ok, exec_fail, denied, answer),
        }
        self._seen().add(session_id)
        with self.evidence_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    @staticmethod
    def _note(level: str, ok: list[str], fail: list[str], denied: list[Any], answer: str) -> str:
        if level == "A":
            return "执行成功：" + ", ".join(ok[:3]) + ("；" + answer[:80] if answer else "")
        if level == "P":
            return "只有分析/推理，没跑执行类工具" + ("；" + answer[:80] if answer else "")
        if level == "C":
            return f"被权限拦下 {len(denied)} 次，未产生结论"
        return "运行失败/未收敛：" + (", ".join(fail[:3]) or "无执行证据")

    def _seen(self) -> set[str]:
        seen = getattr(self, "_seen_cache", None)
        if seen is None:
            seen = {str(r.get("session") or "") for r in self.evidence()}
            self._seen_cache = seen
        return seen

    def evidence(self) -> list[dict[str, Any]]:
        if not self.evidence_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.evidence_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    # ---------- 聚合 ----------
    def state(self) -> dict[str, Any]:
        rows = self.evidence()
        per: dict[int, dict[str, Any]] = {
            no: {"level": c.base_level, "base_level": c.base_level, "ok": 0, "fail": 0,
                 "upgraded": False, "regressed": False, "last_session": "", "note": "",
                 "hits": [], "tools": {}}
            for no, c in self._cats.items()
        }
        pairs: dict[tuple[int, int], dict[str, Any]] = {}
        for row in rows:
            cats = row.get("cats") or []
            level = str(row.get("level") or "B")
            for i, c in enumerate(cats):
                no = int(c.get("no"))
                slot = per.get(no)
                if slot is None:
                    continue
                for tool in row.get("tools_ok") or []:
                    slot.setdefault("tools", {})
                    slot["tools"][tool] = slot["tools"].get(tool, 0) + 1
                slot["last_session"] = row.get("session", "")
                slot["note"] = row.get("note", "")
                slot["hits"] = (c.get("hits") or [])[:5]
                if level == "A":
                    slot["ok"] += 1
                elif level in ("B", "C"):
                    slot["fail"] += 1
                if LEVELS[level] > LEVELS[slot["level"]]:
                    slot["level"] = level
                    slot["upgraded"] = True
                if level in ("B", "C") and LEVELS[slot["level"]] >= LEVELS["P"]:
                    slot["regressed"] = True
            strong = [int(c.get("no")) for c in cats if float(c.get("score") or 0) >= EDGE_MIN_SCORE]
            for i in range(len(strong)):
                for j in range(i + 1, len(strong)):
                    a, b = strong[i], strong[j]
                    key = (min(a, b), max(a, b))
                    info = pairs.setdefault(key, {"from": key[0], "to": key[1], "count": 0, "sessions": []})
                    info["count"] += 1
                    if len(info["sessions"]) < 3:
                        info["sessions"].append(row.get("session", ""))

        curated = {(min(a, b), max(a, b)) for a, b in self._edges}
        auto_edges = [
            {"from": v["from"], "to": v["to"], "count": v["count"], "sessions": v["sessions"],
             "reason": "同一次任务同时命中，自动加边"}
            for k, v in sorted(pairs.items(), key=lambda kv: -kv[1]["count"])
            if k not in curated
        ]

        cats_out = {}
        for no, slot in per.items():
            cat = self._cats[no]
            item = cat.public()
            item.update(slot)
            cats_out[str(no)] = item
        upgraded = sum(1 for v in per.values() if v["upgraded"])
        regressed = sum(1 for v in per.values() if v["regressed"])
        return {
            "version": "v55",
            "updated_at": round(self._clock(), 3),
            "tiers": self._tiers,
            "cats": cats_out,
            "edges": self._edges,
            "auto_edges": auto_edges,
            "evidence_total": len(rows),
            "upgraded": upgraded,
            "regressed": regressed,
            "levels": {k: sum(1 for v in per.values() if v["level"] == k) for k in LEVELS},
        }

    def export(self, path: str | Path) -> Path:
        """导出成图谱页可吃的 EV / edges 结构（不覆盖原 meta，单独一份）。"""
        state = self.state()
        ev = {}
        for no_str, cat in state["cats"].items():
            ev[no_str] = {
                "lv": cat["level"],
                "note": cat["note"] or f"自动回写：证据 {cat['ok']} 成功 / {cat['fail']} 失败",
            }
        edges = list(self._edges) + [[e["from"], e["to"]] for e in state["auto_edges"]]
        payload = dict(self._index)          # 先继承原页全部字段：D/SHORT/ST/KN/SX/CX/RV
        payload.update({
            "meta": {"v": "v55+evidence", "t": time.strftime("%Y-%m-%d %H:%M", time.localtime())},
            "D": dict(self._index.get("D", {}), edges=edges),
            "EV": ev,
            "auto_edges": state["auto_edges"],
            "edges": edges,
        })
        out = Path(path).expanduser()
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return out


def ledger_from_config(workspace: str | Path) -> TreeLedger:
    return TreeLedger(Path(workspace) / "tree")
