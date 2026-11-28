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
EDGE_MIN_COUNT = 2       # 同一对类别至少 2 次共现才连线，避免单次任务刷图
EDGE_MIN_SESSIONS = 2    # 且必须跨 2 个会话
TOOL_MIN_COUNT = 3       # 工具模板 → 类别：至少 3 次
TOOL_MIN_SESSIONS = 2
DEFAULT_TIMEOUT_S = 120  # 工具调用默认超时上限（运行时口径）

# 自然语言策略里的语义短语 → 具体工具/命令标记（只为“过滤”，不做执行）
DESTRUCTIVE_MARKS = {
    "sqlmap", "msfconsole", "msfvenom", "metasploit", "hydra", "medusa", "nikto",
    "wfuzz", "ffuf", "commix", "cobaltstrike", "empire", "beef", "ettercap",
    "bettercap", "responder", "hashcat", "john", "aircrack-ng", "reaver", "pacu",
    "routerSploit", "drozer", "yersinia", "slowloris", "hping3",
}
FILELESS_MARKS = {"powershell", "mshta", "rundll32", "regsvr32", "certutil", "wmic", "msiexec"}


def host_usage() -> dict[str, Any]:
    """本机资源占用快照（/proc 读数，不引第三方依赖）。"""
    import os

    out: dict[str, Any] = {"cpu_count": os.cpu_count() or 1}
    try:
        load = os.getloadavg()
        out["load1"], out["load5"], out["load15"] = (round(x, 2) for x in load)
        out["cpu_pct"] = round(min(100.0, load[0] / out["cpu_count"] * 100), 1)
    except (OSError, AttributeError):
        out["cpu_pct"] = 0
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            info[key.strip()] = int(rest.strip().split()[0]) / 1024  # MiB
        out["mem_total_mb"] = round(info.get("MemTotal", 0))
        out["mem_used_mb"] = round(info.get("MemTotal", 0) - info.get("MemAvailable", 0))
        out["mem_pct"] = round(out["mem_used_mb"] / out["mem_total_mb"] * 100, 1) if out["mem_total_mb"] else 0
    except (OSError, ValueError, IndexError):
        out["mem_pct"] = 0
    try:
        rss = 0
        for line in Path(f"/proc/{os.getpid()}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024
                break
        out["agent_rss_mb"] = round(rss)
    except (OSError, ValueError):
        out["agent_rss_mb"] = 0
    return out


def parse_strategy(text: str) -> dict[str, Any]:
    """把一句自然语言策略拆成 deny / prefer 标记；拆不出就保留原文并给出提示。"""
    text = str(text or "").strip()
    deny: list[dict[str, str]] = []
    prefer: list[dict[str, str]] = []

    def add(bucket: list[dict[str, str]], pattern: str, label: str) -> None:
        pattern = pattern.strip()
        if not pattern or any(d["pattern"] == pattern for d in bucket):
            return
        bucket.append({"pattern": pattern, "label": label})

    for clause in re.split(r"[，,；;。\n]+", text):
        clause = clause.strip()
        if not clause:
            continue
        lowered = clause.lower()
        deny_mode = any(k in clause for k in ("禁止", "禁用", "不要", "不得", "不允许", "避免")) or "avoid" in lowered
        prefer_mode = any(k in clause for k in ("优先", "首选", "尽量用", "先用")) or "prefer" in lowered
        if not deny_mode and not prefer_mode:
            continue
        body = re.sub(r"(禁止使用|禁止|禁用|不要使用|不要|不得使用|不得|不允许|避免|优先走|优先使用|优先|首选|尽量用|先用)", "", clause)
        body = re.sub(r"(任何|所有|一切|一律)", "", body).strip(" 的，,")
        if not body:
            continue
        bucket = deny if deny_mode else prefer
        # 先认语义短语：这类说法里往往夹着英文（payload/fileless），按字面抽英文会抽偏
        if any(k in body for k in ("破坏", "payload", "exploit", "利用载荷")):
            for tok in sorted(DESTRUCTIVE_MARKS):
                add(bucket, tok, "破坏性 payload")
            continue
        if "无文件" in body or "fileless" in lowered:
            for tok in sorted(FILELESS_MARKS):
                add(bucket, tok, "无文件攻击")
            continue
        if "暴力" in body or "爆破" in body:
            for tok in ("hydra", "medusa", "hashcat", "john"):
                add(bucket, tok, body)
            continue
        if "反弹" in body and "shell" in lowered:
            for tok in ("nc", "ncat", "socat", "bash -i", "python -c"):
                add(bucket, tok, body)
            continue
        explicit = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_.\-]{1,}", body)]
        if explicit:
            for tok in explicit:
                add(bucket, tok, body)
            continue
        # 没映射到具体工具：原文保留，页面上提示“未解析成可过滤标记”
        add(bucket, body, body)
    return {
        "text": text,
        "deny": deny,
        "prefer": prefer,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

GENERIC_BINS = {
    "bash", "sh", "zsh", "dash", "python", "python3", "curl", "wget", "git",
    "pip", "pip3", "npm", "npx", "ls", "cat", "head", "tail", "grep", "egrep",
    "awk", "sed", "chmod", "chown", "cp", "mv", "rm", "mkdir", "touch", "tar",
    "zip", "unzip", "ssh", "scp", "rsync", "echo", "printf", "cd", "export",
    "sudo", "env", "which", "find", "xargs", "tee", "test", "sleep", "true",
    "sort", "stat", "wc", "free", "pwd", "uname", "whoami", "hostname", "date",
    "timeout", "set", "unset", "if", "then", "fi", "for", "while", "do", "done",
    "source", "dirname", "basename", "realpath", "readlink", "df", "du", "ps",
    "kill", "pgrep", "nohup", "wait", "seq", "tr", "cut", "paste", "diff", "file",
    "else", "then", "elif", "esac", "case", "break", "continue", "return", "local",
    "null", "dev", "proc", "sys", "tmp", "home", "usr", "bin", "etc",
}
_CMD_SPLIT = re.compile(r"&&|\|\||;|\|")
_LEX_SPLIT = re.compile(r"[^\u4e00-\u9fa5A-Za-z0-9+#./\-]+")
_EXPLICIT = re.compile(r"#\s?([0-9]{1,2})(?![0-9])")


def _arguments(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("arguments")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        import ast

        val = ast.literal_eval(raw)
        if isinstance(val, dict):
            return val
    except (SyntaxError, ValueError):
        pass
    # 审计里存的是 repr 字符串；代码带换行时 literal_eval 会失败，退回按引号手工抽键值
    def take_literal(text: str, pos: int) -> str | None:
        quote = text[pos]
        i, buf = pos + 1, []
        while i < len(text):
            ch = text[i]
            if ch == "\\":
                buf.append(text[i:i + 2])
                i += 2
                continue
            if ch == quote:
                return "".join(buf)
            buf.append(ch)
            i += 1
        return None

    out: dict[str, Any] = {}
    for key in ("command", "code", "cmd", "action", "verb", "tool", "name"):
        m = re.search("'" + key + "':\\s*", raw)
        if not m:
            continue
        pos = m.end()
        if pos >= len(raw) or raw[pos] not in "\"'":
            continue
        lit = take_literal(raw, pos)
        if lit is not None:
            out[key] = lit.replace("\\n", "\n").replace("\\t", "\t").replace("\\'", "'").replace('\\"', '"')
    return out


def command_template(tool: str, event: dict[str, Any]) -> str | None:
    """把一次工具调用归一成 ``bin[:sub]`` 模板；通用底座命令不作为端点。"""
    args = _arguments(event)
    tool = str(tool)
    if tool in ("script.run_shell", "host.sandbox_execute_bash"):
        cmd = str(args.get("command") or args.get("cmd") or "").strip()
        if not cmd:
            return None
        for seg in _CMD_SPLIT.split(cmd):
            toks = [t for t in seg.strip().split() if not t.startswith("-") and "=" not in t]
            if not toks:
                continue
            binary = Path(toks[0]).name
            if (not re.match(r"^[a-z][a-z0-9_.+-]+$", binary)) or binary in GENERIC_BINS:
                continue
            if re.search(r"\.(txt|json|md|log|bin|py|sh|ya?ml|csv|out|data|db|gz|zip)$", binary):
                continue
            sub = toks[1] if len(toks) >= 3 and re.match(r"^[a-z][a-z0-9_.+-]*$", toks[1]) else ""
            return f"{binary}:{sub}" if sub else binary
        return None
    if tool in ("script.run_python", "host.sandbox_execute_code"):
        code = str(args.get("code") or "")
        for line in code.splitlines():
            m = re.match(r"\s*(?:import|from)\s+([a-zA-Z0-9_]+)", line)
            if m:
                mod = m.group(1)
                import sys

                if mod not in sys.stdlib_module_names:
                    return f"python:{mod}"
                return None
        return None
    if tool == "host.browser_gui_execute_action":
        verb = str(args.get("action") or args.get("verb") or "act")[:20]
        return f"browser:{verb}"
    if tool == "toolbox.run":
        name = str(args.get("tool") or args.get("name") or "")[:30]
        return f"toolbox:{name}" if name else None
    return None


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

        templates: list[str] = []
        steps: list[dict[str, Any]] = []
        for e in events:
            if e.get("event") == "permission" and e.get("allowed") is False:
                steps.append({"t": str(e.get("tool") or ""), "ok": False, "ms": 0,
                              "tpl": None, "err": str(e.get("reason") or "")[:120], "denied": True})
                continue
            if e.get("event") != "tool_call":
                continue
            tool = str(e.get("tool") or "")
            tpl = command_template(tool, e)
            steps.append({"t": tool, "ok": bool(e.get("ok")),
                          "ms": int(e.get("elapsed_ms") or 0), "tpl": tpl,
                          "err": str(e.get("error") or "")[:120], "denied": False})
            if e.get("ok") and tool in EXEC_TOOLS and tpl and tpl not in templates:
                templates.append(tpl)
        record = {
            "ts": round(self._clock(), 3),
            "session": session_id,
            "task": str(task or "")[:200],
            "level": level,
            "status": final or "unknown",
            "cats": cats,
            "tools_ok": sorted({t for t in tool_ok})[:12],
            "tools_fail": sorted({t for t in tool_fail})[:12],
            "templates": templates[:8],
            "steps": steps[:24],
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
                 "hits": [], "tools": {}, "tools_fail": {}, "templates": {}}
            for no, c in self._cats.items()
        }
        pairs: dict[tuple[int, int], dict[str, Any]] = {}
        tools: dict[tuple[str, int], dict[str, Any]] = {}
        tool_rows: dict[tuple[int, str], dict[str, Any]] = {}
        global_tools: dict[str, dict[str, Any]] = {}
        for row in rows:
            cats = row.get("cats") or []
            level = str(row.get("level") or "B")
            steps = [s for s in (row.get("steps") or []) if isinstance(s, dict)]
            for st in steps:
                t = str(st.get("t") or "")
                if not t:
                    continue
                key = (0, t)
                gs = global_tools.setdefault(t, {"tool": t, "calls": 0, "ok": 0, "fail": 0,
                                                 "ms": [], "last_ts": 0.0, "error": "",
                                                 "tpl": "", "cats": []})
                gs["calls"] += 1
                gs["ok" if st.get("ok") else "fail"] += 1
                if st.get("ms"):
                    gs["ms"].append(int(st["ms"]))
                gs["last_ts"] = max(gs["last_ts"], float(row.get("ts") or 0))
                if st.get("err") and not gs["error"]:
                    gs["error"] = str(st["err"])
                if st.get("tpl") and not gs["tpl"]:
                    gs["tpl"] = str(st["tpl"])
            for i, c in enumerate(cats):
                no = int(c.get("no"))
                slot = per.get(no)
                if slot is None:
                    continue
                for st in steps:
                    t = str(st.get("t") or "")
                    if not t:
                        continue
                    tr = tool_rows.setdefault(
                        (no, t),
                        {"tool": t, "calls": 0, "ok": 0, "fail": 0, "ms": [], "last_ts": 0.0,
                         "error": "", "tpl": "", "denied": 0},
                    )
                    tr["calls"] += 1
                    tr["ok" if st.get("ok") else "fail"] += 1
                    if st.get("denied"):
                        tr["denied"] += 1
                    if st.get("ms"):
                        tr["ms"].append(int(st["ms"]))
                    tr["last_ts"] = max(tr["last_ts"], float(row.get("ts") or 0))
                    if st.get("err") and not tr["error"]:
                        tr["error"] = str(st["err"])
                    if st.get("tpl") and not tr["tpl"]:
                        tr["tpl"] = str(st["tpl"])
                for t_ in {str(s.get("t") or "") for s in steps}:
                    gs2 = global_tools.get(t_)
                    if gs2 is not None and no not in gs2["cats"] and len(gs2["cats"]) < 8:
                        gs2["cats"].append(no)
                for tool in row.get("tools_ok") or []:
                    slot.setdefault("tools", {})
                    slot["tools"][tool] = slot["tools"].get(tool, 0) + 1
                for tool in row.get("tools_fail") or []:
                    slot.setdefault("tools_fail", {})
                    slot["tools_fail"][tool] = slot["tools_fail"].get(tool, 0) + 1
                for tpl in row.get("templates") or []:
                    slot.setdefault("templates", {})
                    slot["templates"][tpl] = slot["templates"].get(tpl, 0) + 1
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
                    info = pairs.setdefault(key, {"from": key[0], "to": key[1], "count": 0,
                                                 "sessions": [], "examples": []})
                    info["count"] += 1
                    sid = row.get("session", "")
                    if sid and sid not in info["sessions"] and len(info["sessions"]) < 5:
                        info["sessions"].append(sid)
                    if len(info["examples"]) < 2:
                        ca = next((x for x in cats if int(x.get("no")) == a), {})
                        cb = next((x for x in cats if int(x.get("no")) == b), {})
                        info["examples"].append({
                            "session": sid,
                            "a": [str(h) for h in (ca.get("hits") or [])[:3]],
                            "b": [str(h) for h in (cb.get("hits") or [])[:3]],
                        })
            if level == "A":
                for tpl in row.get("templates") or []:
                    for c in cats:
                        no2 = int(c.get("no"))
                        if float(c.get("score") or 0) < EDGE_MIN_SCORE:
                            continue
                        key2 = (tpl, no2)
                        slot2 = tools.setdefault(key2, {"tool": tpl, "cat": no2, "count": 0,
                                                        "sessions": [], "hits": []})
                        slot2["count"] += 1
                        sid2 = row.get("session", "")
                        if sid2 and sid2 not in slot2["sessions"] and len(slot2["sessions"]) < 5:
                            slot2["sessions"].append(sid2)
                        if not slot2["hits"]:
                            slot2["hits"] = [str(h) for h in (c.get("hits") or [])[:3]]

        curated = {(min(a, b), max(a, b)) for a, b in self._edges}
        auto_edges = []
        for k, v in sorted(pairs.items(), key=lambda kv: -kv[1]["count"]):
            if k in curated:
                continue
            if v["count"] < EDGE_MIN_COUNT or len(v["sessions"]) < EDGE_MIN_SESSIONS:
                continue
            names = [self._cats.get(v["from"]), self._cats.get(v["to"])]
            label = "、".join(f"#{c.no}{c.short}" for c in names if c)
            ex = (v.get("examples") or [{}])[0]
            tok = " / ".join(x for x in (ex.get("a") + ex.get("b")) if x) or "—"
            auto_edges.append({
                "from": v["from"], "to": v["to"], "count": v["count"], "sessions": v["sessions"],
                "kind": "cooccur",
                "hits": [ex.get("a") or [], ex.get("b") or []],
                "reason": f"{label} 在 {v['count']} 次任务里同时命中（{len(v['sessions'])} 个会话），命中词：{tok}",
                "explain": f"证据会话 {'/'.join(v['sessions'][:2])}",
            })
        tool_edges = [
            {"tool": v["tool"], "cat": v["cat"], "count": v["count"], "sessions": v["sessions"],
             "kind": "tool", "hits": v["hits"],
             "reason": f"模板 {v['tool']} 在 {v['count']} 次 A 级任务中服务 #{v['cat']}",
             "passed": True}
            for v in tools.values()
            if v["count"] >= TOOL_MIN_COUNT and len(v["sessions"]) >= TOOL_MIN_SESSIONS
        ]

        cats_out = {}
        strat = self.strategy()
        deny = [str(x.get("pattern") or "") for x in strat.get("deny") or [] if x.get("pattern")]
        traces: list[dict[str, Any]] = []
        for no, slot in per.items():
            cat = self._cats[no]
            item = cat.public()
            item.update(slot)
            rows_t = [v for (n2, _t), v in tool_rows.items() if n2 == no]
            rows_t.sort(key=lambda x: (-x["calls"], x["tool"]))
            item["tools_detail"] = [self._tool_out(v, deny) for v in rows_t[:12]]
            mine = [r for r in rows if no in [int(c.get("no")) for c in (r.get("cats") or [])]]
            # 「失败」只算真失败（运行失败/B/C 级，或工具报错）；仅被权限拦下不算
            fails = [
                r for r in mine
                if str(r.get("level")) in ("B", "C") or any(
                    isinstance(s, dict) and not s.get("ok") and not s.get("denied")
                    for s in (r.get("steps") or []))
            ]
            denied_rows = [r for r in mine if any(
                isinstance(s, dict) and s.get("denied") for s in (r.get("steps") or []))]
            item["denied"] = len(denied_rows)
            if fails:
                last = fails[-1]
                item["fails"] = len(fails)
                item["fail_last"] = self._fail_trace(last, no)
                traces.extend(self._fail_trace(r, no) for r in fails)
            else:
                item["fails"] = 0
                item["fail_last"] = None
            cats_out[str(no)] = item
        traces.sort(key=lambda x: -float(x.get("ts") or 0))
        upgraded = sum(1 for v in per.values() if v["upgraded"])
        regressed = sum(1 for v in per.values() if v["regressed"])
        return {
            "version": "v55",
            "updated_at": round(self._clock(), 3),
            "tiers": self._tiers,
            "cats": cats_out,
            "edges": self._edges,
            "auto_edges": auto_edges,
            "tool_edges": sorted(tool_edges, key=lambda x: -x["count"]),
            "evidence_total": len(rows),
            "upgraded": upgraded,
            "regressed": regressed,
            "levels": {k: sum(1 for v in per.values() if v["level"] == k) for k in LEVELS},
            "strategy": strat,
            "host": host_usage(),
            "fails": traces[:16],
            "tool_totals": sorted(
                (self._tool_out(v, deny, with_cats=True) for v in global_tools.values()),
                key=lambda x: (-x["calls"], x["tool"]),
            )[:40],
        }

    def _fail_trace(self, row: dict[str, Any], no: int) -> dict[str, Any]:
        steps = [s for s in (row.get("steps") or []) if isinstance(s, dict)]
        first_fail = None
        try:
            first_fail = next(i for i, s in enumerate(steps)
                              if not s.get("ok") and not s.get("denied"))
        except StopIteration:
            first_fail = None
        return {
            "ts": float(row.get("ts") or 0),
            "at": time.strftime("%m-%d %H:%M", time.localtime(float(row.get("ts") or 0))),
            "session": str(row.get("session") or ""),
            "task": str(row.get("task") or "")[:120],
            "no": no,
            "level": str(row.get("level") or "B"),
            "note": str(row.get("note") or "")[:160],
            "tools_fail": list(row.get("tools_fail") or [])[:6],
            "steps": steps[:16],
            "first_fail": first_fail,
            "denied": sum(1 for s in steps if s.get("denied")),
            "before": [str(s.get("t") or "") for s in steps[:first_fail or 0]][-4:],
        }

    @staticmethod
    def _tool_out(tr: dict[str, Any], deny: list[str], with_cats: bool = False) -> dict[str, Any]:
        ms = sorted(int(x) for x in tr.get("ms") or [])
        calls = int(tr.get("calls") or 0) or 1
        avg = round(sum(ms) / len(ms), 1) if ms else 0
        p95 = ms[min(len(ms) - 1, int(len(ms) * 0.95))] if ms else 0
        name = str(tr.get("tool") or "")
        tpl = str(tr.get("tpl") or "")
        blocked = next((d for d in deny if d and (d.lower() in name.lower() or d.lower() in tpl.lower())), "")
        out = {
            "tool": name,
            "calls": int(tr.get("calls") or 0),
            "ok": int(tr.get("ok") or 0),
            "fail": int(tr.get("fail") or 0),
            "rate": round(int(tr.get("ok") or 0) / calls * 100),
            "avg_ms": avg,
            "p95_ms": p95,
            "timeout_s": DEFAULT_TIMEOUT_S,
            "last_ts": round(float(tr.get("last_ts") or 0), 3),
            "error": str(tr.get("error") or "")[:120],
            "tpl": tpl,
            "blocked": bool(blocked),
            "blocked_by": blocked,
        }
        if with_cats:
            out["cats"] = list(tr.get("cats") or [])
        return out


    # ---------- 推演路径 / 工具台账 / 失败轨迹 / 策略 ----------
    def pulse(self, query: str, limit: int = 6) -> dict[str, Any]:
        """按任务文本推演一条 T0→T4 的走法，供页面做流光动画。"""
        q = str(query or "").strip()
        if not q:
            return {"ok": False, "reason": "空查询"}
        hits = self.classify(q, limit=limit)
        reasons = {h["no"]: h["hits"] for h in hits}
        scores = {h["no"]: float(h["score"]) for h in hits}
        if re.search(r"\b(10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)", q):
            for no, add in ((1, 5.0), (3, 4.0)):
                if no in self._cats:
                    scores[no] = scores.get(no, 0.0) + add
                    reasons.setdefault(no, []).insert(0, "私网地址")
        picked = [no for no, sc in scores.items() if sc >= MIN_SCORE]
        if not picked:
            return {"ok": False, "reason": "没命中任何类别（换个更具体的任务描述）", "query": q}
        tier_of = {}
        for idx, tier in enumerate(self._tiers):
            for no in tier.get("ids") or []:
                tier_of[int(no)] = idx
        picked.sort(key=lambda no: (tier_of.get(no, 9), -scores[no]))
        nodes = [{
            "no": no,
            "name": self._cats[no].name,
            "short": self._cats[no].short,
            "tier": tier_of.get(no, 0),
            "score": round(scores.get(no, 0.0), 1),
            "hits": (reasons.get(no) or [])[:4],
        } for no in picked]
        segments = []
        curated = {(min(a, b), max(a, b)) for a, b in self._edges}
        for a, b in zip(picked, picked[1:]):
            kind = "curated" if (min(a, b), max(a, b)) in curated else "jump"
            segments.append({"from": a, "to": b, "kind": kind})
        return {
            "ok": True,
            "query": q,
            "nodes": nodes,
            "segments": segments,
            "path": picked,
            "steps": len(picked),
            "kind": "推理链（未执行）",
        }

    def strategy(self) -> dict[str, Any]:
        path = self.root / "strategy.json"
        if not path.is_file():
            return {"text": "", "deny": [], "prefer": [], "updated": ""}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"text": "", "deny": [], "prefer": [], "updated": ""}
        if not isinstance(data, dict):
            return {"text": "", "deny": [], "prefer": [], "updated": ""}
        data.setdefault("deny", [])
        data.setdefault("prefer", [])
        data.setdefault("text", "")
        return data

    def save_strategy(self, text: str) -> dict[str, Any]:
        strat = parse_strategy(text)
        path = self.root / "strategy.json"
        path.write_text(json.dumps(strat, ensure_ascii=False, indent=1), encoding="utf-8")
        return strat

    def deny_rules(self) -> list[dict[str, str]]:
        return [{"pattern": str(d.get("pattern") or ""), "reason": "策略禁止 " + str(d.get("label") or d.get("pattern") or "")}
                for d in (self.strategy().get("deny") or []) if d.get("pattern")]

    def fails(self) -> dict[str, Any]:
        state = self.state()
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for tr in state["fails"]:
            by_cat.setdefault(str(tr["no"]), []).append(tr)
        return {"ok": True, "total": len(state["fails"]), "traces": state["fails"], "by_cat": by_cat}

    def export(self, path: str | Path) -> Path:
        """导出成图谱页可吃的 EV / edges 结构（不覆盖原 meta，单独一份）。"""
        state = self.state()
        ev = {}
        for no_str, cat in state["cats"].items():
            ev[no_str] = {
                "lv": cat["level"],
                "note": cat["note"] or f"自动回写：证据 {cat['ok']} 成功 / {cat['fail']} 失败",
                "tools": cat.get("tools_detail") or [],
                "templates": list((cat.get("templates") or {}).keys())[:6],
                "fails": cat.get("fails") or 0,
                "fail_last": cat.get("fail_last"),
            }
        edges = list(self._edges) + [[e["from"], e["to"]] for e in state["auto_edges"]]
        payload = dict(self._index)          # 先继承原页全部字段：D/SHORT/ST/KN/SX/CX/RV
        payload.update({
            "meta": {"v": "v55+evidence", "t": time.strftime("%Y-%m-%d %H:%M", time.localtime())},
            "D": dict(self._index.get("D", {}), edges=edges),
            "EV": ev,
            "auto_edges": state["auto_edges"],
            "tool_edges": state["tool_edges"],
            "toolbox": state["tool_totals"],
            "strategy": state["strategy"],
            "host": state["host"],
            "fails": state["fails"],
            "edges": edges,
        })
        out = Path(path).expanduser()
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        self.snapshot(state)
        return out

    # ---------- 快照与对比 ----------
    def _snapshots(self) -> list[Path]:
        folder = self.root / "snapshots"
        if not folder.is_dir():
            return []
        return sorted(folder.glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name))

    def snapshot(self, state: dict[str, Any] | None = None) -> Path | None:
        """把当前状态落一份快照；与上一份完全相同就不写，保留最近 60 份。"""
        state = state or self.state()
        body = {
            "evidence": state["evidence_total"],
            "levels": state["levels"],
            "regressed": state["regressed"],
            "ev": {no: c["level"] for no, c in state["cats"].items()},
            "auto": [[e["from"], e["to"]] for e in state["auto_edges"]],
            "tool": [[e["tool"], e["cat"]] for e in state["tool_edges"]],
        }
        snaps = self._snapshots()
        if snaps:
            try:
                last = json.loads(snaps[-1].read_text(encoding="utf-8"))
                if last.get("body") == body:
                    return None
            except (OSError, json.JSONDecodeError):
                pass
        folder = self.root / "snapshots"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = folder / f"{stamp}.json"
        n = 1
        while path.exists():
            path = folder / f"{stamp}-{n}.json"
            n += 1
        path.write_text(json.dumps({"ts": round(self._clock(), 3), "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "body": body}, ensure_ascii=False, indent=1), encoding="utf-8")
        for old in self._snapshots()[:-60]:
            try:
                old.unlink()
            except OSError:
                pass
        return path

    def diff(self) -> dict[str, Any]:
        """最近两份快照的差异：新边、消失边、等级变化、证据增量。"""
        snaps = self._snapshots()
        if len(snaps) < 2:
            return {"ok": False, "reason": "至少需要两份快照（先 export 或回写一次）"}
        try:
            a = json.loads(snaps[-2].read_text(encoding="utf-8"))
            b = json.loads(snaps[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"ok": False, "reason": f"快照不可读：{exc}"}
        ba, bb = a.get("body") or {}, b.get("body") or {}
        pa = {f"{x[0]}->{x[1]}" for x in ba.get("auto") or []}
        pb = {f"{x[0]}->{x[1]}" for x in bb.get("auto") or []}
        na = {f"{x[0]}->{x[1]}" for x in ba.get("tool") or []}
        nb = {f"{x[0]}->{x[1]}" for x in bb.get("tool") or []}
        changes = []
        for no, lv in (bb.get("ev") or {}).items():
            old = (ba.get("ev") or {}).get(no)
            if old and old != lv:
                changes.append({"no": int(no), "from": old, "to": lv})
        return {
            "ok": True,
            "from": {"at": a.get("at"), "evidence": ba.get("evidence")},
            "to": {"at": b.get("at"), "evidence": bb.get("evidence")},
            "evidence_delta": (bb.get("evidence") or 0) - (ba.get("evidence") or 0),
            "new_edges": sorted(pb - pa),
            "gone_edges": sorted(pa - pb),
            "new_tool_edges": sorted(nb - na),
            "gone_tool_edges": sorted(na - nb),
            "level_changes": changes,
            "snapshots": len(snaps),
        }


def ledger_from_config(workspace: str | Path) -> TreeLedger:
    return TreeLedger(Path(workspace) / "tree")
