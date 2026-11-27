from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from ..types import Risk, SourceKind, ToolCall, ToolResult, ToolSpec

MAX_OUTPUT = 8000


class ToolboxSource:
    """本地安全工具箱通道：把 `~/tools` 里的 CLI 以"目录 + 受控执行"的方式接进体系。

    一个通道三个动作（不逐个工具注册成独立 tool，避免几十个 schema 撑爆路由）：
    - list  按类查能力（静态分析 / 反编译 / 抓包 / 取证……），给出可用状态与替代
    - probe 真跑一次版本/帮助，验证工具此刻是否起得来
    - run   白名单执行：只认 catalog 里声明过的 entry，参数走列表、拼不进 shell
    """

    name = "toolbox"
    kind = SourceKind.BUILTIN

    def __init__(self, catalog_path: str | Path, default_timeout: float = 120.0) -> None:
        self.catalog_path = Path(catalog_path).expanduser()
        self.default_timeout = default_timeout
        self._tools: dict[str, dict[str, Any]] = {}
        self._load_error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            data = yaml.safe_load(self.catalog_path.read_text("utf-8")) or {}
        except Exception as exc:  # noqa: BLE001 - 目录缺失只降级，不影响其它通道
            self._load_error = f"{type(exc).__name__}: {exc}"
            return
        for item in data.get("tools") or []:
            if isinstance(item, dict) and item.get("name"):
                self._tools[str(item["name"])] = item

    @property
    def tools(self) -> dict[str, dict[str, Any]]:
        return dict(self._tools)

    async def list_tools(self) -> list[ToolSpec]:
        if self._load_error:
            return []
        return [
            ToolSpec(
                name="list",
                description=(
                    "列出本地安全工具箱能力（按 Android/Web/Native/固件/代码/Windows/恶意样本 等类）。"
                    "遇到反编译、脱壳、查壳、抓包改包、二进制调试、内存取证、恶意样本初筛、"
                    "固件解包、静态代码扫描这类活，先查这里有没有现成工具，再考虑自己写脚本。"
                    "返回每项的状态（ok/source_only/unavailable）与不可用项的替代物。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "description": "只看某一类，如 Web/Native/Android"},
                        "keyword": {"type": "string", "description": "按名称或用途关键词过滤"},
                        "only_ready": {"type": "boolean", "description": "只列当前可直接执行的", "default": False},
                    },
                },
                kind=self.kind,
                risk=Risk.READ,
                idempotent=True,
                source=self.name,
                tags=("security", "toolbox"),
            ),
            ToolSpec(
                name="probe",
                description=(
                    "对工具箱里某一项真跑一次版本/帮助命令，确认它此刻能否启动。"
                    "catalog 说 ok 只是登记状态，探活才代表现在可用。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "工具名，如 jadx、rizin、capa"},
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "可选，覆盖默认探活参数",
                        },
                    },
                    "required": ["name"],
                },
                kind=self.kind,
                risk=Risk.READ,
                idempotent=True,
                source=self.name,
                tags=("security", "toolbox"),
            ),
            ToolSpec(
                name="scan",
                description=(
                    "对明确授权目标执行主动探测类工具（nuclei/ffuf/sqlmap/nxc/nmap/ip-checker 等，catalog 里 risk=L2 的项）。"
                    "与 run 同一套白名单与超时控制，但标记 L2，需权限门放行；目标必须已在任务里明确授权。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "工具名，如 nuclei、ffuf、nmap"},
                        "args": {"type": "array", "items": {"type": "string"}, "description": "参数列表"},
                        "timeout": {"type": "number", "description": "超时秒数，默认 120，上限 600"},
                        "cwd": {"type": "string", "description": "工作目录"},
                    },
                    "required": ["name"],
                },
                kind=self.kind,
                risk=Risk.IRREVERSIBLE,
                idempotent=False,
                source=self.name,
                tags=("security", "toolbox", "active"),
            ),
            ToolSpec(
                name="run",
                description=(
                    "在本地执行工具箱里的一个已登记工具。只接受 catalog 里存在的工具名，"
                    "参数以列表传（不拼 shell 字符串），超时与输出截断由本通道兜住。"
                    "主动打流量的工具（nuclei/ffuf/sqlmap/nxc/nmap）标记 L2，需明确授权目标；"
                    "静态分析类（jadx/apktool/capa/yara/ghidra-headless）为 L0/L1，可直接跑。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "工具名，如 jadx、apktool、capa"},
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "参数列表，如 [\"-d\", \"/tmp/out\", \"app.apk\"]",
                        },
                        "timeout": {"type": "number", "description": "超时秒数，默认 120，上限 600"},
                        "cwd": {"type": "string", "description": "工作目录，默认当前工作区"},
                    },
                    "required": ["name"],
                },
                kind=self.kind,
                risk=Risk.REVERSIBLE,
                idempotent=False,
                source=self.name,
                tags=("security", "toolbox"),
            ),
        ]

    def _find(self, name: str) -> dict[str, Any] | None:
        key = str(name or "").strip()
        if key in self._tools:
            return self._tools[key]
        low = key.lower()
        for k, v in self._tools.items():
            if k.lower() == low:
                return v
        # 也认入口名（模型更可能直接说 ghidra-headless 而不是清单里的 Ghidra）
        for v in self._tools.values():
            if v.get("entry") and str(v["entry"]).lower() == low:
                return v
        return None

    def _format(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "工具箱里没有匹配项。"
        lines = []
        for r in rows:
            head = f"{r['name']}　[{r['category']}]　{r['status']}　{r['risk']}"
            lines.append(head)
            lines.append(f"  用途：{r['desc']}")
            if r["entry"]:
                lines.append(f"  入口：{r['entry']}")
            if r["status"] != "ok" and r.get("substitute"):
                lines.append(f"  替代：{r['substitute']}")
        ready = sum(1 for r in rows if r["status"] == "ok")
        lines.append(f"\n共 {len(rows)} 项，其中可直接执行 {ready} 项。")
        return "\n".join(lines)

    async def call_tool(self, call: ToolCall) -> ToolResult:
        args = call.arguments or {}
        if self._load_error:
            return ToolResult(
                call_id=call.call_id, ok=False, error=f"工具箱目录不可读：{self._load_error}"
            )

        if call.name == "list":
            rows = list(self._tools.values())
            cat = str(args.get("category") or "").strip().lower()
            if cat:
                rows = [r for r in rows if str(r["category"]).lower() == cat]
            kw = str(args.get("keyword") or "").strip().lower()
            if kw:
                rows = [
                    r
                    for r in rows
                    if kw in str(r["name"]).lower()
                    or kw in str(r["desc"]).lower()
                    or kw in str(r["category"]).lower()
                ]
            if args.get("only_ready"):
                rows = [r for r in rows if r["status"] == "ok"]
            return ToolResult(
                call_id=call.call_id, ok=True, content=self._format(rows), meta={"count": len(rows)}
            )

        if call.name in ("probe", "run", "scan"):
            item = self._find(str(args.get("name") or ""))
            if item is None:
                return ToolResult(
                    call_id=call.call_id,
                    ok=False,
                    error=f"工具箱里没有 '{args.get('name')}'；先用 toolbox.list 查",
                )
            if item["status"] != "ok" or not item.get("entry"):
                alt = item.get("substitute")
                hint = f"；可用替代：{alt}" if alt else ""
                return ToolResult(
                    call_id=call.call_id,
                    ok=False,
                    error=f"{item['name']} 当前状态 {item['status']}，本机不可执行{hint}",
                )

            item_risk = str(item.get("risk") or "L1")
            if call.name == "run" and item_risk == "L2":
                return ToolResult(
                    call_id=call.call_id,
                    ok=False,
                    error=(
                        f"{item['name']} 属主动探测（L2），不走 run；"
                        "目标已明确授权时改用 toolbox.scan（过权限门）"
                    ),
                )
            if call.name == "scan" and item_risk != "L2":
                return ToolResult(
                    call_id=call.call_id,
                    ok=False,
                    error=f"{item['name']} 是静态/本地类（{item_risk}），直接用 toolbox.run",
                )

            entry = shutil.which(str(item["entry"]))
            if entry is None:
                return ToolResult(
                    call_id=call.call_id,
                    ok=False,
                    error=f"入口 {item['entry']} 不在 PATH（catalog 与实物不符，需重装或修正登记）",
                )

            argv = [entry]
            raw_args = args.get("args")
            if call.name == "probe" and not raw_args:
                argv += ["--version"] if item["name"] in PROBE_VERSION else ["-h"]
            elif isinstance(raw_args, list):
                argv += [str(a) for a in raw_args]
            elif raw_args is not None:
                return ToolResult(call_id=call.call_id, ok=False, error="args 必须是字符串列表")

            timeout = min(600.0, max(1.0, float(args.get("timeout") or self.default_timeout)))
            cwd = str(args.get("cwd") or os.getcwd())
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=cwd,
                    env={"PATH": os.environ.get("PATH", ""), "HOME": os.path.expanduser("~"), "LANG": "C.UTF-8"},
                )
                try:
                    out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    return ToolResult(
                        call_id=call.call_id, ok=False, error=f"{item['name']} 超时（{timeout:.0f}s）已终止"
                    )
            except FileNotFoundError as exc:
                return ToolResult(call_id=call.call_id, ok=False, error=f"启动失败：{exc}")

            text = (out or b"").decode("utf-8", "replace").strip()
            truncated = len(text) > MAX_OUTPUT
            body = text[:MAX_OUTPUT] + "\n…（输出已截断）" if truncated else text
            if proc.returncode != 0 and call.name == "probe":
                return ToolResult(
                    call_id=call.call_id,
                    ok=False,
                    error=f"{item['name']} 探活退出码 {proc.returncode}：{body[:400]}",
                )
            return ToolResult(
                call_id=call.call_id,
                ok=True,
                content=body or f"{item['name']} 无输出（退出码 {proc.returncode}）",
                meta={"argv": argv, "exit": proc.returncode, "truncated": truncated},
            )

        return ToolResult(call_id=call.call_id, ok=False, error=f"unknown toolbox action: {call.name}")

    async def health(self) -> tuple[bool, str]:
        if self._load_error:
            return False, self._load_error
        ok = sum(1 for v in self._tools.values() if v["status"] == "ok")
        return True, f"catalog {len(self._tools)} 项，可直接执行 {ok} 项"


PROBE_VERSION = {
    "jadx", "apktool", "androguard", "objection", "APKiD", "fernflower", "frida", "Scrapling",
    "CipherBridge", "nuclei", "ffuf", "mitmproxy", "impacket", "netexec", "sqlmap", "radare2",
    "rizin", "capa", "YARA", "UPX", "pwntools", "ROPgadget", "semgrep", "dex2jar", "CFR",
    "nmap", "Wireshark", "Volatility3", "oletools", "garak", "binwalk", "unblob",
}
