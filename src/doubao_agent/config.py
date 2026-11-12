from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PORT = 8765


def _default_workspace() -> Path:
    """工作区根目录：优先 AGENT_WORKSPACE，其次当前目录（不再写死 /workspace）。"""
    return Path(os.environ.get("AGENT_WORKSPACE") or Path.cwd()).resolve()


@dataclass
class MCPServerConfig:
    name: str
    transport: str = "stdio"          # stdio | sse | http
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ApiSourceConfig:
    name: str
    spec_path: str


@dataclass
class Config:
    workspace: Path = field(default_factory=_default_workspace)
    runs_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("AGENT_RUNS_DIR") or _default_workspace() / "runs")
    )
    db_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("AGENT_DB_PATH") or _default_workspace() / ".agent" / "agent.db"
        )
    )
    permission_mode: str = "guarded"   # auto | guarded | manual
    script_risk: str = "L1"            # 脚本通道的风险等级：L1 直接执行，L2 起需确认
    script_max_cpu: float = 0.0        # 脚本 CPU 时间上限（秒），0 表示不限
    record_recipes: bool = True        # 成功任务自动录制为可零模型回放的配方
    planning_enabled: bool = False     # 入口单次规划：每任务 1 次模型调用，执行阶段零推理
    skills_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("AGENT_SKILLS_DIR") or _default_workspace() / "skills"
        )
    )
    context_limit: int = 40            # 单次请求最多携带的消息条数，超出按 user 边界裁剪
    scheduler_enabled: bool = True     # 控制台/守护进程是否轮询 cron 调度
    scheduler_interval: float = 30.0   # 调度轮询间隔（秒）
    max_steps: int = 12
    max_calls: int = 0                 # 单次运行模型调用预算，0 表示不限
    max_tokens: int = 0                # 单次运行 token 预算，0 表示不限
    repair_rounds: int = 1             # 规划执行失败时允许模型修正参数的轮数
    port: int = DEFAULT_PORT
    auth_token: str = ""               # 控制台 /api 鉴权令牌；为空表示不鉴权（仅限本机）
    model: dict[str, Any] = field(default_factory=dict)
    bridge: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, Any] = field(default_factory=dict)  # 宿主 embedding 能力，默认自动探测
    auto_approve: list[dict[str, Any]] = field(default_factory=list)  # 规则预授权，替代人工审批
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)
    mcp_auto_discover: bool = True    # 未配置 MCP 时，启动探测云电脑标准 MCP（8091）
    api_sources: list[ApiSourceConfig] = field(default_factory=list)

    def ensure_dirs(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


def _default_model() -> dict[str, Any]:
    if os.environ.get("OPENAI_API_KEY"):
        return {
            "provider": "openai",
            "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "api_key_env": "OPENAI_API_KEY",
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0.2,
        }
    if os.environ.get("AGENT_BRIDGE_URL"):
        # 豆包 App 云电脑：模型能力由宿主提供，无需 key。
        return {"provider": "host"}
    return {"provider": "mock"}


def _parse_mcp(items: list[dict[str, Any]]) -> list[MCPServerConfig]:
    out = []
    for item in items or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        out.append(
            MCPServerConfig(
                name=str(item["name"]),
                transport=item.get("transport", "stdio"),
                command=item.get("command"),
                args=list(item.get("args", [])),
                url=item.get("url"),
                env=dict(item.get("env", {})),
            )
        )
    return out


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    cfg_path = Path(
        path or os.environ.get("DOUBAO_AGENT_CONFIG") or (cfg.workspace / "agent.yaml")
    )
    if not cfg_path.exists():
        legacy = Path("/workspace/agent.yaml")
        if legacy.exists():
            cfg_path = legacy

    if cfg_path.exists():
        try:
            data = yaml.safe_load(cfg_path.read_text("utf-8")) or {}
        except (yaml.YAMLError, OSError):
            # 坏配置不能拖垮启动：回退默认配置，其余渠道照常工作。
            data = {}
        if not isinstance(data, dict):
            data = {}
        cfg.workspace = Path(data.get("workspace", cfg.workspace))
        cfg.runs_dir = Path(data.get("runs_dir", cfg.workspace / "runs"))
        cfg.db_path = Path(data.get("db_path", cfg.workspace / ".agent" / "agent.db"))
        cfg.permission_mode = data.get("permission_mode", cfg.permission_mode)
        cfg.script_risk = data.get("script_risk", cfg.script_risk)
        cfg.script_max_cpu = float(data.get("script_max_cpu", cfg.script_max_cpu))
        cfg.record_recipes = bool(data.get("record_recipes", cfg.record_recipes))
        cfg.skills_dir = Path(data.get("skills_dir", cfg.workspace / "skills"))
        cfg.context_limit = max(4, int(data.get("context_limit", cfg.context_limit)))
        cfg.scheduler_enabled = bool(data.get("scheduler_enabled", cfg.scheduler_enabled))
        cfg.scheduler_interval = max(5.0, float(data.get("scheduler_interval", cfg.scheduler_interval)))
        planning = data.get("planning") or {}
        if isinstance(planning, dict):
            cfg.planning_enabled = bool(planning.get("enabled", cfg.planning_enabled))
        cfg.max_steps = int(data.get("max_steps", cfg.max_steps))
        cfg.max_calls = max(0, int(data.get("max_calls", cfg.max_calls)))
        cfg.max_tokens = max(0, int(data.get("max_tokens", cfg.max_tokens)))
        cfg.repair_rounds = max(0, int(data.get("repair_rounds", cfg.repair_rounds)))
        cfg.port = int(data.get("port", cfg.port))
        cfg.auth_token = str(data.get("auth_token", cfg.auth_token) or "")
        cfg.model = dict(data.get("model") or _default_model())
        cfg.bridge = dict(data.get("bridge") or {})
        cfg.embedding = dict(data.get("embedding") or {})
        cfg.auto_approve = [
            dict(item) for item in data.get("auto_approve", []) if isinstance(item, dict)
        ]
        cfg.mcp_servers = _parse_mcp(data.get("mcp_servers", []))
        cfg.mcp_auto_discover = bool(data.get("mcp_auto_discover", cfg.mcp_auto_discover))
        cfg.api_sources = [
            ApiSourceConfig(name=str(i["name"]), spec_path=str(i["spec"]))
            for i in data.get("api_sources", [])
            if isinstance(i, dict) and i.get("name") and i.get("spec")
        ]

    # Environment overrides win, so the cloud desktop can inject config at runtime.
    cfg.port = int(os.environ.get("AGENT_PORT", cfg.port))
    cfg.auth_token = os.environ.get("AGENT_AUTH_TOKEN", cfg.auth_token)
    cfg.permission_mode = os.environ.get("AGENT_PERMISSION_MODE", cfg.permission_mode)
    cfg.script_risk = os.environ.get("AGENT_SCRIPT_RISK", cfg.script_risk)
    if os.environ.get("AGENT_SCRIPT_MAX_CPU"):
        cfg.script_max_cpu = float(os.environ["AGENT_SCRIPT_MAX_CPU"])
    if os.environ.get("AGENT_SCHEDULER_ENABLED") is not None:
        cfg.scheduler_enabled = os.environ["AGENT_SCHEDULER_ENABLED"].lower() in ("1", "true", "yes", "on")
    if os.environ.get("AGENT_SCHEDULER_INTERVAL"):
        cfg.scheduler_interval = max(5.0, float(os.environ["AGENT_SCHEDULER_INTERVAL"]))
    if os.environ.get("AGENT_RECORD_RECIPES") is not None:
        cfg.record_recipes = os.environ["AGENT_RECORD_RECIPES"].lower() in ("1", "true", "yes", "on")
    if os.environ.get("AGENT_PLANNING") is not None:
        cfg.planning_enabled = os.environ["AGENT_PLANNING"].lower() in ("1", "true", "yes", "on")
    cfg.skills_dir = Path(os.environ.get("AGENT_SKILLS_DIR", cfg.skills_dir))
    if os.environ.get("AGENT_CONTEXT_LIMIT"):
        cfg.context_limit = max(4, int(os.environ["AGENT_CONTEXT_LIMIT"]))
    if os.environ.get("AGENT_MAX_CALLS"):
        cfg.max_calls = max(0, int(os.environ["AGENT_MAX_CALLS"]))
    if os.environ.get("AGENT_MAX_TOKENS"):
        cfg.max_tokens = max(0, int(os.environ["AGENT_MAX_TOKENS"]))
    if os.environ.get("AGENT_REPAIR_ROUNDS"):
        cfg.repair_rounds = max(0, int(os.environ["AGENT_REPAIR_ROUNDS"]))
    cfg.db_path = Path(os.environ.get("AGENT_DB_PATH", cfg.db_path))
    cfg.runs_dir = Path(os.environ.get("AGENT_RUNS_DIR", cfg.runs_dir))
    if not cfg.model:
        cfg.model = _default_model()

    bridge_url = os.environ.get("AGENT_BRIDGE_URL")
    if bridge_url:
        cfg.bridge["url"] = bridge_url
        cfg.bridge["enabled"] = True
    if os.environ.get("AGENT_BRIDGE_TOKEN"):
        cfg.bridge["token"] = os.environ["AGENT_BRIDGE_TOKEN"]
    if os.environ.get("AGENT_MCP_AUTO_DISCOVER") is not None:
        cfg.mcp_auto_discover = os.environ["AGENT_MCP_AUTO_DISCOVER"].lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    if os.environ.get("AGENT_EMBEDDING_CAPABILITY"):
        cfg.embedding["capability"] = os.environ["AGENT_EMBEDDING_CAPABILITY"]

    # 运行时登记的 MCP（doubao-agent mcp add/remove 写入），不污染带注释的 agent.yaml。
    managed = cfg.db_path.parent / "mcp.local.yaml"
    if managed.is_file():
        try:
            extra = yaml.safe_load(managed.read_text(encoding="utf-8")) or []
        except (yaml.YAMLError, OSError):
            extra = []
        known = {server.name for server in cfg.mcp_servers}
        cfg.mcp_servers.extend(s for s in _parse_mcp(extra) if s.name not in known)

    cfg.ensure_dirs()
    return cfg
