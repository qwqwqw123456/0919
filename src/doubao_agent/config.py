from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PORT = 8765


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
    workspace: Path = Path("/workspace")
    runs_dir: Path = Path("/workspace/runs")
    db_path: Path = Path("/workspace/.agent/agent.db")
    permission_mode: str = "guarded"   # auto | guarded | manual
    script_risk: str = "L1"            # 脚本通道的风险等级：L1 直接执行，L2 起需确认
    script_max_cpu: float = 0.0        # 脚本 CPU 时间上限（秒），0 表示不限
    record_recipes: bool = True        # 成功任务自动录制为可零模型回放的配方
    planning_enabled: bool = False     # 入口单次规划：每任务 1 次模型调用，执行阶段零推理
    skills_dir: Path = Path("/workspace/skills")  # Skill 包目录（*/SKILL.md）
    context_limit: int = 40            # 单次请求最多携带的消息条数，超出按 user 边界裁剪
    scheduler_enabled: bool = True     # 控制台/守护进程是否轮询 cron 调度
    scheduler_interval: float = 30.0   # 调度轮询间隔（秒）
    max_steps: int = 12
    port: int = DEFAULT_PORT
    model: dict[str, Any] = field(default_factory=dict)
    bridge: dict[str, Any] = field(default_factory=dict)
    auto_approve: list[dict[str, Any]] = field(default_factory=list)  # 规则预授权，替代人工审批
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)
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
        out.append(
            MCPServerConfig(
                name=item["name"],
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
    cfg_path = Path(path or os.environ.get("DOUBAO_AGENT_CONFIG", "/workspace/agent.yaml"))

    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text("utf-8")) or {}
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
        cfg.port = int(data.get("port", cfg.port))
        cfg.model = dict(data.get("model") or _default_model())
        cfg.bridge = dict(data.get("bridge") or {})
        cfg.auto_approve = [
            dict(item) for item in data.get("auto_approve", []) if isinstance(item, dict)
        ]
        cfg.mcp_servers = _parse_mcp(data.get("mcp_servers", []))
        cfg.api_sources = [
            ApiSourceConfig(name=i["name"], spec_path=i["spec"])
            for i in data.get("api_sources", [])
        ]

    # Environment overrides win, so the cloud desktop can inject config at runtime.
    cfg.port = int(os.environ.get("AGENT_PORT", cfg.port))
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

    cfg.ensure_dirs()
    return cfg
