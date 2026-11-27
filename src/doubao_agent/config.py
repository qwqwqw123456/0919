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


def _load_env_file(path: Path) -> None:
    """从 .env.local 注入环境变量；已存在的环境变量优先，不覆盖。

    仅解析 KEY=VALUE 行，跳过注释与空行；文件缺失或损坏时静默跳过，不拖垮启动。
    """
    if not path.is_file():
        return
    try:
        lines = path.read_text("utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _load_local_env() -> None:
    """加载当前目录与工作区下的 .env.local（不引入 python-dotenv 依赖）。"""
    _load_env_file(Path.cwd() / ".env.local")
    _load_env_file(_default_workspace() / ".env.local")


@dataclass
class MCPServerConfig:
    name: str
    transport: str = "stdio"          # stdio | sse | http
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    inherit_env: bool = False          # stdio 子进程默认只拿最小环境，不继承全量（含密钥）
    retry_cooldown: float = 60.0       # 初始化失败后的冷却时间，冷却结束允许重连


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
    voice_enabled: bool = True        # 语音通道：本地 Piper 合成 + faster-whisper 识别
    voice_config: dict = field(default_factory=dict)  # asr/tts/模型路径等，见 agent.yaml
    toolbox_enabled: bool = True       # 本地安全工具箱通道（~/tools 下的 CLI 以目录方式接入）
    toolbox_catalog: Path = field(
        default_factory=lambda: Path(os.environ.get("AGENT_TOOLBOX_CATALOG") or "")
    )
    script_risk: str = "L1"            # 脚本通道的风险等级：L1 直接执行，L2 起需确认
    script_max_cpu: float = 0.0        # 脚本 CPU 时间上限（秒），0 表示不限
    record_recipes: bool = True        # 成功任务自动录制为可零模型回放的配方
    evolution_enabled: bool = True     # 技能用量遥测 + 生命周期审计（抄自 Hermes curator）
    evolution_stale_days: int = 14     # 多久没用算闲置
    evolution_archive_days: int = 30   # 多久没用进归档状态（只改状态位，不删文件）
    provider_retries: int = 1          # 端点异常重试次数；401/403 不重试，直接进降级判断
    provider_retry_delay: float = 0.2  # 重试退避基数（秒），第 n 次重试按 n 倍放大
    provider_fallback: str = ""        # 备用 provider："" 不降级（默认）；"mock" 用内置 mock
    tool_cache_enabled: bool = True    # 幂等只读工具结果缓存开关
    tool_cache_ttl: float = 60.0       # 结果缓存存活秒数；0 或开关关闭即不缓存
    tool_channel_concurrency: int = 3  # 单通道工具并发上限；0 表示不限制
    planning_enabled: bool = False     # 入口单次规划：每任务 1 次模型调用，执行阶段零推理
    pipeline_enabled: bool = False     # 任务流水线：拆解→逐原子步骤→独立核验→定点修复
    pipeline_max_atoms: int = 6        # 流水线最多拆几个原子步骤
    pipeline_atom_steps: int = 4       # 单个原子步骤最多几轮模型/工具往返
    pipeline_repair_rounds: int = 1    # 核验未通过时允许修复的轮数
    pipeline_deadline_seconds: float = 0.0  # 0=不限；一条任务走流水线的整体时限
    pipeline_only_risk: bool = True  # 只有高风险任务进三阶段，其余走单循环
    guard_abort_seconds: float = 1800.0  # 单次运行硬上限，0=关闭；防跑飞
    guard_warn_tokens: int = 500_000  # 单次运行 token 超此值只告警不拦
    calibration_sample_rate: float = 0.1  # 分流标定采样率：按此比例落任务原文供离线回放
    skills_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("AGENT_SKILLS_DIR") or _default_workspace() / "skills"
        )
    )
    context_limit: int = 40            # 单次请求最多携带的消息条数，超出按 user 边界裁剪
    context_model_summary: bool = False  # 压缩早期消息时是否调模型浓缩（默认关，避免额外模型调用）
    scheduler_enabled: bool = True     # 控制台/守护进程是否轮询 cron 调度
    scheduler_interval: float = 30.0   # 调度轮询间隔（秒）
    watch_enabled: bool = False        # 文件变更监听开关（默认关，避免默认扫工作区起任务）
    watch_dirs: list[str] = field(default_factory=list)  # 监听目录，空=工作区
    watch_interval: float = 10.0       # 目录快照轮询间隔（秒）
    watch_task: str = ""               # 变更触发任务模板，占位符 {path}/{event}；空=只广播事件不起任务
    watch_suffixes: list[str] = field(default_factory=list)  # 只监听这些后缀，空=全部
    max_steps: int = 12
    max_calls: int = 0                 # 单次运行模型调用预算，0 表示不限
    max_tokens: int = 0                # 单次运行 token 预算，0 表示不限
    repair_rounds: int = 1             # 规划执行失败时允许模型修正参数的轮数
    tool_retries: int = 1              # 工具失败后按策略重试的次数（仅幂等工具）
    tool_rate_limits: dict[str, int] = field(default_factory=dict)  # 按来源的每分钟调用上限
    port: int = DEFAULT_PORT
    serve_host: str = "127.0.0.1"      # 控制台监听地址：默认只绑回环，绑全网必须同时配 auth_token
    auth_token: str = ""               # 控制台 /api 鉴权令牌；为空表示不鉴权（仅限本机）
    a2a_allow: list[str] = field(default_factory=lambda: ["127.0.0.1", "localhost", "::1"])  # A2A 出站白名单
    model: dict[str, Any] = field(default_factory=dict)
    model_fallback: dict[str, Any] = field(default_factory=dict)  # 备用真实 provider 配置，结构同 model；空=不启用
    models: list[dict[str, Any]] = field(default_factory=list)     # 可选通道清单，供 model use 快切
    bridge: dict[str, Any] = field(default_factory=dict)
    outbound: dict[str, Any] = field(default_factory=dict)  # 出站通道：console / feishu / smtp
    secret_sources: dict[str, str] = field(default_factory=dict)  # 兜底补进来的变量 -> 来源（不含值）
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
                inherit_env=bool(item.get("inherit_env", False)),
                retry_cooldown=max(0.0, float(item.get("retry_cooldown", 60.0))),
            )
        )
    return out


def _secret_names(cfg: Config) -> list[str]:
    """需要在兜底层找的变量名：各 provider 的 api_key_env + 鉴权令牌。"""
    names: list[str] = []
    for item in (cfg.model, cfg.model_fallback, cfg.embedding):
        if not isinstance(item, dict):
            continue
        env = item.get("api_key_env")
        if env:
            names.append(str(env))
    names.append("AGENT_AUTH_TOKEN")
    return list(dict.fromkeys(names))


def _fill_secrets(cfg: Config) -> dict[str, str]:
    """环境没有的密钥，从 vault / keyring 兜底补进环境；只记录来源名，不记值。"""
    from .core.secrets import apply_fallback

    return apply_fallback(_secret_names(cfg))


def load_config(path: str | Path | None = None) -> Config:
    _load_local_env()
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
        voice = data.get("voice") or {}
        if isinstance(voice, dict):
            cfg.voice_enabled = bool(voice.get("enabled", cfg.voice_enabled))
            cfg.voice_config = dict(voice)
        toolbox = data.get("toolbox") or {}
        if isinstance(toolbox, dict):
            cfg.toolbox_enabled = bool(toolbox.get("enabled", cfg.toolbox_enabled))
            catalog = toolbox.get("catalog")
            if catalog:
                cfg.toolbox_catalog = Path(str(catalog)).expanduser()
        cfg.script_risk = data.get("script_risk", cfg.script_risk)
        cfg.script_max_cpu = float(data.get("script_max_cpu", cfg.script_max_cpu))
        cfg.record_recipes = bool(data.get("record_recipes", cfg.record_recipes))
        evolution = data.get("evolution") or {}
        if isinstance(evolution, dict):
            cfg.evolution_enabled = bool(evolution.get("enabled", cfg.evolution_enabled))
            cfg.evolution_stale_days = max(
                1, int(evolution.get("stale_days", cfg.evolution_stale_days))
            )
            cfg.evolution_archive_days = max(
                cfg.evolution_stale_days,
                int(evolution.get("archive_days", cfg.evolution_archive_days)),
            )
        cfg.provider_retries = max(0, int(data.get("provider_retries", cfg.provider_retries)))
        cfg.provider_retry_delay = max(
            0.0, float(data.get("provider_retry_delay", cfg.provider_retry_delay))
        )
        cfg.provider_fallback = str(data.get("provider_fallback", cfg.provider_fallback) or "")
        cfg.tool_cache_enabled = bool(data.get("tool_cache_enabled", cfg.tool_cache_enabled))
        cfg.tool_cache_ttl = max(0.0, float(data.get("tool_cache_ttl", cfg.tool_cache_ttl)))
        cfg.tool_channel_concurrency = max(0, int(data.get("tool_channel_concurrency", cfg.tool_channel_concurrency)))
        cfg.skills_dir = Path(data.get("skills_dir", cfg.workspace / "skills"))
        cfg.context_limit = max(4, int(data.get("context_limit", cfg.context_limit)))
        cfg.context_model_summary = bool(
            data.get("context_model_summary", cfg.context_model_summary)
        )
        cfg.scheduler_enabled = bool(data.get("scheduler_enabled", cfg.scheduler_enabled))
        cfg.scheduler_interval = max(5.0, float(data.get("scheduler_interval", cfg.scheduler_interval)))
        cfg.watch_enabled = bool(data.get("watch_enabled", cfg.watch_enabled))
        cfg.watch_dirs = [str(x) for x in (data.get("watch_dirs") or [])]
        cfg.watch_interval = max(2.0, float(data.get("watch_interval", cfg.watch_interval)))
        cfg.watch_task = str(data.get("watch_task", cfg.watch_task) or "")
        cfg.watch_suffixes = [str(x) for x in (data.get("watch_suffixes") or [])]
        cfg.outbound = dict(data.get("outbound") or {})
        planning = data.get("planning") or {}
        if isinstance(planning, dict):
            cfg.planning_enabled = bool(planning.get("enabled", cfg.planning_enabled))
        pipeline = data.get("pipeline") or {}
        if isinstance(pipeline, dict):
            cfg.pipeline_enabled = bool(pipeline.get("enabled", cfg.pipeline_enabled))
            cfg.pipeline_max_atoms = max(1, int(pipeline.get("max_atoms", cfg.pipeline_max_atoms)))
            cfg.pipeline_atom_steps = max(1, int(pipeline.get("atom_steps", cfg.pipeline_atom_steps)))
            cfg.pipeline_repair_rounds = max(
                0, int(pipeline.get("repair_rounds", cfg.pipeline_repair_rounds))
            )
            cfg.pipeline_only_risk = bool(pipeline.get("only_risk", cfg.pipeline_only_risk))
            cfg.pipeline_deadline_seconds = max(
                0.0, float(pipeline.get("deadline_seconds", cfg.pipeline_deadline_seconds))
            )
            cfg.calibration_sample_rate = min(
                1.0, max(0.0, float(pipeline.get("sample_rate", cfg.calibration_sample_rate)))
            )
        guard = data.get("guard") or {}
        cfg.guard_abort_seconds = max(0.0, float(guard.get("abort_seconds", cfg.guard_abort_seconds)))
        cfg.guard_warn_tokens = max(0, int(guard.get("warn_tokens", cfg.guard_warn_tokens)))
        cfg.max_steps = int(data.get("max_steps", cfg.max_steps))
        cfg.max_calls = max(0, int(data.get("max_calls", cfg.max_calls)))
        cfg.max_tokens = max(0, int(data.get("max_tokens", cfg.max_tokens)))
        cfg.repair_rounds = max(0, int(data.get("repair_rounds", cfg.repair_rounds)))
        cfg.tool_retries = max(0, int(data.get("tool_retries", cfg.tool_retries)))
        limits = data.get("tool_rate_limits")
        if isinstance(limits, dict):
            cfg.tool_rate_limits = {
                str(key): max(0, int(value)) for key, value in limits.items()
            }
        cfg.port = int(data.get("port", cfg.port))
        cfg.serve_host = str(data.get("serve_host", cfg.serve_host) or "127.0.0.1")
        cfg.auth_token = str(data.get("auth_token", cfg.auth_token) or "")
        allow = data.get("a2a_allow")
        if isinstance(allow, list):
            cfg.a2a_allow = [str(x).strip() for x in allow if str(x).strip()]
        cfg.model = dict(data.get("model") or _default_model())
        cfg.model_fallback = dict(data.get("model_fallback") or {})
        cfg.models = [x for x in (data.get("models") or []) if isinstance(x, dict) and x.get("name")]
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
    allow_env = os.environ.get("AGENT_A2A_ALLOW")
    if allow_env is not None:
        cfg.a2a_allow = [x.strip() for x in allow_env.split(",") if x.strip()]
    cfg.permission_mode = os.environ.get("AGENT_PERMISSION_MODE", cfg.permission_mode)
    cfg.script_risk = os.environ.get("AGENT_SCRIPT_RISK", cfg.script_risk)
    if os.environ.get("AGENT_SCRIPT_MAX_CPU"):
        cfg.script_max_cpu = float(os.environ["AGENT_SCRIPT_MAX_CPU"])
    if os.environ.get("AGENT_SCHEDULER_ENABLED") is not None:
        cfg.scheduler_enabled = os.environ["AGENT_SCHEDULER_ENABLED"].lower() in ("1", "true", "yes", "on")
    if os.environ.get("AGENT_SCHEDULER_INTERVAL"):
        cfg.scheduler_interval = max(5.0, float(os.environ["AGENT_SCHEDULER_INTERVAL"]))
    if os.environ.get("AGENT_WATCH_ENABLED") is not None:
        cfg.watch_enabled = os.environ["AGENT_WATCH_ENABLED"].lower() in ("1", "true", "yes", "on")
    if os.environ.get("AGENT_WATCH_INTERVAL"):
        cfg.watch_interval = max(2.0, float(os.environ["AGENT_WATCH_INTERVAL"]))
    if os.environ.get("AGENT_WATCH_TASK") is not None:
        cfg.watch_task = os.environ["AGENT_WATCH_TASK"]
    if os.environ.get("AGENT_EVOLUTION") is not None:
        cfg.evolution_enabled = os.environ["AGENT_EVOLUTION"].lower() in ("1", "true", "yes", "on")
    if os.environ.get("AGENT_RECORD_RECIPES") is not None:
        cfg.record_recipes = os.environ["AGENT_RECORD_RECIPES"].lower() in ("1", "true", "yes", "on")
    if os.environ.get("AGENT_PLANNING") is not None:
        cfg.planning_enabled = os.environ["AGENT_PLANNING"].lower() in ("1", "true", "yes", "on")
    if os.environ.get("AGENT_PIPELINE") is not None:
        cfg.pipeline_enabled = os.environ["AGENT_PIPELINE"].lower() in ("1", "true", "yes", "on")
    if os.environ.get("AGENT_PROVIDER_RETRIES"):
        cfg.provider_retries = max(0, int(os.environ["AGENT_PROVIDER_RETRIES"]))
    if os.environ.get("AGENT_PROVIDER_FALLBACK") is not None:
        cfg.provider_fallback = os.environ["AGENT_PROVIDER_FALLBACK"].strip().lower()
    if os.environ.get("AGENT_TOOL_CACHE") is not None:
        cfg.tool_cache_enabled = os.environ["AGENT_TOOL_CACHE"].lower() in ("1", "true", "yes", "on")
    if os.environ.get("AGENT_TOOL_CACHE_TTL"):
        cfg.tool_cache_ttl = max(0.0, float(os.environ["AGENT_TOOL_CACHE_TTL"]))
    if os.environ.get("AGENT_TOOL_CHANNEL_CONCURRENCY"):
        cfg.tool_channel_concurrency = max(
            0, int(os.environ["AGENT_TOOL_CHANNEL_CONCURRENCY"])
        )
    cfg.skills_dir = Path(os.environ.get("AGENT_SKILLS_DIR", cfg.skills_dir))
    if os.environ.get("AGENT_CONTEXT_LIMIT"):
        cfg.context_limit = max(4, int(os.environ["AGENT_CONTEXT_LIMIT"]))
    if os.environ.get("AGENT_CONTEXT_MODEL_SUMMARY"):
        cfg.context_model_summary = (
            os.environ["AGENT_CONTEXT_MODEL_SUMMARY"].strip().lower()
            not in ("0", "false", "no", "off")
        )
    if os.environ.get("AGENT_MAX_CALLS"):
        cfg.max_calls = max(0, int(os.environ["AGENT_MAX_CALLS"]))
    if os.environ.get("AGENT_MAX_TOKENS"):
        cfg.max_tokens = max(0, int(os.environ["AGENT_MAX_TOKENS"]))
    if os.environ.get("AGENT_REPAIR_ROUNDS"):
        cfg.repair_rounds = max(0, int(os.environ["AGENT_REPAIR_ROUNDS"]))
    if os.environ.get("AGENT_TOOL_RETRIES"):
        cfg.tool_retries = max(0, int(os.environ["AGENT_TOOL_RETRIES"]))
    cfg.db_path = Path(os.environ.get("AGENT_DB_PATH", cfg.db_path))
    cfg.runs_dir = Path(os.environ.get("AGENT_RUNS_DIR", cfg.runs_dir))
    if not cfg.model:
        cfg.model = _default_model()
    cfg.secret_sources = _fill_secrets(cfg)

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
