from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config, MCPServerConfig
from .core.agent import Agent
from .core.audit import AuditLog
from .core.coordinator import Coordinator
from .core.journal import ToolJournal
from .core.knowledge import KnowledgeStore
from .core.evolution import EvolutionStore
from .core.limits import RateLimiter
from .core.longterm import LongTermMemory
from .core.memory import MemoryStore
from .core.permissions import Approver, PermissionPolicy
from .core.netpolicy import Policy as SpeedPolicy
from .core.outbound import Notifier, config_from_dict as outbound_config
from .core.planner import Planner
from .core.profile import ProfileStore
from .core.recipe import RecipeStore
from .core.scheduler import ScheduleStore
from .core.session import SessionStore
from .core.skilltree import TreeLedger
from .core.telemetry import Telemetry
from .core.usage import UsageMeter
from .core.working import WorkingMemory
from .providers import ModelProvider, build_provider
from .providers.embedding import BridgeEmbedding
from .registry import CapabilityRegistry
from .skills import SkillManager
from .core.pipeline import Pipeline
from .sources import (
    A2ASource,
    ApiSource,
    HttpHostBridge,
    KnowledgeSource,
    MCPSource,
    NativeSource,
    ScriptSource,
    SkillSource,
    SystemSource,
    ToolboxSource,
    VoiceSource,
)
from .sources.mcp import DEFAULT_HOST_MCP_URL, probe_http_mcp
from .types import Risk


VOICE_SYSTEM = (
    "你在电话里和用户说话：口语、短句，一次不超过 40 字；先接话再给结论；"
    "不用列表、不用术语、不加表情符号；没听清就问一句；不复述用户原话；"
    "需要查东西的事，答应下来并说回头告诉你，不在这通电话里长跑。"
)


@dataclass


class Runtime:
    config: Config
    registry: CapabilityRegistry
    provider: ModelProvider
    agent: Agent
    sessions: SessionStore
    memory: MemoryStore
    working: WorkingMemory
    longterm: LongTermMemory
    knowledge: KnowledgeStore
    recipes: RecipeStore
    skills: SkillManager
    usage: UsageMeter
    scheduler: ScheduleStore
    profile: ProfileStore
    notifier: Notifier
    journal: ToolJournal
    telemetry: Telemetry
    evolution: EvolutionStore | None = None
    voice: Any = None
    tree: TreeLedger | None = None
    # 正在运行的会话 → 该次运行专属的 Agent（取消只作用于对应会话）
    _runs: dict[str, Agent] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    async def build(cls, config: Config, approver: Approver | None = None) -> "Runtime":
        # 策略层：网络通信 / 模型等待的超时、重试、退避、轮询集中在一处，运行期可热改。
        perf_path = str(config.config_path) if str(config.config_path) else None
        perf = SpeedPolicy(perf_path)
        cache_ttl = config.tool_cache_ttl if config.tool_cache_enabled else 0.0
        registry = CapabilityRegistry(
            cache_ttl=cache_ttl,
            channel_concurrency=config.tool_channel_concurrency,
            rate_limiter=RateLimiter(config.tool_rate_limits),
        )

        mcp_servers = list(config.mcp_servers)
        if config.mcp_auto_discover and not mcp_servers:
            if await probe_http_mcp():
                mcp_servers.append(
                    MCPServerConfig(name="host", transport="http", url=DEFAULT_HOST_MCP_URL)
                )
                print(f"[info] 已自动接入宿主 MCP：{DEFAULT_HOST_MCP_URL}", file=sys.stderr)
            else:
                print(
                    f"[info] 宿主 MCP {DEFAULT_HOST_MCP_URL} 暂不可达，跳过（相关能力暂缺）",
                    file=sys.stderr,
                )
        for server in mcp_servers:
            registry.add_source(MCPSource(server))

        for api in config.api_sources:
            try:
                registry.add_source(ApiSource.from_file(api.name, api.spec_path))
            except Exception as exc:  # noqa: BLE001 - report, keep other channels alive
                print(f"[warn] api source '{api.name}' skipped: {exc}", file=sys.stderr)

        bridge: Any = None
        bridge_cfg = config.bridge or {}
        if bridge_cfg.get("enabled") and bridge_cfg.get("url"):
            bridge = HttpHostBridge(
                bridge_cfg["url"],
                bridge_cfg.get("token"),
                capabilities_path=bridge_cfg.get("capabilities_path", "/capabilities"),
                policy=perf,
            )
        # All native capabilities (browser, desktop, notifications, clipboard,
        # media, ...) come from one discovered manifest. No hardcoded list.
        registry.add_source(NativeSource(bridge, fallback=bridge_cfg.get("fallback", True)))

        try:
            script_risk = Risk(config.script_risk)
        except ValueError:
            script_risk = Risk.REVERSIBLE
        cancel_event = asyncio.Event()
        registry.add_source(
            ScriptSource(
                config.workspace,
                risk=script_risk,
                max_cpu_seconds=config.script_max_cpu,
                cancel_event=cancel_event,
            )
        )

        evolution = (
            EvolutionStore(
                config.db_path,
                stale_days=config.evolution_stale_days,
                archive_days=config.evolution_archive_days,
            )
            if config.evolution_enabled
            else None
        )
        skills = SkillManager(config.skills_dir, usage=evolution)
        skills.refresh()
        if skills.list():
            registry.add_source(SkillSource(skills))

        # 宿主 embedding：桥可用且暴露对应能力时才启用，否则保持关键词检索。
        aembed: Callable[[str], Any] | None = None
        embedding_cfg = dict(config.embedding or {})
        if bridge is not None and embedding_cfg.get("enabled", True):
            embedding = BridgeEmbedding(bridge, str(embedding_cfg.get("capability") or "embed"))
            if await embedding.available():
                aembed = embedding.embed
            else:
                print(
                    "[warn] 宿主未提供 embedding 能力，知识库退回关键词检索",
                    file=sys.stderr,
                )

        knowledge = KnowledgeStore(config.db_path, aembed=aembed)
        registry.add_source(KnowledgeSource(knowledge))
        registry.add_source(SystemSource(config.workspace))
        voice_src: VoiceSource | None = None
        if getattr(config, "voice_enabled", True):
            # 回复函数要用 Runtime 实例，而实例在 build 末尾才构造：先占位，构造后回填
            voice_src = VoiceSource(lambda text: None, dict(getattr(config, "voice_config", {}) or {}))
            registry.add_source(voice_src)
        if getattr(config, "toolbox_enabled", True):

            named = str(config.toolbox_catalog or "").strip()
            root = Path(__file__).resolve().parents[2]
            catalog = (Path(named).expanduser() if named else root / "toolbox.yaml")
            if not catalog.is_absolute():
                catalog = root / catalog
            if catalog.is_file():
                registry.add_source(ToolboxSource(catalog))
            else:
                print(f"[warn] 工具箱目录缺失：{catalog}，toolbox 通道跳过", file=sys.stderr)
        registry.add_source(
            A2ASource(allow=list(getattr(config, "a2a_allow", []) or []), token="")
        )

        await registry.refresh()

        policy = PermissionPolicy(config.permission_mode, approver, config.auto_approve)
        sessions = SessionStore(config.db_path)
        memory = MemoryStore(config.db_path)
        working = WorkingMemory(config.db_path)
        longterm = LongTermMemory(config.db_path)
        recipes = RecipeStore(config.db_path)
        scheduler = ScheduleStore(config.db_path)
        profile = ProfileStore(config.db_path)
        notifier = Notifier(outbound_config(config.outbound), config.db_path)
        audit = AuditLog(config.runs_dir)
        provider = build_provider(config.model, bridge, perf)
        # 备用 provider：默认不降级（不静默换模型）。显式 provider_fallback=mock 用内置 mock；
        # 配置了 model_fallback 块则装配备用真实 provider；两者皆无则 None。
        fallback = None
        if (config.provider_fallback or "").strip().lower() == "mock":
            from .providers.mock import MockProvider

            fallback = MockProvider(script=None)
        elif config.model_fallback:
            fallback = build_provider(config.model_fallback, bridge, perf)
        # 进程级累计计量：只统计不设预算；预算按单次 run 单独开表（见 _run_meter）。
        usage = UsageMeter()
        journal = ToolJournal(config.db_path)
        telemetry = Telemetry(config.db_path)
        agent = Agent(
            registry=registry,
            provider=provider,
            policy=policy,
            sessions=sessions,
            audit=audit,
            memory=memory,
            max_steps=config.max_steps,
            recipes=recipes,
            record_recipes=config.record_recipes,
            planner=Planner(provider, skills, usage) if config.planning_enabled else None,
            usage=cls._run_meter(config),
            total_usage=usage,
            context_limit=config.context_limit,
            cancel_event=cancel_event,
            working=working,
            repair_rounds=config.repair_rounds,
            fallback_provider=fallback,
            provider_retries=config.provider_retries,
            retry_delay=config.provider_retry_delay,
            perf=perf,
            model_summary=config.context_model_summary,
            journal=journal,
            telemetry=telemetry,
            tool_retries=config.tool_retries,
            skills=skills,
            sample_rate=config.calibration_sample_rate,
            pipeline_only_risk=config.pipeline_only_risk,
            guard_abort_seconds=config.guard_abort_seconds,
            guard_warn_tokens=config.guard_warn_tokens,
        )
        if config.pipeline_enabled:
            agent.pipeline = Pipeline(
                agent,
                max_atoms=config.pipeline_max_atoms,
                atom_steps=config.pipeline_atom_steps,
                repair_rounds=config.pipeline_repair_rounds,
                deadline_seconds=config.pipeline_deadline_seconds,
            )
        runtime = cls(
            config=config,
            registry=registry,
            provider=provider,
            agent=agent,
            sessions=sessions,
            memory=memory,
            working=working,
            longterm=longterm,
            knowledge=knowledge,
            recipes=recipes,
            skills=skills,
            usage=usage,
            scheduler=scheduler,
            profile=profile,
            notifier=notifier,
            journal=journal,
            telemetry=telemetry,
            evolution=evolution,
        )
        if voice_src is not None:
            voice_src.bind_reply(runtime.voice_reply)
            runtime.voice = voice_src
        runtime.tree = TreeLedger(Path(config.workspace) / "tree")
        try:
            # 图谱策略里的硬禁止直接灌进权限层：命中即拒，不进审批
            policy.deny = runtime.tree.deny_rules()
        except Exception as exc:  # noqa: BLE001 - 策略读不出来不影响启动
            print(f"[warn] 策略装载失败：{exc}", file=sys.stderr)
        return runtime

    async def voice_reply(self, text: str, history: list[dict[str, str]] | None = None) -> str:
        """语音通道的作答：直连模型单轮，不挂工具、不跑 agent 循环。

        电话要的是 1 秒内接话；每句都跑完整 agent 循环会拖到几十秒，
        需要查东西的活交给后台任务，不在通话里做。
        """
        msgs: list[dict[str, Any]] = [{"role": "system", "content": VOICE_SYSTEM}]
        msgs += [m for m in (history or []) if m.get("role") in ("user", "assistant")][-6:]
        if not msgs[1:] or msgs[-1].get("role") != "user":
            msgs.append({"role": "user", "content": text})
        try:
            out = await self.provider.complete(msgs, [])
        except Exception:  # noqa: BLE001 - 通话不因一次推理失败断掉
            return "我这边卡了一下，你再说一遍好吗？"
        return str(out.get("content") or "").strip()

    async def aclose(self) -> None:
        await self.registry.aclose()
        await self.provider.aclose()
        self.sessions.close()
        self.memory.close()
        self.working.close()
        self.longterm.close()
        self.knowledge.close()
        self.recipes.close()
        self.scheduler.close()
        self.profile.close()
        self.notifier.close()
        self.journal.close()
        self.telemetry.close()

    async def reload_skills(self) -> int:
        """重新扫描技能目录并热更新工具注册表（新增/删除技能无需重启）。"""
        count = self.skills.refresh()
        if count and not self.registry.has_source("skill"):
            self.registry.add_source(SkillSource(self.skills))
        await self.registry.refresh()
        return count

    def spawn_agent(self) -> Agent:
        """创建一个独立 Agent 实例用于并行协同。

        共享工具注册表、模型、权限策略、审计、记忆、配方与进程级计量，仅取消信号与
        本次运行的预算表独立，因此子任务并发执行互不干扰，成本计入进程级汇总。
        端点兜底配置（备用 provider / 重试 / 摘要）必须显式透传，否则配置里写的
        兜底策略在正常跑任务时不生效。
        """
        spawned = Agent(
            registry=self.registry,
            provider=self.provider,
            policy=self.agent.policy,
            sessions=self.sessions,
            audit=self.agent.audit,
            memory=self.memory,
            max_steps=self.config.max_steps,
            recipes=self.recipes,
            record_recipes=self.config.record_recipes,
            planner=self.agent.planner,
            usage=self._run_meter(self.config),
            total_usage=self.usage,
            context_limit=self.config.context_limit,
            cancel_event=asyncio.Event(),
            working=self.working,
            repair_rounds=self.config.repair_rounds,
            fallback_provider=self.agent.fallback_provider,
            provider_retries=self.agent.provider_retries,
            retry_delay=self.agent.retry_delay,
            model_summary=self.agent.model_summary,
            journal=self.journal,
            telemetry=self.telemetry,
            tool_retries=self.config.tool_retries,
            profile=self.profile,
            skills=self.skills,
            sample_rate=self.config.calibration_sample_rate,
        )
        if self.config.pipeline_enabled:
            spawned.pipeline = Pipeline(
                spawned,
                max_atoms=self.config.pipeline_max_atoms,
                atom_steps=self.config.pipeline_atom_steps,
                repair_rounds=self.config.pipeline_repair_rounds,
                deadline_seconds=self.config.pipeline_deadline_seconds,
            )
        return spawned

    @staticmethod
    def _run_meter(config: Config) -> UsageMeter:
        """单次运行的预算表：预算按每次 run 独立计，进程级汇总另记（见 self.usage）。"""
        return UsageMeter(max_calls=config.max_calls, max_tokens=config.max_tokens)

    def running_sessions(self) -> list[str]:
        return sorted(self._runs)

    async def start_run(
        self,
        task: str,
        session_id: str | None = None,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
        images: list[str] | None = None,
        guard_seconds: float | None = None,
    ) -> dict[str, Any]:
        """每次运行使用独立 Agent，取消只影响该会话，不波及其他并发任务。"""
        session_id = session_id or self.sessions.create(task)
        agent = self.spawn_agent()
        if guard_seconds:
            # 评测逐条上限交给 agent 自己收敛：外部 wait_for 取消会传染 MCP 会话
            agent.guard_abort_seconds = float(guard_seconds)
        self._runs[session_id] = agent
        try:
            return await agent.run(task, session_id, on_event=on_event, images=images)
        finally:
            self._runs.pop(session_id, None)
            self.ingest_tree(session_id)

    def ingest_tree(self, session_id: str) -> dict[str, Any] | None:
        """把一次运行的证据回写图谱；任何异常都不能影响任务本身。"""
        if self.tree is None or not self.tree.ready:
            return None
        try:
            info = self.sessions.get(session_id) or {}
            events = self.agent.audit.read(session_id)
            spans = self.telemetry.list(session_id)
            return self.tree.ingest(
                session_id,
                str(info.get("task") or ""),
                events,
                spans,
                status=str(info.get("status") or ""),
            )
        except Exception as exc:  # noqa: BLE001 - 记账失败不拖垮运行
            print(f"[warn] 技能树回写失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return None

    def cancel_session(self, session_id: str) -> bool:
        """请求取消指定会话；没有正在运行的该会话时返回 False。"""
        agent = self._runs.get(session_id)
        if agent is None:
            return False
        agent.cancel()
        return True

    def make_coordinator(self) -> Coordinator:
        return Coordinator(self.spawn_agent, self.provider)
