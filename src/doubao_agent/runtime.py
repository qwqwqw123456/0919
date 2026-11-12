from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config, MCPServerConfig
from .core.agent import Agent
from .core.audit import AuditLog
from .core.coordinator import Coordinator
from .core.knowledge import KnowledgeStore
from .core.memory import MemoryStore
from .core.permissions import Approver, PermissionPolicy
from .core.planner import Planner
from .core.recipe import RecipeStore
from .core.scheduler import ScheduleStore
from .core.session import SessionStore
from .core.usage import UsageMeter
from .providers import ModelProvider, build_provider
from .providers.embedding import BridgeEmbedding
from .registry import CapabilityRegistry
from .skills import SkillManager
from .sources import (
    ApiSource,
    HttpHostBridge,
    KnowledgeSource,
    MCPSource,
    NativeSource,
    ScriptSource,
    SkillSource,
)
from .sources.mcp import DEFAULT_HOST_MCP_URL, probe_http_mcp
from .types import Risk


@dataclass
class Runtime:
    config: Config
    registry: CapabilityRegistry
    provider: ModelProvider
    agent: Agent
    sessions: SessionStore
    memory: MemoryStore
    knowledge: KnowledgeStore
    recipes: RecipeStore
    skills: SkillManager
    usage: UsageMeter
    scheduler: ScheduleStore
    # 正在运行的会话 → 该次运行专属的 Agent（取消只作用于对应会话）
    _runs: dict[str, Agent] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    async def build(cls, config: Config, approver: Approver | None = None) -> "Runtime":
        registry = CapabilityRegistry()

        mcp_servers = list(config.mcp_servers)
        if config.mcp_auto_discover and not mcp_servers:
            if await probe_http_mcp():
                mcp_servers.append(
                    MCPServerConfig(name="host", transport="http", url=DEFAULT_HOST_MCP_URL)
                )
                print(f"[info] 已自动接入宿主 MCP：{DEFAULT_HOST_MCP_URL}")
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

        skills = SkillManager(config.skills_dir)
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

        await registry.refresh()

        policy = PermissionPolicy(config.permission_mode, approver, config.auto_approve)
        sessions = SessionStore(config.db_path)
        memory = MemoryStore(config.db_path)
        recipes = RecipeStore(config.db_path)
        scheduler = ScheduleStore(config.db_path)
        audit = AuditLog(config.runs_dir)
        provider = build_provider(config.model, bridge)
        usage = UsageMeter()
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
            usage=usage,
            context_limit=config.context_limit,
            cancel_event=cancel_event,
        )
        return cls(
            config=config,
            registry=registry,
            provider=provider,
            agent=agent,
            sessions=sessions,
            memory=memory,
            knowledge=knowledge,
            recipes=recipes,
            skills=skills,
            usage=usage,
            scheduler=scheduler,
        )

    async def aclose(self) -> None:
        await self.registry.aclose()
        await self.provider.aclose()
        self.sessions.close()
        self.memory.close()
        self.knowledge.close()
        self.recipes.close()
        self.scheduler.close()

    async def reload_skills(self) -> int:
        """重新扫描技能目录并热更新工具注册表（新增/删除技能无需重启）。"""
        count = self.skills.refresh()
        if count and not self.registry.has_source("skill"):
            self.registry.add_source(SkillSource(self.skills))
        await self.registry.refresh()
        return count

    def spawn_agent(self) -> Agent:
        """创建一个独立 Agent 实例用于并行协同。

        共享工具注册表、模型、权限策略、审计、记忆、配方与计量，仅取消信号独立，
        因此子任务并发执行互不干扰，且成本计入同一份计量。
        """
        return Agent(
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
            usage=self.usage,
            context_limit=self.config.context_limit,
            cancel_event=asyncio.Event(),
        )

    def running_sessions(self) -> list[str]:
        return sorted(self._runs)

    async def start_run(
        self,
        task: str,
        session_id: str | None = None,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        """每次运行使用独立 Agent，取消只影响该会话，不波及其他并发任务。"""
        session_id = session_id or self.sessions.create(task)
        agent = self.spawn_agent()
        self._runs[session_id] = agent
        try:
            return await agent.run(task, session_id, on_event=on_event)
        finally:
            self._runs.pop(session_id, None)

    def cancel_session(self, session_id: str) -> bool:
        """请求取消指定会话；没有正在运行的该会话时返回 False。"""
        agent = self._runs.get(session_id)
        if agent is None:
            return False
        agent.cancel()
        return True

    def make_coordinator(self) -> Coordinator:
        return Coordinator(self.spawn_agent, self.provider)
