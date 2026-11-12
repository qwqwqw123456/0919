"""宿主体系契约的机器可读抽取。

唯一依据：`assets/interface-manifest.md`（v1.62）。只登记清单里实际暴露的
节点、风险与批准要求，不凭记忆补写。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Risk(str, Enum):
    READ = "read"    # 只读
    LOCAL = "local"  # 本地计算 / 本地产物（落 output/，不外发）
    DOC = "doc"      # 飞书文档（对外低风险写，可回删）
    WRITE = "write"  # P 级写操作
    SEND = "send"    # S/P 级外发


class Approval(str, Enum):
    NONE = "none"                    # 宿主契约未要求人工批准
    HUMAN = "human"                  # 宿主契约标注「人工批准」
    HUMAN_PREAUTH = "human_preauth"  # 人工批准，但契约明确支持白名单预授权


@dataclass(frozen=True)
class Node:
    tool: str
    delegate: str          # data_agent / delivery_agent / external_runtime
    action: str
    risk: Risk
    approval: Approval
    produces: str = ""
    needs: tuple[str, ...] = ()
    note: str = ""


# 21 个规划器契约节点（dag_planner.TOOL_SCHEMAS）
NODES: dict[str, Node] = {
    # --- 本地子 Agent 执行链（13）---
    "search_web": Node("search_web", "data_agent", "search_web", Risk.READ, Approval.NONE, "results"),
    "read_many": Node("read_many", "data_agent", "read_many", Risk.READ, Approval.NONE, "summary", ("search_web",)),
    "memory_retrieve": Node("memory_retrieve", "data_agent", "memory_retrieve", Risk.READ, Approval.NONE, "summary"),
    "read_web": Node("read_web", "data_agent", "read_web", Risk.READ, Approval.NONE, "summary"),
    "format_text": Node("format_text", "data_agent", "format_text", Risk.LOCAL, Approval.NONE, "summary"),
    "select_best": Node("select_best", "data_agent", "select_best", Risk.LOCAL, Approval.NONE, "pick",
                        ("read_many", "search_web", "memory_retrieve")),
    "trace_check": Node("trace_check", "data_agent", "trace_check", Risk.LOCAL, Approval.NONE, "trace_report",
                        ("read_many", "search_web", "memory_retrieve")),
    "create_sheet": Node("create_sheet", "delivery_agent", "create_sheet", Risk.LOCAL, Approval.NONE, "artifact",
                         ("read_many", "memory_retrieve", "search_web", "select_best")),
    "create_ppt": Node("create_ppt", "delivery_agent", "create_ppt", Risk.LOCAL, Approval.NONE, "artifact",
                       ("read_many", "memory_retrieve", "search_web", "select_best")),
    "create_doc": Node("create_doc", "delivery_agent", "create_doc", Risk.DOC, Approval.NONE, "doc_url",
                       ("read_many", "memory_retrieve", "search_web", "select_best")),
    "create_calendar": Node("create_calendar", "delivery_agent", "create_calendar", Risk.WRITE, Approval.HUMAN, "event",
                            ("read_many", "memory_retrieve", "search_web", "select_best"),
                            "P 级写操作，人工批准"),
    "send_message": Node("send_message", "delivery_agent", "send_message", Risk.SEND, Approval.HUMAN, "receipt",
                         ("read_many", "memory_retrieve", "search_web", "select_best"),
                         "S 级主动外发，人工批准；目标渠道取 LARK_HITL_CHAT_ID"),
    "send_file": Node("send_file", "delivery_agent", "send_file", Risk.SEND, Approval.HUMAN_PREAUTH, "receipt",
                      ("read_many", "memory_retrieve", "search_web", "select_best"),
                      "P 级外发，人工批准；本人/白名单可预授权，名单外规则拦截"),

    # --- 宿主 MCP 只读连接器（8，delegate=external_runtime，grade=A，无前驱）---
    "finance_search": Node("finance_search", "external_runtime", "seed_finance_search", Risk.READ, Approval.NONE),
    "finance_lookup": Node("finance_lookup", "external_runtime", "seed_finance_lookup", Risk.READ, Approval.NONE),
    "legal_search": Node("legal_search", "external_runtime", "seed_legal_search", Risk.READ, Approval.NONE),
    "company_search": Node("company_search", "external_runtime", "datapro_search", Risk.READ, Approval.NONE),
    "map_transit": Node("map_transit", "external_runtime", "map_transit_query", Risk.READ, Approval.NONE),
    "map_route": Node("map_route", "external_runtime", "route_plan_v2", Risk.READ, Approval.NONE),
    "github_code": Node("github_code", "external_runtime", "github_oauth_search_code", Risk.READ, Approval.NONE),
    "github_pr": Node("github_pr", "external_runtime", "github_oauth_search_pull_requests", Risk.READ, Approval.NONE),
}

# 13 个已授权已装的主权 Skill（云端 TEE 运行，唯一业务入口 invoke 扣额度）
SOVEREIGN_SKILLS: tuple[str, ...] = (
    "pi-digit-nth",
    "e-digit-nth",
    "code-review",
    "sast-result-review",
    "interview-summary",
    "office-task-brief",
    "weekly-report-generator",
    "meeting-and-brief",
    "security-threat-review",
    "fatianshi-contract",
    "fatianshi-review-guarantee",
    "cookie-banner-inspector",
    "cookie-notice-link-finder",
)


def get_node(tool: str) -> Node | None:
    return NODES.get(tool)
