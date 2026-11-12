# 接口清单 interface-manifest（v1.62 · 2026-11-12）

> 本清单只登记体系当前实际暴露的调用形态，来源为源码与登记文件实读，不凭记忆补写。
>
> 来源文件：`scripts/dag_planner.py`（TOOL_SCHEMAS）、`scripts/subagents/data_agent.py`、
> `scripts/subagents/delivery_agent.py`、`assets/agents.yaml`、`assets/external-capabilities.md`、
> `assets/capability-mapping.yaml`、各主链路脚本 argparse。
>
> 实跑基线（`python3 scripts/capability-registry.py --json`）：本地脚本 67 · 实验 2 ·
> 规划器本地工具 21 · 宿主连接器 8 · 子 Agent 2；孤儿脚本 0、未接动作 0。

---

## 0. 调用面总览

| 层 | 形态 | 数量 | 说明 |
| --- | --- | --- | --- |
| A. 规划器契约节点 | tool → delegate → action | 21（本地 13 + 宿主连接器 8） | 规划层只能编出白名单内节点；幻觉工具被契约校验拒绝 |
| B. 子 Agent 执行动作 | do_run(ctx) 按 action 分发 | data 7 · delivery 6 | 本地 Python 进程内执行；入口统一过安全闸 |
| C. 宿主 MCP 连接器 | delegate=external_runtime，channel=host_mcp | 8（A 级只读） | 本地不持 MCP 客户端与凭证，不直连、不伪造；交宿主在线运行时按 action 调 |
| D. 主权 Skill CLI | sovereign-skill invoke ... | 13 个已授权已装 | 云端 TEE 运行，唯一业务入口 invoke 扣额度 |
| E. 本地主链路 CLI | python3 scripts/\<脚本\> \<子命令\> [参数] | 见第 4 节 | 本地规则层：规划 / 编排 / 决策 / 恢复 |

> 宿主目录实物 114 个 Skill（内置 95 + 自定义 19）走对话层激活器匹配，不在自主脚本主链路内；
> 主链路只认内置意图表 `INTENT_SKILLS` 与 `assets/capability-mapping.yaml` 授权映射表，
> 两表未命中落 `general_search` 兜底（常态，非故障）。

---

## 1. 规划器契约节点（`dag_planner.TOOL_SCHEMAS`）

### 1.1 本地子 Agent 执行链（13 个）

| 规划 tool | delegate | action | 产出 provides | 合法前驱 needs | 入参（ctx 键） | 出参 | 风险 / 批准 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| search_web | data_agent | search_web | results | 无 | query:str，top_n:int=3 | list[str]（TopN 链接） | 只读 |
| read_many | data_agent | read_many | summary | search_web | urls:list[str]，max_chars | str（多源汇总摘要） | 只读，单源失败不传染 |
| memory_retrieve | data_agent | memory_retrieve | summary | 无 | query:str，k:int=3 | str（本地记忆召回片段） | 只读 |
| read_web | data_agent | read_web | summary | 无 | url:str，max_chars | str（正文提纯，带注入扫描与不可信标记） | 只读 |
| format_text | data_agent | format_text | summary | 无 | text:str，max_chars | str（截断 + 注入特征扫描） | 本地，不联网不外发 |
| select_best | data_agent | select_best | pick | read_many / search_web / memory_retrieve | 上游候选 list[str] + criteria:str | str（选中一份纯文本）；附 artifact.pick / criteria / picked_from | 本地 |
| trace_check | data_agent | trace_check | trace_report | read_many / search_web / memory_retrieve | body（或 body_text），sources（或 source_texts:list） | dict(trace_report)：matched / unverified 两态 | 本地只读机械比对；不做单位换算与实体级冲突判定 |
| create_sheet | delivery_agent | create_sheet | artifact | read_many / memory_retrieve / search_web / select_best | title:str，header:list，rows:list，filename，content | path(.xlsx)，落 output/，不外发 | 本地产物 |
| create_ppt | delivery_agent | create_ppt | artifact | 同上 | title:str，slides:list[{heading,bullets}]，filename | path(.pptx)，落 output/，不外发 | 本地产物 |
| create_doc | delivery_agent | create_doc | doc_url | 同上 | title:str，content:str | dict（doc_id / doc_url） | 飞书文档 |
| send_message | delivery_agent | send_message | receipt | 同上 | text:str | dict(receipt)；summary 为文本前 80 字 | S 级主动外发，人工批准；目标渠道取 LARK_HITL_CHAT_ID |
| create_calendar | delivery_agent | create_calendar | event | 同上 | title:str，meeting_time:str，meeting_end:str，description:str | dict(event)；缺 end 自动补 30min，时区归一 +08:00 | P 级写操作，人工批准 |
| send_file | delivery_agent | send_file | receipt | 同上 | file_path:str，target_user_id:str，task_id，node_id，_approved_by_human:bool，approval_id，operator | dict(receipt)；产物须在 output/ 沙箱内 | P 级外发，人工批准；本人 / 白名单可预授权，名单外规则拦截 |

### 1.2 宿主 MCP 只读连接器（8 个，A 级，无前驱）

| 规划 tool | 传入运行时 action | 宿主工具 | 入参 | 出参 |
| --- | --- | --- | --- | --- |
| finance_search | seed_finance_search | seed_finance_search | query:str | list[record]（公告 / 研报 / 财报文本） |
| finance_lookup | seed_finance_lookup | seed_finance_lookup | query:str | list[record]（行情 / 估值 / 指标 / 汇率 / 利率 / 净值） |
| legal_search | seed_legal_search | seed_legal_search | query:str | list[record]（法律法规 / 司法来源） |
| company_search | datapro_search | datapro_search | query:str | list[record]（工商档案等专业数据集） |
| map_transit | map_transit_query | 地图 transit_query | query:str | list[route]（公交 / 轨道线路与站点） |
| map_route | route_plan_v2 | 地图 route_plan_v2 | query:str | list[route]（驾车 / 步 / 骑行 / 公交路线） |
| github_code | github_oauth_search_code | GitHub OAuth search_code | query:str | list[code_hit]（代码只读搜索） |
| github_pr | github_oauth_search_pull_requests | GitHub OAuth search_pull_requests | query:str | list[pr_hit]（PR 只读检索） |

> 通道口径：本地 Python 进程不持有 MCP 客户端与凭证，不直连、不伪造结果；规划层只识别意图
> 并编出合法节点，执行交宿主在线运行时按 action 调对应连接器。本地落跑 / dry-run 遇到此类
> 节点不得伪造成功结果。

---

## 2. 子 Agent 执行动作（源码 do_run 实际分支）

### 2.1 data_agent（`scripts/subagents/data_agent.py`）

上下文 `ctx: dict`，入口 `do_run(ctx)`；未知动作返回 `{"ok": False, "error": "未知动作 ...", "agent": ...}`。

| action | 读取的 ctx 键 | 返回要点 |
| --- | --- | --- |
| search_web | query，top_n=3 | 免费 HTML 搜索端点解析 TopN，逐条过公网校验剔内网 / 回环 / 内部域 |
| read_web | url，max_chars | 复用 safe-web-fetch：公网校验 / 抓取 / 清洗提纯 / 注入扫描 / 截断 |
| read_many | urls:list，max_chars | 批量循环 read_web，单源失败不传染，汇总多源摘要 |
| memory_retrieve | query，k=3 | 复用 memory-rag query_bag 本地召回 |
| format_text | text，max_chars | 纯文本截断 + 注入特征扫描 |
| select_best | 上游候选 + criteria | LLM 裁判 + 规则兜底 + 防幻觉清洗，出纯字符串 pick |
| trace_check | body / body_text，sources / source_texts | 复用 trace-check 纯函数，matched / unverified 两态 |

### 2.2 delivery_agent（`scripts/subagents/delivery_agent.py`）

入口先 `_resolve(ctx)` 把 `{summary}` / `{results}` 等占位符换成上游真实值；未知动作返回 `{"ok": False, "error": ...}`。

| action | 读取的 ctx 键 | 闸口与返回要点 |
| --- | --- | --- |
| send_message | text | 目标渠道取环境变量 LARK_HITL_CHAT_ID；dry-run 不真实外发；空文本拒发 |
| create_doc | title，content | 真实 lark-cli docs +create，返回 doc_id / doc_url；title 与 content 皆空拒发 |
| create_sheet | title，header，rows，filename，content | 本地 .xlsx，落 output/ |
| create_ppt | title，slides，filename | 本地 .pptx，落 output/ |
| create_calendar | title，meeting_time，meeting_end，description | 真实 lark-cli calendar +create；缺 end 自动补 30min，时区归一 +08:00 |
| send_file | file_path，target_user_id，task_id，node_id，_approved_by_human，approval_id，operator | 执行闸口细节未收到；以 1.1 契约为准：产物须在 output/ 沙箱内，P 级外发，人工批准，白名单可预授权 |

---

## 3. 主权 Skill CLI（云端 TEE 运行）

```
sovereign-skill invoke --skill <命令名> --task <任务> [--attachment <绝对路径>]
```

- 唯一业务入口是 `invoke`，只有 `invoke` 扣额度；额度用尽走 `quota-apply`，不绕过。
- 已授权已装 13 个命令：`pi-digit-nth`、`e-digit-nth`、`code-review`、`sast-result-review`、
  `interview-summary`、`office-task-brief`、`weekly-report-generator`、`meeting-and-brief`、
  `security-threat-review`、`fatianshi-contract`、`fatianshi-review-guarantee`、
  `cookie-banner-inspector`、`cookie-notice-link-finder`
  （分级与额度见 `external-capabilities.md` 第一节）。
- 任务 → Skill 静态映射在 `capability-mapping.yaml`，含 `level`、`capabilities`、
  `anti_capabilities`、`required_materials`、`routing_keywords`；表内未命中须诚实返回
  "无此能力授权"，不得臆造调用。

---

## 4. 本地主链路 CLI 入口形态

| 脚本 | 子命令 / 参数 |
| --- | --- |
| `work-task.py` | `run "<goal>" [--task T-xxx] [--append ...] [--dry-run] [--approved] [--host-only] [--json] [--trace] [--verbose]`；`show` / `resume` / `panel <task_id>` |
| `pipeline.py` | `run --spec <file> [--force]`；`validate` / `plan --spec`；`gen` / `auto --request`；`ack --task --step --spec --result`；`repair --task --step --spec [--cmd/--prompt/--reason]` |
| `dag_planner.py` | `plan --intent ...`；`spec --intent ...` |
| `resume_task.py` | `[--approval <单号>] [--dry-run] [--no-resume]` |

---

## 5. 备注口径

- 宿主目录实物 **114 个 Skill**（内置 95 + 自定义 19）走对话层激活器匹配，未接入自主脚本
  主链路；主链路只认 `INTENT_SKILLS` 内置意图表 + `capability-mapping.yaml` 授权映射表，
  两表未命中落 `general_search` 兜底——常态非故障。
- `external-capabilities.md` 还列了**未入规划器契约**的连接器：`general_search`、`web.fetch`、
  汽水、GitHub 其余只读（`list_commits` / `get_file_contents` / `search_issues` / `issue_read`）、
  EdgeOne `deploy_html`（D）、火山安全运营研判与止损类（D，禁探活空跑）。

---

## 附：子 Agent 注册登记口径差（`agents.yaml`）

| 子 Agent | 源码 `do_run` 实际分发 | `agents.yaml` `allowed_actions` 登记 |
| --- | --- | --- |
| data_agent | search_web、read_web、read_many、memory_retrieve、format_text、select_best、trace_check（7） | 仅前 5 个 |
| delivery_agent | send_message、create_doc、create_sheet、create_ppt、create_calendar、send_file（6） | 仅 send_message、create_doc、create_calendar |

后 5 个动作（select_best、trace_check、create_sheet、create_ppt、send_file）源码分支已接通
且在规划器契约里，但 `agents.yaml` 未同步登记——不是调用断层（可达性 0 断层），是注册描述
滞后，是否补登属待拍板项。

---

## 收录状态

- ✅ 第 0 节 调用面总览（interface-manifest 原文）
- ✅ 第 1 节 规划器契约节点 13 + 8（interface-manifest 原文）
- ✅ 第 2 节 子 Agent 执行动作（2.2 `send_file` 执行闸口细节未收到，以 1.1 契约为准）
- ✅ 第 3 节 主权 Skill CLI（更早实读核对转述）
- ✅ 第 4 节 本地主链路 CLI（更早实读核对转述）
- ✅ 第 5 节 备注口径（更早实读核对转述）
- ✅ 附：agents.yaml 登记口径差（更早实读核对转述）
