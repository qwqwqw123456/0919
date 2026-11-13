# 接口清单 interface-manifest（doubao-agent v0.2.0 · 2026-11-13）

> 本清单只登记当前源码实际暴露的调用形态，来源为 `src/doubao_agent/` 实读与本地实跑，不凭记忆补写。
>
> 实跑基线：`src/` 下 48 个 Python 文件 / 8371 行；测试 30 个文件、177 passed。注册表工具随通道装配：静态内置为 builtin 2（`script.run_python` / `script.run_shell`）+ system 2（`system.status` / `system.net_check`）+ knowledge 2（`knowledge.search` / `knowledge.add`）；`skill` 通道仅在技能目录扫描到至少一个技能时挂载；mcp / api / native / browser / desktop 按配置与宿主可达性条件触发（见第 3 节）。离线 golden 评估集见 `eval/golden/`，判定器 `eval/runner.py`。
> 源码根：`src/doubao_agent/`；入口脚本 `doubao-agent = doubao_agent.cli:main`。

---

## 1. 调用面总览

| 层 | 形态 | 说明 |
| --- | --- | --- |
| CLI | `doubao-agent <子命令> [参数]` | 本地命令行，覆盖运行、记忆、知识、调度、配方、并行、技能/MCP 管理、指标、服务与守护 |
| 控制台 HTTP | FastAPI 路由 `/api/*` + SSE `/api/events` | 本地 Web 控制台，工具/技能/知识/会话/审计/调度/并行/指标与追踪 |
| Agent 内核 | `Agent.run()` 规划-执行-观察-反思主循环 | 单次入口规划（可选）→ 工具执行 → 结果归一化 → 记忆/配方沉淀 |
| 能力注册表 | `CapabilityRegistry` 聚合多通道工具 | 上层只认 `ToolSpec`，不关心工具来自 MCP / API / 浏览器 / 桌面 / 原生 / 脚本 / 技能 / 知识 |
| 模型供给 | `host / openai / mock` 三档 Provider | 统一 `complete(messages, tools) -> {content, tool_calls, usage}` |
| 知识向量 | 可选宿主 embedding（`BridgeEmbedding`） | 桥清单暴露 `embed` 能力才启用；否则知识库保持关键词/混合检索，不伪造向量 |
| 持久化 | SQLite + JSONL 审计 | 三层记忆、会话与检查点、配方、调度、工具执行日志（`tool_runs`）与 span 追踪（`spans`）；断连可续跑 |
| 可靠性 | `ToolJournal` + 重试策略 | 幂等工具按参数指纹重放，只读/可逆写不重复执行；瞬时错误指数退避重试，受 `tool_retries` 控制 |

---

## 2. 通道（SourceKind，共 8 类）

注册表启动时对每个通道调用 `list_tools()` 发现工具；工具名含 `.` 保留原名，裸名自动加通道前缀（`<source>.<name>`）。

| 通道 | 源码 | 工具来源 | 默认风险 | 备注 |
| --- | --- | --- | --- | --- |
| `builtin` 脚本 | `sources/script.py` | 代码即动作：Python / bash 非交互执行 | 可配（默认 L1） | `run_python` / `run_shell`；CPU 上限可选 RLIMIT_CPU；支持会话级取消 |
| `skill` 技能 | `sources/skill.py` | `skills/<name>/SKILL.md`；可执行技能注册为工具，知识技能按需阅读 | 技能 frontmatter 声明，缺省 L1 | 入口限定在技能目录内；参数以 JSON 从 stdin 传入 |
| `knowledge` 知识 | `sources/knowledge.py` | 长文档分块检索与回写 | search L0 / add L1 | 与短期记忆分工：知识记长文档，记忆记短事实/偏好/经验；文档可归属七个知识域之一 |
| `system` 系统 | `sources/system.py` | 只读运行环境指标（CPU/内存/磁盘/进程 PID·线程·RSS）与 TCP 连通探测 | status / net_check 均 L0 | 标准库采集，不依赖 shell 与宿主桥；供规划判断资源可行性与网络可达性 |
| `native` 原生 | `sources/native.py` | 宿主 `/capabilities` 清单动态发现 | 清单声明，否则按名称关键词推断 | 桥不可用时返回空；桥存在但清单为空且 `fallback=true` 时挂浏览器/桌面兜底目录 |
| `browser` 浏览器 | `sources/browser.py` | 兜底工具目录（9 个） | 只读 L0 / 可逆 L1 | 仅在原生通道走兜底目录时注册；实际执行交宿主桥 |
| `desktop` 桌面 | `sources/desktop.py` | 兜底工具目录（8 个） | 只读 L0 / 可逆 L1 | 同上 |
| `mcp` | `sources/mcp.py` | MCP stdio / SSE / Streamable HTTP，工具自描述 | 按名称关键词推断 | 工具由配置的 MCP Server 启动时发现，代码不写死清单；配置来源为 `agent.yaml` 的 `mcp_servers` 与运行时 `mcp add` 写入的 `mcp.local.yaml`（同名以 `agent.yaml` 为准）。未配置且 `mcp_auto_discover`（默认 true）时，启动探测 `http://127.0.0.1:8091/mcp`，JSON-RPC `initialize` 拿到 `result` 才接入；端口不通先做 TCP 预检并缓存失败，后续调用直接报不可用，不拖崩进程 |
| `api` | `sources/http_api.py` | OpenAPI 文档或声明式 JSON → 动态生成工具 | 文档声明，否则按名称关键词推断 | 工具由配置的 API 规格启动时发现 |

---

## 3. 静态内置工具（本地实跑基线）

不依赖外部配置、注册后稳定存在的工具：

| 工具名 | 通道 | 风险 | 入参要点 | 说明 |
| --- | --- | --- | --- | --- |
| `script.run_python` | builtin | L1 | `code:str`，可选 `cwd`、`timeout` | 执行 Python 代码，print 输出关键结果 |
| `script.run_shell` | builtin | L1 | `command:str`，可选 `cwd`、`timeout` | 执行非交互 bash 命令 |
| `knowledge.search` | knowledge | L0 | `query:str`，可选 `limit:int=4`、`domain:str` | 通道内走异步 `asearch`；`domain=auto` 时按关键词自动路由到七域之一（`route_domain`），留空全库；有 `embed`/`aembed` 时切余弦向量，异步 embedding 失败自动退回关键词 + 字符 bigram 混合检索；分块向量按文本 sha1 缓存 |
| `knowledge.add` | knowledge | L1 | `title:str`、`text:str`，可选 `source:str` | 写入知识库并自动分块；兼容 `content`/`body`、`name`/`heading` 别名，缺字段时错误信息回显实际收到的字段；域归属留空，由 CLI/API 显式指定 |
| `system.status` | system | L0 | 可选 `path:str`（磁盘用量目录，默认工作区） | 返回 CPU 核数、内存总量/可用、本进程 PID/线程数/RSS、磁盘用量；结构化指标在 `meta.metrics`；非 Linux 内存/线程项标注不可读 |
| `system.net_check` | system | L0 | `host:str`、`port:int`，可选 `timeout:number=3.0` | 异步 TCP 连通探测，建连后立即关闭、不发业务数据；可达性与原因在 `meta.probe`，不可达返回原因不报错 |
| `skill.read` | skill | L0 | `name:str` | 阅读指定技能完整说明与资源清单（渐进式加载） |
| `skill.text-stats` | skill | L0 | 见技能 `input_schema` | 示例可执行技能，入口 `skills/text-stats/main.py`，统计字符/中文/词/行/高频词 |
| `skill.pre-write-check` | skill | L0 | `text:str` | 写操作前 D/S/P/A 风险扫描（删除/发送/付费/权限迁移），切句 + 否定紧邻豁免；命中输出 `VERDICT=block`，shell 入口 `skills/pre-write-check/check.sh`，退出码恒 0、结果看文本 |
| `skill.task-triage` | skill | L0 | `task:str` | 任务轻重分级 L1/L2/L3 与步骤预算（≤5/≤12/≤20）、耗时与 token 档、判定信号；python 入口 `skills/task-triage/main.py` |
| `skill.safety-checklist` | skill | 知识型 | — | 无入口，不注册为工具；高风险写操作前置清单与留痕协议，经 `skill.read` 阅读 |

### 3.1 原生兜底工具目录（条件触发）

仅当宿主桥存在、清单为空且 `fallback=true` 时注册；桥不可用时不挂。

- 浏览器 9 个：`browser.open`、`browser.snapshot`、`browser.screenshot`、`browser.click`、`browser.type`、`browser.upload`、`browser.tabs`、`browser.extract`、`browser.back`
- 桌面 8 个：`desktop.screenshot`、`desktop.screen_size`、`desktop.windows`、`desktop.focus`、`desktop.move`、`desktop.click`、`desktop.type_text`、`desktop.hotkey`

### 3.2 动态工具（不在代码内枚举）

- MCP 工具：由 `agent.yaml` 的 `mcp_servers` 启动对应 Server 后发现；未配置时启动自动探测宿主标准 MCP `127.0.0.1:8091/mcp`（见第 2 节），探测不到则跳过。
- API 工具：由 `api_sources` 指向的 OpenAPI / 声明式规格生成。
- 可执行技能：扫描 `skills_dir/*/SKILL.md`，有 `entry` 的注册为 `skill.<name>`。

---

## 4. CLI 命令（`cli.py`，19 个子命令）

全局选项：`--config <路径>`；配置路径依次取 `--config` → 环境变量 `DOUBAO_AGENT_CONFIG` → `<workspace>/agent.yaml`，不存在时回退 legacy `/workspace/agent.yaml`。`workspace` 默认取 `AGENT_WORKSPACE` 或当前目录。

| 子命令 | 位置参数 / 选项 | 作用 |
| --- | --- | --- |
| `tools` | — | 列出注册表全部工具（风险 / 通道 / 名称 / 描述） |
| `sources` | — | 检查各接入通道健康状态 |
| `skills` | `[list\|reload\|install\|uninstall]`；`target?` | 列出技能；`reload` 热扫描并刷新注册表；`install <目录>` 校验 `SKILL.md` 后复制进 `skills_dir`（同名拒绝覆盖）；`uninstall <名称>` 仅删技能根目录内的目录，越界拒绝 |
| `mcp` | `[list\|add\|remove]`；`name?`；`--url`；`--transport`；`--command`；`--args` | `list` 打印生效的 MCP（含启动时合并的本地登记）；`add/remove` 只写 `<db_path 所在目录>/mcp.local.yaml`，下次启动合并进 `mcp_servers`，不改 `agent.yaml`；stdio 用 `--command/--args`，http/sse 用 `--url/--transport`（默认 http） |
| `mcp-serve` | `--expose read\|all`（默认 `read`）；`--transport stdio\|http`（默认 `stdio`）；`--host`（默认 `127.0.0.1`）；`--port`（默认 `8766`） | 把注册表工具作为标准 MCP server 对外暴露，供外部 MCP client 挂载。`stdio` 走子进程标准输入输出；`http` 为 Streamable HTTP，端点 `/mcp`，用 URL 直挂（如 `http://127.0.0.1:8766/mcp`），默认仅绑本机回环。`read` 仅暴露 L0 只读工具（默认；外部调用绕过本地审批链，不可逆动作不对外）；`all` 放开全部风险等级，仅限本机受控场景。调用走注册表 `call`，复用限流 / 缓存 / 并发 |
| `metrics` | `--session <id>?`；`--detail` | 输出 span 汇总：总数、失败数、总耗时、平均耗时、按类型/工具的计数与最慢 5 条；省略 `--session` 为全局视图；`--detail` 按顺序列出该会话全部 span |
| `sessions` | — | 列出最近会话 |
| `resume` | `session_id` | 从检查点恢复会话 |
| `run` | `task`；`--session <id>`；`--no-record`；`--image <路径>`（可重复） | 执行一个任务；`--no-record` 不录制配方；`--image` 把本地图片编码为 OpenAI 多模态 `image_url`（单张 ≤5MB，需模型支持视觉） |
| `recipes` | — | 列出已录制配方 |
| `replay` | `recipe_id:int` | 按配方零模型重放 |
| `memory` | `action`；`value?`；`--kind`；`--limit=20`；`--days=30` | `list / add / search / forget / stats / decay`；`decay` 按半衰期衰减非偏好记忆，`--days` 指定半衰期；`--kind` 见第 9 节四类 |
| `schedule` | `action`；`args*`；`--name` | `add / list / remove / enable / disable / run`；cron 5 段 |
| `knowledge` | `action`；`value?`；`--title`；`--text`；`--domain`；`--limit=4` | `list / add / search / remove / stats / domains`；`domains` 输出七域文档数 |
| `state` | `[show\|set]`；`key?`；`value?` | 查看 / 修改第一层会话状态（工作记忆），字段见 9.1 |
| `longterm` | `kind`；`[add\|list]`；`--date` 及各字段选项 | `lesson / observation / deviation` 三表增查，`stats` 输出计数 |
| `audit` | `[verify\|sessions\|export]`；`session?`；`--out` | `verify` 重算哈希链；`sessions` 列出有审计的会话；`export` 校验通过后导出副本（默认 `runs/<session>/audit.export.jsonl`），链损坏拒绝导出 |
| `evolve` | `action`；`target?`；`--apply`；`--top=10`；`--force` | `status` 用量排行 / `audit` 生命周期流转（默认预演，`--apply` 才执行）/ `pin`·`unpin` 钉住 / `distill` 配方转技能草案（`--force` 覆盖已有草案）/ `report` 周报 |
| `parallel` | `task`；`--subtasks a,b,c`；`--review` | 并行协同；省略子任务则先由模型拆解；`--review` 追加审阅者角色，合并结果并指出冲突 |
| `serve` | `--port` | 启动本地控制台（uvicorn） |
| `daemon` | `action`；`--port` | `start / stop / status`，子进程崩溃按退避重启 |

---

## 5. 控制台 HTTP API（`server/app.py`）

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | `/api/health` | 运行态：注册表快照、通道健康、用量计量 |
| GET | `/api/tools` | 全部工具清单 |
| GET | `/api/skills` | 技能清单（含可执行标记、入口、风险） |
| GET | `/api/knowledge` | 知识库文档列表 |
| POST | `/api/knowledge` | 写入知识（`title`、`text`、可选 `source`） |
| GET | `/api/knowledge/search` | 检索知识（`q`、`limit`） |
| DELETE | `/api/knowledge/{doc_id}` | 删除文档及其分块 |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/sessions/{session_id}` | 会话详情：消息 + 产物 |
| GET | `/api/sessions/{session_id}/audit` | 该会话审计链 |
| POST | `/api/sessions/{session_id}/resume` | 后台恢复会话 |
| POST | `/api/sessions/{session_id}/cancel` | 请求取消指定会话 |
| POST | `/api/run` | 后台执行任务（`task`） |
| POST | `/api/webhook` | 入站事件触发（`task`，可选 `event`/`payload`）；令牌或 `X-Agent-Signature: sha256=<HMAC-SHA256>` 验签 |
| POST | `/api/parallel` | 后台并行执行（`task`，可选 `subtasks`、`review`） |
| GET | `/api/metrics` | span 汇总；`?session=<id>` 只看单会话，省略为全局 |
| GET | `/api/sessions/{session_id}/trace` | 该会话的 span 明细（名称、类型、耗时、结果、属性） |
| GET | `/api/events` | SSE 事件流（见第 6 节） |
| GET | `/api/schedules` | 定时任务列表 |
| POST | `/api/schedules` | 新增定时任务（`name`、`cron`、`task`） |
| DELETE | `/api/schedules/{schedule_id}` | 删除定时任务 |
| POST | `/api/schedules/{schedule_id}/toggle` | 启用 / 停用（`enabled`） |
| POST | `/api/schedules/{schedule_id}/run` | 立即执行一次 |
| GET | `/` | 控制台静态页 `server/static/index.html` |

鉴权：配置了 `auth_token`（或 `AGENT_AUTH_TOKEN`）时，中间件保护全部 `/api/*`；令牌可来自 `X-Agent-Token` 头、`Authorization: Bearer`、`?token=` 查询参数或 cookie。带 `?token=` 访问一次会种下 HttpOnly cookie。未配置令牌时不鉴权（仅限本机使用）。`POST /api/webhook` 另支持 `X-Agent-Signature: sha256=<raw body 的 HMAC-SHA256 摘要>` 验签，带签名头的请求由端点校验、不经令牌中间件拦截。

调度器：`_run_scheduler` 每轮扫描到期任务，为每个任务建独立 `asyncio.Task` 并发派发（`asyncio.Semaphore(3)` 限流）；`inflight` 集合去重，同一任务未结束不会被重复触发；执行结束才 `mark_ran`，服务关闭时取消所有未完成子任务。单个任务异常不影响调度循环。

文件监听：`watch_enabled` 开启后 `_run_watcher` 按 `watch_interval` 轮询 `watch_dirs`（空=工作区）文件 mtime 快照，首批只建基线不上报；新增/修改/删除广播 `file_change` 事件，同路径同类事件在一个轮询窗口内防抖合并；配置了 `watch_task` 模板（占位符 `{path}`/`{event}`）才触发一次 `start_run`，否则只广播不起任务；`watch_suffixes` 可按后缀过滤。默认关闭。

限流：`CapabilityRegistry` 可挂 `RateLimiter`（滑动 60 秒窗口，键为通道名，`default` 兜底）；超限调用直接返回 `rate limited: <source> 超过 N 次/分钟`，不触达底层通道。限额来自 `tool_rate_limits`。

---

## 6. SSE 事件类型（`/api/events`）

| 事件 | 载荷要点 | 触发处 |
| --- | --- | --- |
| `start` | `session`、`task`、可选 `mode` | 任务/规划开始 |
| `tool_start` | `tool` | 工具调用前 |
| `tool_end` | `tool`、`ok`、`error` | 工具返回后 |
| `subtasks` | `task`、`subtasks`、`count` | 并行拆解结果 |
| `review` | `task`、`verdict`、`conflicts` | 审阅者给出合并结论（`--review` / `review=true`） |
| `finish` | `session`、`answer`、可选 `subtasks` | 任务收尾 |
| `schedule_start` | `id`、`name` | 定时任务派发 |
| `file_change` | `event`（created/modified/deleted）、`path` | 文件监听捕获变更 |

---

## 7. 核心数据结构（`types.py`）

| 结构 | 字段 |
| --- | --- |
| `ToolSpec` | `name`、`description`、`input_schema`、`kind`、`risk`、`tags`、`idempotent`、`source` |
| `ToolCall` | `name`、`arguments`、`call_id`、`cancel_event`（本次运行独立取消信号） |
| `ToolResult` | `call_id`、`ok`、`content`、`error`、`elapsed_ms`、`meta` |
| `Message` | `role`（system / user / assistant / tool）、`content`、`tool_calls`、`tool_call_id`、`name` |

风险分级 `Risk`：`L0` 读 / `L1` 可逆写 / `L2` 不可逆 / `L3` 高危。

---

## 8. 权限模型（`core/permissions.py`）

按可逆性分级，不按白名单限制：

| 模式 | 行为 |
| --- | --- |
| `auto` | 除 L3 外全自动 |
| `guarded` | L2 起需确认 |
| `manual` | 除只读外都需确认 |

- 未配置审批人时，需确认的操作一律拒绝。
- 预授权规则 `PolicyRule`：按工具名（支持通配符）+ 风险上限 + 参数全等约束命中即放行，替代无人值守下的人工审批；配置在 `auto_approve`。
- 非法权限模式回退 `guarded`；非法风险等级回退关键词推断。

---

## 9. 持久化存储（SQLite，16 张表）

| 表 | 所属模块 | 用途 |
| --- | --- | --- |
| `facts` | `core/memory.py` | 长期记忆四类：`preference` / `fact` / `experience` / `procedure`；按指纹去重加权 |
| `working_state` | `core/working.py` | 第一层会话状态（单条例行 + 写入锁），字段见 9.1 |
| `lessons` | `core/longterm.py` | 第二层萃取库：日期、事件、判断、结果、教训 |
| `observations` | `core/longterm.py` | 第二层观察记录：日期、现象、观察、原因分析、后续动作 |
| `deviations` | `core/longterm.py` | 第二层偏差档案：日期、预期、实际、偏差、原因、修正动作 |
| `sessions` | `core/session.py` | 会话：任务、状态、结果、步数、时间戳 |
| `messages` | `core/session.py` | 会话消息流水（JSON 载荷） |
| `checkpoints` | `core/session.py` | 检查点：步骤与状态，断点续跑 |
| `artifacts` | `core/session.py` | 会话产物路径登记 |
| `documents` | `core/knowledge.py` | 知识库文档（标题、来源、`domain` 知识域、时间）；老库自动补列 |
| `chunks` | `core/knowledge.py` | 文档分块（默认目标 500 字/块） |
| `recipes` | `core/recipe.py` | 成功工具序列，可零模型重放 |
| `schedules` | `core/scheduler.py` | cron 定时任务：表达式、任务、启用态、上次/下次运行 |
| `skill_usage` | `core/evolution.py` | 技能用量台账：`use`/`view`/`patch` 计数、最后使用时间、生命周期状态（`active`/`idle`/`archived`）、`pinned` 保护位；14 天闲置、30 天归档，复用已归档技能自动复活 |
| `tool_runs` | `core/journal.py` | 工具执行日志：会话、步号、工具、参数 sha1、状态、尝试次数、耗时、结果预览 |
| `spans` | `core/telemetry.py` | span 追踪：模型调用 / 工具调用的耗时、成败与属性，用于 `metrics` 聚合 |

审计为追加式哈希链 JSONL：`<runs_dir>/<session_id>/audit.jsonl`；写入前对 `token` / `password` / `secret` / `cookie` / `api_key` / `private_key` 及 `*_token`、`*_secret` 等字段递归脱敏为 `***`（`max_tokens` 等非凭据字段不误伤）。长期记忆落库前另对 Bearer 令牌、`api_key=/secret=/token=` 赋值、JWT、`ak/sk/AKLT` 前缀等明显凭据模式打码；凭据本身只从环境变量/`.env.local` 注入，运行时不做加密 vault（归平台密钥保管）。`audit verify` 重算整链，被篡改返回退出码 1；`audit export` 先校验再复制，链损坏拒绝导出。

### 9.1 三层记忆体系

| 层 | 载体 | 生命周期 | 读法 |
| --- | --- | --- | --- |
| 第一层 会话状态（工作记忆） | `working_state` 单条例行 | 每次会话开头读、结束时回写 | `Agent.run()` 入口调 `begin_session()`，状态渲染进系统提示，先读再开口；结束调 `end_session()` |
| 第二层 长期经验 | `lessons` / `observations` / `deviations` 三表 | 跨会话累积 | `LongTermMemory` 增查；CLI `longterm`；偏差值可自动计算 |
| 第三层 知识域 | `documents.domain` + `chunks` | 长期 | 七域固定编号 `01~07`：认知与思维 / 自我成长 / 财富与商业 / 人际与关系 / 行业与赛道 / 工具与参考 / 国家政策；检索可按域过滤，`domain_stats()` 出域索引 |

- 工作记忆字段（16 项）：知识库、远程调用、我的名字、用户称呼、记忆点、上次会话日期、新点主题、离线时间线、工作流节点、初始化阶段、待回顾、暂停状态、写入锁、会话计数、时间戳、焦点线。
- 写入锁：`locked()` 上下文置 `locked`，期间其它 `update()` 抛 `RuntimeError`，退出自动解锁；未知字段抛 `ValueError`。
- 会话计数每次 `begin_session()` +1；`end_session(status, summary)` 写工作流节点、初始化阶段、待回顾（completed 时取回答摘要）或暂停状态（cancelled / paused）。
- 记忆闭环：`_system_prompt_for()` 记录本次注入的记忆 id；任务结束按成败回写 `feedback()`（成功 +0.2，失败 -0.15，夹在 `[0.1, 5.0]`）；`decay()` 按未使用时长做半衰期衰减，偏好不衰减。
- 知识闭环：`route_domain(query)` 按关键词把查询路由到七域之一，`knowledge.search(domain="auto")` 自动收窄检索范围；判断不出时退回全库，不强行归类。

---

## 10. 模型供给（`providers/`）

| Provider | 选择条件 | 说明 |
| --- | --- | --- |
| `host` | `model.provider=host`（默认有桥时） | 委托宿主能力 `model.complete`，无需 API key；桥缺失时退回 `mock` |
| `openai` | `model.provider=openai` | OpenAI 兼容 `/chat/completions`；`base_url` 必须显式配置，缺失时告警并退回 `mock`（不默认连 `api.openai.com`，避免误产生费用）；key 取 `api_key` 或 `api_key_env`（默认 `OPENAI_API_KEY`）；支持 `tool_calls` 与多模态 `image_url` |
| `mock` | 未识别或兜底 | 确定性脚本回复，供测试与离线演示 |

统一返回 `{content, tool_calls, usage}`；空 `choices` 或缺 `message` 回退空结果，不崩溃。主端点重试耗尽后，按 `provider_fallback=mock` 或 `model_fallback` 配置切备用通道，结果带 `degraded` / `degraded_provider` 标注；默认不降级。

### 10.1 知识向量供给（`providers/embedding.py`）

- `BridgeEmbedding(bridge, capability="embed")`：桥非空且 `embedding.enabled`（默认 true）时装配；`available()` 查桥能力清单是否含该能力名（结果缓存），不含则不启用。
- `embed(text)` 调桥 `invoke(capability, {"input": text})`；`_vectors_of` 兼容 `[float]` / `{"embedding"}` / `{"vectors":[[...]]}` / `{"data":[{"embedding"}]}` / 嵌套 `result`；解析不出抛 `ValueError`，不本地伪造向量。
- 不可用或向量解析失败时，知识库退回关键词/混合检索；向量函数为异步 `aembed`，供 `KnowledgeStore.asearch` 调用。

> 当前云电脑现状：宿主未暴露 `model.complete` 桥与 `embed` 能力，桥保持关闭；模型走 OpenAI 兼容接口（`agent.yaml` 显式配置 `base_url` 与 `model`，key 由 `.env.local` 的 `OPENAI_API_KEY` 注入，该文件不入库），知识库仍走混合检索。平台开放桥后由 `AGENT_BRIDGE_URL` / `AGENT_BRIDGE_TOKEN` 注入启用。

---

## 11. 配置项与环境变量

配置文件（`agent.yaml`，模板 `agent.example.yaml`）字段：

`workspace`、`runs_dir`、`db_path`、`permission_mode`、`script_risk`、`script_max_cpu`、`record_recipes`、`planning.enabled`、`skills_dir`、`context_limit`、`context_model_summary`、`scheduler_enabled`、`scheduler_interval`、`watch_enabled`、`watch_dirs`、`watch_interval`、`watch_task`、`watch_suffixes`、`max_steps`、`max_calls`、`max_tokens`、`repair_rounds`、`provider_retries`、`provider_fallback`、`tool_cache_enabled`、`tool_cache_ttl`、`tool_channel_concurrency`、`tool_retries`、`tool_rate_limits`、`port`、`auth_token`、`model`、`model_fallback`、`bridge`、`embedding`（`enabled` / `capability`）、`evolution`（`enabled` / `stale_days` / `archive_days`）、`auto_approve`、`mcp_servers`、`mcp_auto_discover`、`api_sources`。

- `max_calls` / `max_tokens`：模型调用数与 token 预算，`0` 表示不限；当前 UsageMeter 进程级累计、跨 run 不复位（非单次运行预算，待修见 REVIEW.md P0-6）；超限时任务以 `incomplete` 收尾并直接说明预算耗尽。
- `repair_rounds`：工具失败后允许模型修正参数重试的轮数，`0` 表示不修正。
- `provider_retries`：模型端点异常重试次数；设计上 401/403/404 不重试，但当前状态码提取未覆盖 `httpx.HTTPStatusError.response.status_code`，401/403 会被误判为可重试（待修见 REVIEW.md P0-5）；`provider_fallback` 为 `"mock"` 时主端点失败切内置 mock 兜底；配置 `model_fallback` 块（结构同 `model`）时装配备用真实 provider（备用配置当前未透传到 `spawn_agent` 主路径，待修见 REVIEW.md P0-3）；降级标注尚未透出到最终结果（待修见 REVIEW.md P0-4），两者皆留空则不降级、异常上抛。
- `tool_cache_enabled` / `tool_cache_ttl`：幂等且只读（L0）工具的结果进程内缓存开关与存活秒数；写工具、非幂等工具不缓存；配方 replay 强制绕过缓存真实重放。
- `tool_channel_concurrency`：单通道工具并发上限（默认 3），registry 按通道信号量排队；`0` 表示不限制。
- `context_model_summary`：上下文压缩时是否调模型浓缩被裁早期消息（默认关）；开启后额外产生模型调用，摘要失败回落规则要点。
- `evolution`：技能用量遥测与生命周期审计；`stale_days`（默认 14）未用算闲置，`archive_days`（默认 30）进归档，只改状态位不删文件；`pinned` 技能不参与自动流转。
- `tool_retries`：同一幂等工具遇到瞬时错误（超时 / 连接失败 / 不可达）时的重试次数，`0` 表示不重试；指数退避上限 2 秒。
- `tool_rate_limits`：按通道的每分钟调用上限，形如 `{"browser": 30, "default": 120}`；未配置的通道不限流。
- `mcp_servers` 会与 `<db_path 所在目录>/mcp.local.yaml`（`mcp add/remove` 写入）合并，同名以 `agent.yaml` 为准。

环境变量覆盖（前缀 `AGENT_*`）：

`AGENT_WORKSPACE`、`AGENT_RUNS_DIR`、`AGENT_DB_PATH`、`AGENT_SKILLS_DIR`、`AGENT_PORT`、`AGENT_AUTH_TOKEN`、`AGENT_PERMISSION_MODE`、`AGENT_SCRIPT_RISK`、`AGENT_SCRIPT_MAX_CPU`、`AGENT_SCHEDULER_ENABLED`、`AGENT_SCHEDULER_INTERVAL`、`AGENT_WATCH_ENABLED`、`AGENT_WATCH_INTERVAL`、`AGENT_WATCH_TASK`、`AGENT_RECORD_RECIPES`、`AGENT_PLANNING`、`AGENT_CONTEXT_LIMIT`、`AGENT_CONTEXT_MODEL_SUMMARY`、`AGENT_MAX_CALLS`、`AGENT_MAX_TOKENS`、`AGENT_REPAIR_ROUNDS`、`AGENT_PROVIDER_RETRIES`、`AGENT_PROVIDER_FALLBACK`、`AGENT_TOOL_CACHE`、`AGENT_TOOL_CACHE_TTL`、`AGENT_TOOL_CHANNEL_CONCURRENCY`、`AGENT_TOOL_RETRIES`、`AGENT_EVOLUTION`、`AGENT_BRIDGE_URL`、`AGENT_BRIDGE_TOKEN`、`AGENT_MCP_AUTO_DISCOVER`、`AGENT_EMBEDDING_CAPABILITY`；配置路径 `DOUBAO_AGENT_CONFIG`；OpenAI key 环境变量名由 `model.api_key_env` 指定（默认 `OPENAI_API_KEY`）。

- `.env.local` 自动加载：`load_config()` 启动时读取当前目录与工作区的 `.env.local`，解析 `KEY=VALUE` 行（兼容剥离 `export ` 前缀，注释/空行跳过）；只补缺省变量，已存在的环境变量优先不覆盖；文件缺失或读取失败静默跳过，不引入 `python-dotenv` 依赖。

---

## 12. 备注口径

- **动态发现，不写死清单**：MCP / API / 原生能力在启动或刷新时由对应通道发现；本清单只登记代码内静态存在的工具与通道形态。
- **MCP 自动探测**：未配置 `mcp_servers` 且 `mcp_auto_discover=true`（默认）时，启动发 JSON-RPC `initialize` 探测 `http://127.0.0.1:8091/mcp`；拿到 `result` 才追加名为 `host` 的 HTTP MCP Server，探测失败 best-effort 跳过、不报错；可用 `mcp_auto_discover: false` 或环境变量 `AGENT_MCP_AUTO_DISCOVER` 关闭。
- **宿主桥契约**：`GET {bridge.url}/capabilities` 返回 `{capabilities:[{name,description,input_schema,risk}]}`；`POST {bridge.url}/invoke`，body `{capability,payload}`，鉴权头 `Authorization: Bearer {token}`。地址/令牌由 `AGENT_BRIDGE_URL` / `AGENT_BRIDGE_TOKEN` 注入；地址留空时不建桥，模型退回 mock。
- **embedding 兜底**：宿主桥未暴露 `embed` 能力、或向量解析失败时，知识库不伪造向量，退回关键词/混合检索；向量能力名可由 `embedding.capability` 或 `AGENT_EMBEDDING_CAPABILITY` 覆盖。
- **脚本通道边界**：`run_python` / `run_shell` 的 `cwd` 必须落在工作区内（绝对路径或 `..` 越界一律拒绝），bash 以非登录方式执行；超时或取消按进程组整组回收；`returncode is None` 判失败不兜底 0；风险等级由 `script_risk` 配置，经权限门管控。
- **三层记忆口径**：第一层会话状态单条例行 + 写入锁，入口先读后回写；第二层三张经验表跨会话累积；第三层七域作为知识库文档的域索引。三层与 `facts` 四类记忆并存，互不覆盖。
- **规划模式**：入口单次规划输出严格 JSON，三种模式 `script`（合并脚本）/ `tools`（分步工具）/ `answer`（直接回答）；解析失败回退常规逐步循环。
- **计划 DAG**：`tools` 模式步骤可用 `id` 与 `depends_on:["上游id"]` 声明依赖；执行时上游失败或被跳过，下游标记「跳过：上游未成功」并不再执行，无依赖的分支继续；未声明依赖时按原顺序串行执行。
- **预算与自修正**：`max_calls` / `max_tokens` 超限立即以 `incomplete` 收尾并说明耗尽原因；工具失败且 `repair_rounds > 0` 时，以同一工具、修正后的参数重试（不擅自换工具），修正也受预算与取消信号约束。
- **技能与 MCP 运行时登记**：`skills install/uninstall` 只操作 `skills_dir` 内目录（安装拒绝同名覆盖，卸载拒绝越界路径）；`mcp add/remove` 写 `mcp.local.yaml` 而非 `agent.yaml`，下次启动合并生效。
- **多模态入口**：`run --image`（可重复）把本地图片读为 base64 `data:` URL 塞进 OpenAI `image_url` 内容块，单张上限 5MB；模型不支持视觉时由模型侧报错，不做本地降级伪造。
- **幂等重放**：`ToolJournal.key()` 对「工具名 + 规范化参数 JSON」取 sha1；仅 `idempotent` 且风险为 L0/L1 的工具在同会话内命中成功记录时直接回放，不再触达通道，审计记 `tool_replay`。L2/L3 不重放，避免重复产生副作用。
- **策略化重试**：加固后只有超时 / 连接失败 / 不可达等瞬时错误才重试，权限拒绝、参数错误、未知工具立即放弃；重试次数取 `tool_retries`，指数退避上限 2 秒，响应取消信号。
- **并行协同**：`Coordinator` 一次模型拆解为 2~5 个子任务，`Runtime.spawn_agent()` 为每个子任务建独立 Agent（共享注册表/模型/策略/记忆/配方/计量，独立取消信号与会话）；子任务异常转 `failed`，不拖垮整批。
- **角色化协作**：执行者（`executor`）结果写入共享 `Blackboard`；`review=true` 时追加审阅者（`reviewer`）一次模型调用，输出 `{answer, conflicts, verdict}`，解析失败退回机械汇总；有冲突且原本全部成功时状态降为 `incomplete`，不假装完成。
- **可观测**：`Telemetry` 记录模型调用与工具调用的 span（名称、类型、耗时、成败、属性）；`metrics` 输出总数、失败数、总/平均耗时、按工具计数与最慢 5 条；span 与审计分离——审计管「谁做了什么」，span 管「花了多久」。
- **治理限流**：`tool_rate_limits` 按通道限流，超限调用在注册表层直接拒绝并回显限额，不进入底层通道；限流与权限门、预算熔断相互独立、可叠加。
- **审计导出**：`audit export` 先 `verify` 整链，校验失败拒绝导出；成功则原样复制已脱敏 JSONL，便于离线取证。
- **与旧体系差异**：本清单对应 `src/doubao_agent/` 新运行时；旧版 `scripts/` 子 Agent、主权 Skill CLI、宿主连接器白名单等形态在本工程代码内不存在，未登记。
