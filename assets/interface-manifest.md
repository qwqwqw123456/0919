# 接口清单 interface-manifest（doubao-agent v0.1.0 · 2026-11-12）

> 本清单只登记当前源码实际暴露的调用形态，来源为 `src/doubao_agent/` 实读与本地实跑，不凭记忆补写。
>
> 实跑基线：`src/` 下 38 个 Python 文件 / 4957 行；测试 46 passed；注册表静态工具 6 个，原生兜底工具 17 个（条件触发，见第 3 节）。
> 源码根：`src/doubao_agent/`；入口脚本 `doubao-agent = doubao_agent.cli:main`。

---

## 1. 调用面总览

| 层 | 形态 | 说明 |
| --- | --- | --- |
| CLI | `doubao-agent <子命令> [参数]` | 本地命令行，覆盖运行、记忆、知识、调度、配方、并行、服务与守护 |
| 控制台 HTTP | FastAPI 路由 `/api/*` + SSE `/api/events` | 本地 Web 控制台，工具/技能/知识/会话/审计/调度/并行 |
| Agent 内核 | `Agent.run()` 规划-执行-观察-反思主循环 | 单次入口规划（可选）→ 工具执行 → 结果归一化 → 记忆/配方沉淀 |
| 能力注册表 | `CapabilityRegistry` 聚合多通道工具 | 上层只认 `ToolSpec`，不关心工具来自 MCP / API / 浏览器 / 桌面 / 原生 / 脚本 / 技能 / 知识 |
| 模型供给 | `host / openai / mock` 三档 Provider | 统一 `complete(messages, tools) -> {content, tool_calls, usage}` |
| 持久化 | SQLite + JSONL 审计 | 记忆 / 会话与检查点 / 知识库 / 配方 / 调度，断连可续跑 |

---

## 2. 通道（SourceKind，共 8 类）

注册表启动时对每个通道调用 `list_tools()` 发现工具；工具名含 `.` 保留原名，裸名自动加通道前缀（`<source>.<name>`）。

| 通道 | 源码 | 工具来源 | 默认风险 | 备注 |
| --- | --- | --- | --- | --- |
| `builtin` 脚本 | `sources/script.py` | 代码即动作：Python / bash 非交互执行 | 可配（默认 L1） | `run_python` / `run_shell`；CPU 上限可选 RLIMIT_CPU；支持会话级取消 |
| `skill` 技能 | `sources/skill.py` | `skills/<name>/SKILL.md`；可执行技能注册为工具，知识技能按需阅读 | 技能 frontmatter 声明，缺省 L1 | 入口限定在技能目录内；参数以 JSON 从 stdin 传入 |
| `knowledge` 知识 | `sources/knowledge.py` | 长文档分块检索与回写 | search L0 / add L1 | 与短期记忆分工：知识记长文档，记忆记短事实/偏好/经验 |
| `native` 原生 | `sources/native.py` | 宿主 `/capabilities` 清单动态发现 | 清单声明，否则按名称关键词推断 | 桥不可用时返回空；桥存在但清单为空且 `fallback=true` 时挂浏览器/桌面兜底目录 |
| `browser` 浏览器 | `sources/browser.py` | 兜底工具目录（9 个） | 只读 L0 / 可逆 L1 | 仅在原生通道走兜底目录时注册；实际执行交宿主桥 |
| `desktop` 桌面 | `sources/desktop.py` | 兜底工具目录（8 个） | 只读 L0 / 可逆 L1 | 同上 |
| `mcp` | `sources/mcp.py` | MCP stdio / SSE / Streamable HTTP，工具自描述 | 按名称关键词推断 | 工具由配置的 MCP Server 启动时发现，代码不写死清单 |
| `api` | `sources/http_api.py` | OpenAPI 文档或声明式 JSON → 动态生成工具 | 文档声明，否则按名称关键词推断 | 工具由配置的 API 规格启动时发现 |

---

## 3. 静态内置工具（本地实跑基线）

不依赖外部配置、注册后稳定存在的工具：

| 工具名 | 通道 | 风险 | 入参要点 | 说明 |
| --- | --- | --- | --- | --- |
| `script.run_python` | builtin | L1 | `code:str`，可选 `cwd`、`timeout` | 执行 Python 代码，print 输出关键结果 |
| `script.run_shell` | builtin | L1 | `command:str`，可选 `cwd`、`timeout` | 执行非交互 bash 命令 |
| `knowledge.search` | knowledge | L0 | `query:str`，可选 `limit:int=4` | 关键词 + 字符 bigram 混合检索；接 embedding 后切余弦向量 |
| `knowledge.add` | knowledge | L1 | `title:str`、`text:str`，可选 `source:str` | 写入知识库并自动分块 |
| `skill.read` | skill | L0 | `name:str` | 阅读指定技能完整说明与资源清单（渐进式加载） |
| `skill.text-stats` | skill | L0 | 见技能 `input_schema` | 示例可执行技能，入口 `skills/text-stats/main.py`，统计字符/中文/词/行/高频词 |

### 3.1 原生兜底工具目录（条件触发）

仅当宿主桥存在、清单为空且 `fallback=true` 时注册；桥不可用时不挂。

- 浏览器 9 个：`browser.open`、`browser.snapshot`、`browser.screenshot`、`browser.click`、`browser.type`、`browser.upload`、`browser.tabs`、`browser.extract`、`browser.back`
- 桌面 8 个：`desktop.screenshot`、`desktop.screen_size`、`desktop.windows`、`desktop.focus`、`desktop.move`、`desktop.click`、`desktop.type_text`、`desktop.hotkey`

### 3.2 动态工具（不在代码内枚举）

- MCP 工具：由 `agent.yaml` 的 `mcp_servers` 启动对应 Server 后发现。
- API 工具：由 `api_sources` 指向的 OpenAPI / 声明式规格生成。
- 可执行技能：扫描 `skills_dir/*/SKILL.md`，有 `entry` 的注册为 `skill.<name>`。

---

## 4. CLI 命令（`cli.py`，14 个子命令）

全局选项：`--config <路径>`（默认 `/workspace/agent.yaml`，亦可由环境变量 `DOUBAO_AGENT_CONFIG` 指定）。

| 子命令 | 位置参数 / 选项 | 作用 |
| --- | --- | --- |
| `tools` | — | 列出注册表全部工具（风险 / 通道 / 名称 / 描述） |
| `sources` | — | 检查各接入通道健康状态 |
| `skills` | `[list\|reload]` | 列出技能；`reload` 热扫描技能目录并刷新注册表 |
| `sessions` | — | 列出最近会话 |
| `resume` | `session_id` | 从检查点恢复会话 |
| `run` | `task`；`--session <id>`；`--no-record` | 执行一个任务；`--no-record` 不录制配方 |
| `recipes` | — | 列出已录制配方 |
| `replay` | `recipe_id:int` | 按配方零模型重放 |
| `memory` | `action`；`value?`；`--kind`；`--limit=20` | `list / add / search / forget / stats`；`--kind` 见第 9 节四类 |
| `schedule` | `action`；`args*`；`--name` | `add / list / remove / enable / disable / run`；cron 5 段 |
| `knowledge` | `action`；`value?`；`--title`；`--text`；`--limit=4` | `list / add / search / remove / stats` |
| `parallel` | `task`；`--subtasks a,b,c` | 并行协同；省略子任务则先由模型拆解 |
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
| POST | `/api/parallel` | 后台并行执行（`task`，可选 `subtasks`） |
| GET | `/api/events` | SSE 事件流（见第 6 节） |
| GET | `/api/schedules` | 定时任务列表 |
| POST | `/api/schedules` | 新增定时任务（`name`、`cron`、`task`） |
| DELETE | `/api/schedules/{schedule_id}` | 删除定时任务 |
| POST | `/api/schedules/{schedule_id}/toggle` | 启用 / 停用（`enabled`） |
| POST | `/api/schedules/{schedule_id}/run` | 立即执行一次 |
| GET | `/` | 控制台静态页 `server/static/index.html` |

---

## 6. SSE 事件类型（`/api/events`）

| 事件 | 载荷要点 | 触发处 |
| --- | --- | --- |
| `start` | `session`、`task`、可选 `mode` | 任务/规划开始 |
| `tool_start` | `tool` | 工具调用前 |
| `tool_end` | `tool`、`ok`、`error` | 工具返回后 |
| `subtasks` | `task`、`subtasks`、`count` | 并行拆解结果 |
| `finish` | `session`、`answer`、可选 `subtasks` | 任务收尾 |
| `schedule_start` | `id`、`name` | 定时任务派发 |

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

## 9. 持久化存储（SQLite，9 张表）

| 表 | 所属模块 | 用途 |
| --- | --- | --- |
| `facts` | `core/memory.py` | 长期记忆四类：`preference` / `fact` / `experience` / `procedure`；按指纹去重加权 |
| `sessions` | `core/session.py` | 会话：任务、状态、结果、步数、时间戳 |
| `messages` | `core/session.py` | 会话消息流水（JSON 载荷） |
| `checkpoints` | `core/session.py` | 检查点：步骤与状态，断点续跑 |
| `artifacts` | `core/session.py` | 会话产物路径登记 |
| `documents` | `core/knowledge.py` | 知识库文档（标题、来源、时间） |
| `chunks` | `core/knowledge.py` | 文档分块（默认目标 500 字/块） |
| `recipes` | `core/recipe.py` | 成功工具序列，可零模型重放 |
| `schedules` | `core/scheduler.py` | cron 定时任务：表达式、任务、启用态、上次/下次运行 |

审计为追加式哈希链 JSONL：`<runs_dir>/<session_id>/audit.jsonl`。

---

## 10. 模型供给（`providers/`）

| Provider | 选择条件 | 说明 |
| --- | --- | --- |
| `host` | `model.provider=host`（默认有桥时） | 委托宿主能力 `model.complete`，无需 API key；桥缺失时退回 `mock` |
| `openai` | `model.provider=openai` | OpenAI 兼容 `/chat/completions`；key 取配置或 `OPENAI_API_KEY` |
| `mock` | 未识别或兜底 | 确定性脚本回复，供测试与离线演示 |

统一返回 `{content, tool_calls, usage}`；空 `choices` 或缺 `message` 回退空结果，不崩溃。

---

## 11. 配置项与环境变量

配置文件（`agent.yaml`，模板 `agent.example.yaml`）字段：

`workspace`、`runs_dir`、`db_path`、`permission_mode`、`script_risk`、`script_max_cpu`、`record_recipes`、`planning.enabled`、`skills_dir`、`context_limit`、`scheduler_enabled`、`scheduler_interval`、`max_steps`、`port`、`model`、`bridge`、`auto_approve`、`mcp_servers`、`api_sources`。

环境变量覆盖（前缀 `AGENT_*`）：

`AGENT_PORT`、`AGENT_PERMISSION_MODE`、`AGENT_SCRIPT_RISK`、`AGENT_SCRIPT_MAX_CPU`、`AGENT_SCHEDULER_ENABLED`、`AGENT_SCHEDULER_INTERVAL`、`AGENT_RECORD_RECIPES`、`AGENT_PLANNING`、`AGENT_SKILLS_DIR`、`AGENT_CONTEXT_LIMIT`、`AGENT_DB_PATH`、`AGENT_RUNS_DIR`、`AGENT_BRIDGE_URL`、`AGENT_BRIDGE_TOKEN`；配置路径 `DOUBAO_AGENT_CONFIG`；OpenAI 兜底 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`。

---

## 12. 备注口径

- **动态发现，不写死清单**：MCP / API / 原生能力在启动或刷新时由对应通道发现；本清单只登记代码内静态存在的工具与通道形态。
- **脚本通道边界**：`run_python` / `run_shell` 的 `cwd` 默认工作区，可由参数指定；风险等级由 `script_risk` 配置，经权限门管控。
- **规划模式**：入口单次规划输出严格 JSON，三种模式 `script`（合并脚本）/ `tools`（分步工具）/ `answer`（直接回答）；解析失败回退常规逐步循环。
- **并行协同**：`Coordinator` 一次模型拆解为 2~5 个子任务，`Runtime.spawn_agent()` 为每个子任务建独立 Agent（共享注册表/模型/策略/记忆/配方/计量，独立取消信号与会话）；子任务异常转 `failed`，不拖垮整批。
- **与旧体系差异**：本清单对应 `src/doubao_agent/` 新运行时；旧版 `scripts/` 子 Agent、主权 Skill CLI、宿主连接器白名单等形态在本工程代码内不存在，未登记。
