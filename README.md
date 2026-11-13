# doubao-agent

跑在豆包 App 云电脑上的 AI Agent 运行时。CPU 环境不自建模型，推理走 OpenAI 兼容接口，工具走本地通道 + 宿主 MCP。

## 快速开始

```bash
# 1. 依赖
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# 2. 密钥（不要写进仓库文件；.env.local 已被 .gitignore 排除）
# 运行时启动自动读取 .env.local 注入环境变量，无需手动 source。
cat > .env.local <<'EOF'
OPENAI_API_KEY='你的密钥'
EOF

# 3. 配置
cp agent.example.yaml agent.yaml   # 按需修改 model 段
.venv/bin/doubao-agent run "往知识库写一条：项目已启动，然后检索‘项目’"
```

## 配置要点（`agent.yaml`）

| 段 | 说明 |
| --- | --- |
| `model` | `provider: openai` + `base_url`（带版本段）+ `model` + `api_key_env`；`mock` 仅用于离线联通测试 |
| `permission_mode` | `auto` / `guarded` / `manual`；配合 `auto_approve` 规则可无人值守 |
| `max_calls` / `max_tokens` | 单次运行的模型预算，0 表示不限；接真实模型建议设上限 |
| `repair_rounds` | 规划步骤工具失败时，让模型修正参数的轮数 |
| `tool_retries` | 幂等工具遇到瞬时错误（超时/连接/不可达）的重试次数 |
| `tool_rate_limits` | 按通道的每分钟调用上限，如 `{browser: 30, default: 120}` |
| `mcp_servers` / `mcp_auto_discover` | 标准 MCP 接入；留空时启动自动探测 `127.0.0.1:8091/mcp` |
| `bridge` | 通用 REST 宿主桥（当前云电脑未提供，保持 `enabled: false`） |
| `auth_token` | 控制台 `/api` 鉴权；空表示不鉴权（仅本机） |

## 常用命令

```bash
doubao-agent run "任务描述" [--session <ID>] [--no-record] [--image <路径>]
doubao-agent resume <会话ID>
doubao-agent sessions
doubao-agent tools | sources | skills [reload|install <目录>|uninstall <名称>]
doubao-agent mcp list | mcp add <名称> --url <地址> | mcp remove <名称>
doubao-agent state show
doubao-agent longterm stats
doubao-agent knowledge domains
doubao-agent memory decay --days 30     # 按半衰期衰减非偏好记忆
doubao-agent metrics [--session <ID>] [--detail]  # span 汇总 / 明细、慢调用、失败数
doubao-agent audit verify [会话ID] | audit sessions | audit export <会话ID>
doubao-agent parallel "任务" --review    # 并行执行 + 审阅者合并仲裁
doubao-agent serve                     # 控制台 + SSE，默认 8765
```

## 可靠性与可观测

- 工具执行写入 `tool_runs`：同会话内幂等工具（L0/L1）按「工具 + 参数指纹」命中成功记录时直接回放，不再重复执行；L2/L3 永不重放。
- 瞬时错误（超时/连接/不可达）按 `tool_retries` 指数退避重试；权限拒绝与参数错误不重试。
- 模型调用与工具调用写入 `spans`，`metrics` / `GET /api/metrics` 可看总耗时、慢调用与失败数。
- 审计链支持 `audit export`：先校验再导出，链损坏拒绝导出。

## 记忆结构

- **工作记忆**：`working_state` 单条状态，含焦点线、工作流节点、写入锁等 16 项，会话开头先读。
- **长期记忆**：`lessons`（日期/事件/判断/结果/教训）、`observations`、`deviations` 三张表。
- **知识域**：知识文档可归属 `01 认知与思维` … `07 国家政策` 七个域，检索可域内过滤，`domain=auto` 按查询自动路由。
- **记忆闭环**：被注入且任务成功则加分、失败则降权；`decay` 按未使用时长半衰期衰减，偏好不衰减。

## 通道

`script`（Python/Shell）· `knowledge`（分块检索）· `skill`（渐进加载）· `mcp`（stdio/sse/http）· `api`（OpenAPI 声明）· `native` / `browser` / `desktop`（依赖宿主能力）。

脚本通道边界：工作区内的 `cwd`、非登录 shell、超时或取消整组回收子进程。

## 测试

```bash
.venv/bin/python -m pytest tests -q
```
