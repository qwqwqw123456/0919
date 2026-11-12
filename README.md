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

## 换机 / 整机重建恢复

云电脑沙箱重建后本地状态会清空，源码与配置模板在 git 仓库，按下面恢复：

```bash
git clone <仓库地址> && cd doubao-agent
./scripts/bootstrap.sh          # 建 .venv、装依赖、备配置、装入内置技能、自检（幂等，不含密钥）
# 脚本检测到缺 .env.local 时会提示；手动写入密钥后即可运行
```

- **随仓库恢复**：源码、Python 依赖（`pyproject.toml`）、配置模板（`agent.example.yaml`）、内置技能；脚本把 `skills/` 下内置技能装入 `skills_dir`（同名已存在则跳过，不覆盖自建）。
- **不随包迁移**：`.env.local`（密钥，已 gitignore，换机后手动填，不入库不外发）；`.agent/`（会话 / 记忆 / 知识库 / 调度登记）、`runs/`（审计产物）、以及你在 `skills_dir` 自建的技能均为本机易失状态，新环境从空态起步，需要时另行导出恢复。
- 浏览器 / 电脑操作驱动属于平台层可选包，不在本仓库依赖内；需要真实 GUI 操作时按平台说明补装。

## 配置要点（`agent.yaml`）

| 段 | 说明 |
| --- | --- |
| `model` | `provider: openai` + `base_url`（带版本段）+ `model` + `api_key_env`；`mock` 仅用于离线联通测试 |
| `permission_mode` | `auto` / `guarded` / `manual`；配合 `auto_approve` 规则可无人值守 |
| `max_calls` / `max_tokens` | 单次运行的模型预算，0 表示不限；接真实模型建议设上限 |
| `repair_rounds` | 规划步骤工具失败时，让模型修正参数的轮数 |
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
doubao-agent evolve status | audit [--apply] | pin <技能> | distill <配方号> | report
doubao-agent state show
doubao-agent longterm stats
doubao-agent knowledge domains
doubao-agent audit verify [会话ID] | audit sessions
doubao-agent serve                     # 控制台 + SSE，默认 8765
```

## 自进化

技能用量自动记账（use / view / 最后使用），按 14 / 30 天阈值流转 active → 闲置 → 归档，
只改状态位不删文件；成功配方可蒸馏成技能草案放进 `skills/_drafts`，人工确认后再安装。
详见 [docs/EVOLUTION.md](docs/EVOLUTION.md)；架构方案见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，缺口路线图见 [docs/GAPS.md](docs/GAPS.md)。

## 记忆结构

- **工作记忆**：`working_state` 单条状态，含焦点线、工作流节点、写入锁等 16 项，会话开头先读。
- **长期记忆**：`lessons`（日期/事件/判断/结果/教训）、`observations`、`deviations` 三张表。
- **知识域**：知识文档可归属 `01 认知与思维` … `07 国家政策` 七个域，检索可域内过滤。

## 通道

`script`（Python/Shell）· `knowledge`（分块检索）· `skill`（渐进加载）· `mcp`（stdio/sse/http）· `api`（OpenAPI 声明）· `native` / `browser` / `desktop`（依赖宿主能力）。

脚本通道边界：工作区内的 `cwd`、非登录 shell、超时或取消整组回收子进程。

## 测试

```bash
.venv/bin/python -m pytest tests -q
```
