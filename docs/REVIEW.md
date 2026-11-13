# 体系审查报告 REVIEW

> 审查对象：`qwqwqw123456/0919` 仓库，分支 `feat/evolution-layer`，commit `7f0972b`，工作区 clean。
> 审查方法：`src/doubao_agent/` 源码逐文件实读 + 关键缺陷本机实测复现 + 全量回归 `pytest -q`。
> 基线实测：`src/doubao_agent` 47 个 Python 文件 / 8208 行；`tests` 29 个 `test_*.py`（含 `conftest.py` 共 30 个）/ 3458 行；`eval` 1 个 / 132 行；`skills` 2 个 / 135 行；全仓 80 个 Python 文件 / 11933 行；SQLite 16 张业务表（10 个 Store）；**171 passed**（3 个第三方/配置类弃用告警）；版本 0.2.0。

## 一、总体结论

分层抽象和功能广度已达到可产品化原型水准，但**默认安全**与**长驻可靠性**两条线尚未闭环。当前版本**不建议直接暴露到网络，也不建议长时间无人值守运行守护进程**。

最紧急的是 6 个 P0：控制台裸奔、脚本免审批、主执行路径丢失备用模型配置、降级链会崩且不可见、401/403 被误判可重试、预算进程级累计不复位。

## 二、体系优点（先肯定）

- **统一注册表**：MCP / HTTP API / 浏览器 / 桌面 / 脚本 / 技能 / 知识 / 系统走同一 `ToolSpec` 协议，source 接口清晰，新增通道成本低。
- **记忆分层完整**：短期工作记忆 / 长期教训观察 / 知识库 / 配方回放四层；审计追加式哈希链；事件总线；span 级遥测。
- **权限按可逆性分级** + 预授权规则；工具输出用不可信头脚包裹；检查点续跑；配方零模型回放；并行协同 + 可选审阅。
- 离线评估集进入 pytest 闸门，171 项回归全绿。

## 三、P0：先止血（安全与确定性缺陷）

### P0-1 控制台默认监听全网，且可不鉴权

- `cli.py:900` 固定 `host="0.0.0.0"`，没有参数可改回环。
- `config.py:110` `auth_token=""` 默认空；`server/app.py:154` 空 token 时所有 `/api/*` 直接放行。
- 裸奔暴露的写/敏感端点：`POST /api/run`（`app.py:284`）、`POST /api/webhook`（`app.py:301`）、`POST /api/parallel`（`app.py:333`）、`DELETE /api/knowledge/{id}`（`app.py:226`）、`DELETE /api/schedules/{id}`（`app.py:389`）；`GET /api/events` 是全局广播（`app.py:356-370`），任意客户端能看到所有会话的任务与工具输出。
- webhook 在无签名头时直接判过（`app.py:39-40`）；空 token + 无签名即接受。
- **组合风险**：配置真实模型时，结合 P0-2 的脚本免审批，等于网络上任何人可触发命令执行；mock 模型下至少可删知识库、窃听全部事件。

### P0-2 脚本通道默认 L1，guarded 模式免审批跑任意命令

- `config.py:77` 默认 `script_risk="L1"`；`types.py:13` L1 = 可撤销。
- `permissions.py:103-106` guarded 只拦 L2/L3，L1 直接放行。
- `sources/script.py:154,160` 直接 `python -c code` / `bash -c command`；只把 cwd 限在工作区（`script.py:109-118`），代码内仍可用绝对路径读写、联网，**不是沙箱**。
- 服务端 `app.py:123` 未传 approver，L2/L3 一律拒，但 L1 脚本照跑。

### P0-3 主执行路径丢失备用 provider / 重试 / 模型摘要配置

- `runtime.py:175-197` 主 `agent` 装配了 fallback / retries / model_summary；但 `runtime.py:244-263` 的 `spawn_agent` **未传** `fallback_provider`、`provider_retries`、`retry_delay`、`model_summary`。
- `runtime.py:277` 的 `start_run` 每次都走 spawn_agent；CLI run/resume/schedule、服务端 run/webhook/parallel 全是这条路。只有配方 replay 用带兜底的 `runtime.agent`（`cli.py:683`）。
- 即：配置里写的备用模型、重试次数、摘要开关，**正常跑任务时未生效**。

### P0-4 降级链：残留状态会崩，且降级对用户不可见（已实测复现）

- `core/agent.py:502` 在 `reply` 赋值（`agent.py:524`）之前就引用它。本机复现：降级残留后二次 run 首轮抛 `UnboundLocalError: cannot access local variable 'reply'`。
- 这段标注本身是死代码：它修改的是上一轮已写进消息的 `reply`，随后即被覆盖；`reply["degraded"]`（`agent.py:331-332`）从未进入任何最终返回 dict，**用户看到的降级答复与正常答复无差别**。
- 测试盲区：`tests/test_provider_fallback.py:91-93` 只断言 `_complete` 返回的 reply 带 degraded，未验证最终 answer/结果是否透出。

### P0-5 401/403/400/404 被误判为可重试（已实测复现）

- `agent.py:184-189` `_status_of` 只找 `status_code/status/code`；`httpx.HTTPStatusError` 只挂在 `exc.response.status_code` 上。
- 实测：401 时 `hasattr(exc, "status_code")=False` → `_status_of=None` → `_retryable=True`。**密钥错、权限错也会重试**，可能静默切到 mock/备用，产出看似正常的错误结果。
- 测试盲区：现有用例用自定义 `EndpointError`（自带 `.status_code`，`test_provider_fallback.py:9-12`），未覆盖真实 httpx 异常。`config.py:83` 注释与 GAPS.md 写的"401/403 不重试"与实现不符。

### P0-6 预算按进程累计、不复位，守护进程跑一阵后永久"预算用尽"

- `core/usage.py:11-31` 无 reset；`runtime.py:172` 只构造一个 meter，所有 run 复用。
- `config.py:104-105` 注释写的是"单次运行预算"，口径不一致。CLI 一次性进程等价单次；**守护进程/控制台长期跑，累计超上限后，后续所有任务一进来就返回"预算已用尽"**。并行任务还共享同一份计量互相挤占。

## 四、P1：稳定性、一致性与并发

| # | 问题 | 证据 |
| --- | --- | --- |
| P1-1 | **工作记忆全局单行**，并行子任务互相覆盖：焦点线/工作流节点/待回顾被最后完成者改写，计数被放大 | `working.py:37` `CHECK(id=1)`；`working.py:119-145` |
| P1-2 | **10 个 Store 各自开 SQLite 连接**，无 WAL、无 busy_timeout；跨进程（daemon+CLI 同库）易 `database is locked`，默认回滚日志读写互阻 | session/memory/working/longterm/knowledge/recipe/scheduler/journal/telemetry/evolution 各一处 `sqlite3.connect`；全库无 `journal_mode`/`busy_timeout` |
| P1-3 | **异常时会话不收尾**：`run()` finally 只结工作记忆，没 `sessions.finish("failed")`，CLI/并行路径会话悬挂 running；只有服务端外层兜底 | `agent.py:429-436`；兜底见 `app.py:271,296,327` |
| P1-4 | **审计链偏弱**：哈希截断 16 hex（64bit）；追加读尾+写无文件锁，`_last` 内存缓存跨进程会 stale 断链；脱敏只按字段名，`code/command/content` 内嵌的 Bearer/api_key= 不打码；无 seq；长文件全量读入再切片 | `audit.py:85`、`audit.py:88-91`、`audit.py:27-40`、`audit.py:93-106` |
| P1-5 | **Planner/Coordinator/摘要绕过统一模型层**：直连 `provider.complete`，没重试/降级/遥测；decompose docstring 承诺"失败兜底单任务"但无 try/except；摘要调用不计 usage | `planner.py:134-141`、`coordinator.py:101-107`、`coordinator.py:218`、`agent.py:84-102` |
| P1-6 | **MCP 一次失败永久锁存**：`_failed` 后不再重连、无退避，health 也不重置；stdio 子进程继承全量环境变量（含密钥） | `mcp.py:73-79`、`mcp.py:133` |
| P1-7 | **技能通道取消/进程组回收缺失**：无 `start_new_session`，超时只 kill 直进程、子进程泄漏；不读 cancel_event。对照 script 通道三者齐全，能力不一致 | `skill.py:105-128`；对照 `script.py:121-134,181` |
| P1-8 | **缓存/限流**：先扣限流再查缓存（命中也耗配额）；FIFO 淘汰非 LRU；缓存无 session/租户隔离；写操作不失效读缓存；后缀名二义性直接返回 unknown | `registry.py:88-99`、`registry.py:151-155`、`registry.py:80-81` |
| P1-9 | **HTTP API body 混入 query**：有 body 的端点，body 字典把未分类的 query 键也带进去 | `http_api.py:175-189` |
| P1-10 | **配方/回放**：失败步骤仍录制并回放；配方按次 INSERT 不去重；journal preview 截断 2000 字符且无 truncated 标记，回放返回静默截断内容 | `agent.py:570-573`、`recipe.py:37`、`journal.py:28,93` |
| P1-11 | **可观测缺口**：遥测/审计无保留策略，长期无限增长；全局 metrics 每次扫 5000 行；SSE 无心跳；多处 `except Exception: pass` 吞故障且无结构化日志 | `telemetry.py:107-112`、`app.py:360-368`、`watcher.py:75,100-101` |

## 五、P2：架构与工程化

- **`agent.py` 981 行职责过重**：主循环、上下文压缩、计划执行、工具执行、修复重试、产物登记、记忆回写混在一个类里。
- **SQLite 样板重复 10 份**：连接、建表、锁、close 同构，应抽统一 DB 层（顺带解决 P1-2）。
- **`_emit` 在 Agent 与 Coordinator 各写一份**（`agent.py:350`、`coordinator.py:109-117`）。
- **无 CI、无静态检查**：没有 `.github/workflows`，pyproject 无 ruff/mypy/black 配置，无依赖锁文件。
- **配置无 schema 校验**：坏 YAML 静默回退默认（`config.py:172-176`），不报错也不告警。
- **服务端无审批闭环**：guarded 下 L2/L3 硬拒；脚本风险一旦提到 L2，控制台脚本全不可用，缺"待审批队列 + 批准/拒绝"API 和界面。
- **硬编码路径**：daemon 未传 config 时写死 `/workspace/agent.yaml`（`cli.py:222`）。
- 知识库向量缓存满即整体清空、非 LRU 且不落库，重启重算（`knowledge.py:271-272,283-284`）；工作记忆用中文字段名做持久化键，耦合展示与存储；多模态仅图片；无流式输出。

## 六、升级路线（分阶段，不含工期估算）

### 阶段 0：止血（默认安全 + 确定性缺陷）

1. 监听默认 `127.0.0.1`；绑 `0.0.0.0` 必须显式开关且强制非空 token；非回环 + 空 token 直接拒启动。
2. 脚本默认风险提到 L2；加危险命令分类（`rm -rf`、`sudo`、`dd/mkfs`、`curl|sh`、系统路径重定向、外联下载执行）动态升 L3；审批通道就绪前高危默认拒。
3. 删 P0-4 死代码块；降级信息写进最终 result/answer、事件、审计、遥测；保证二次 run 不崩。
4. `_status_of` 补 `exc.response.status_code`；400/401/403/404 不重试；补真实 httpx 异常用例。
5. `spawn_agent` 透传 fallback/retries/retry_delay/model_summary；补配置一致性测试。
6. 每 run 独立 `UsageMeter`（或按 session 计量），进程级只做累计统计；补"两次连续 run 预算各自独立"测试。

### 阶段 1：可靠性收口

1. 统一 DB 层：WAL + `busy_timeout=5000` + 外键开关，按路径复用连接，版本化 migration。
2. 会话生命周期兜底：异常/取消都 `finish`，CLI/并行路径对齐服务端兜底。
3. 审计加固：完整 SHA-256、文件锁、值级正则脱敏、记录 seq。
4. 抽 `complete_with_resilience()`（重试/降级/计量/遥测统一入口），Planner/Coordinator/摘要全部接入；decompose 真正兜底 `[task]`。
5. MCP 指数退避重连 + 健康态重置；stdio 环境变量白名单。
6. 技能通道对齐 script：新会话组、killpg、cancel 监听。
7. 缓存：先查缓存再扣限流、LRU、按 session 分桶、写操作失效。
8. API 区分 body/query；配方只录成功步骤并按哈希去重；截断补 `truncated` 标记。
9. 遥测/审计保留滚动策略；SSE 心跳 + 按会话订阅；吞异常处落结构化日志。

### 阶段 2：架构升级

1. 拆 Agent 为 ContextBuilder / ToolExecutor / PlanExecutor / MemoryWriter，主循环只编排。
2. 服务端审批服务：待办队列 API + 控制台 UI，超时自动拒绝；L2/L3 走队列而非硬拒。
3. 工作记忆按 session 分区，全局行只留汇总。
4. 配置 Pydantic schema，启动即校验报错；事件总线按 session 主题隔离。
5. 结构化日志 + OTel/Prometheus；CI 接 pytest + ruff + mypy + 密钥扫描，固定依赖锁。

### 阶段 3：能力演进

1. 执行沙箱（容器 / Landlock / seccomp / nsjail）：env 白名单、只读根、工作区可写。
2. 模型流式输出，降首字延迟。
3. 多租户 RBAC：token→角色，会话/知识库/缓存/审计按租户隔离。
4. A2A 跨进程协作与共享记忆契约；eval 扩充安全/并发/降级/断链用例。

## 七、验收标准（可测）

- 非回环 + 空 token 启动失败；无 token 请求 401；SSE 只见本会话事件。
- 100 并发 run：0 次 `database is locked`、0 个悬挂 running 会话。
- 401/403 调用次数 = 1；降级答复在 answer/事件/审计里都有可见标注。
- 两次连续 run 预算各自独立；spawn 出的 agent 与主 agent 的 fallback/retries 一致。
- 跨进程追加审计后 verify 通过；篡改任一行即校验失败。
- CI 绿灯，核心模块有单测覆盖；配置错误启动即报错，不再静默默认。

## 八、附：实测复现记录

1. **401 误重试**：构造 `httpx.Response(401).raise_for_status()` → `_status_of(...) = None` → `_retryable(...) = True`；`_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}`。
2. **降级残留崩溃**：Agent 实例手动置 `last_degraded` 非空后二次 `run()`，首轮循环抛 `UnboundLocalError: cannot access local variable 'reply'`。

## 九、文档口径偏差清单（待随阶段 0 修复回填）

| 位置 | 当前表述 | 实现现状 |
| --- | --- | --- |
| `docs/GAPS.md:39` G6 | "401/403/404 不重试""降级标注透出""备用 provider 主路径生效"，状态"已落地" | 401/403 被误判可重试；降级字段未透出最终结果；备用 provider 未透传到 spawn_agent 主路径 |
| `docs/GAPS.md:14` | 预算为"单次运行" | UsageMeter 进程级累计、不复位 |
| `docs/GAPS.md:13` | 审计"凭据字段递归脱敏" | 审计只按字段名脱敏；值模式脱敏仅在长期记忆出口 |
| `docs/GAPS.md:21` | 回放按参数指纹命中 | preview 截断 2000 字符，回放静默截断，无 truncated 标记 |
| `README.md:43`、`assets/interface-manifest.md:253` | 预算"单次运行" | 同 P0-6，进程级累计 |
| `assets/interface-manifest.md:255` | "401/403/404 不重试""切换后结果标注 degraded" | 同 P0-4、P0-5 |
| `docs/ARCHITECTURE.md:228` | "拆解失败则整体作为单任务兜底" | `decompose` 无 try/except，异常直接传播，兜底未实现 |

## 十、附：Trae 互连通路实测（2026-11-13，环境 Ubuntu 22.04 / Node 22）

### 已实测落地

- **CLI 安装**：`traecli 0.214.1`（官方安装脚本，用户级目录 `~/.local/bin` + `~/.trae`，无 sudo；附带 bwrap 沙箱，复用系统 ripgrep）。安装产物在 home 目录，未污染本仓库。
- **Trae 作为 MCP client（挂外部标准 MCP server）**：`traecli mcp add <name> --url <URL>` 或 `-- <COMMAND>`，配置写入 `~/.trae/traecli.toml` 的 `[mcp_servers.*]`；`list/remove` 本地读写已实测通过。
- **Trae 作为被调方的服务面**：
  - `exec-server --listen ws://IP:PORT`（也支持 `stdio`）：JSON-RPC 2.0 over WebSocket，Codex app-server 契约；可 `--remote` 注册为远程 executor。
  - `acp serve`：stdio 上的 ACP（Agent Client Protocol）服务。
  - `app-server daemon/proxy`：本地守护 + control socket 代理。
  - 注意：旧版文档里的 `mcp-server` 子命令在 0.214.1 **已不存在**；该版本对外暴露走 exec-server / ACP / app-server，不是标准 MCP server。
- **握手可达**：`exec-server` 起服务后 WS 升级返回 `101 Switching Protocols`；JSON-RPC `initialize`（params `clientName`/`clientVersion`）成功，返回 `sessionId` 与 `environmentInfo`（shell、cwd、沙箱能力清单）。**未登录状态下握手层不拦连接**。

### 当前卡点

- **鉴权**：`doctor` 显示 `Auth: required trae, status: not_logged_in`。真实模型任务在调用层校验；无头 `exec` 先撞 CLI 内置鉴权（解析内网 `code.byted.org` 失败 / 代理 502）。登录方式：交互 `traecli login`（浏览器/设备码）、PAT 注入环境变量 `TRAE_ACCESS_TOKEN`、端点可用 `TRAE_API_BASE_URL` 覆盖。
- **协议适配差距**：本仓库 `sources/mcp.py` 只认标准 MCP（stdio / SSE / Streamable HTTP）；接 Trae 的 exec-server/ACP 需另写一层 JSON-RPC / ACP 客户端适配，不能直接挂进现有 MCP 通道。
- **契约参考**：`traecli app-server generate-json-schema --out <DIR>` 可导出协议 JSON Schema（本机临时目录，未纳入仓库）；协议含 initialize 握手、会话/命令执行/文件系统/配置/审批/MCP elicitation 等方法，服务端主动推消息 delta、命令输出、FS 变更等通知。

### 双向打通的最小路径

1. 先解决 Trae 登录态（PAT 或交互登录），在当前出口网络下验证一次无头 `traecli exec --json` 能跑通真实任务。
2. 方向 A（Trae 调我方工具）：在我方补一个标准 MCP server 适配层，把注册表 `ToolSpec` 暴露成 MCP tools，再由 `traecli mcp add` 挂入。
3. 方向 B（我方调 Trae）：在我方补 exec-server JSON-RPC 客户端（或 ACP 客户端），按导出的 schema 走 initialize → 会话创建 → 任务下发 → 通知接收；这对应 G9「A2A 跨进程协议」缺口。
