# 缺口路线图 GAPS

> 口径：以下结论来自 `src/doubao_agent/` 源码实读与本机实跑（基线 47 个 Python 文件 / 8208 行 / 29 测试文件 / 171 passed，分支 `feat/evolution-layer`）。
> 本表只登记「现状证据 → 缺口 → 归属 → 优先级 → 验收方向」，不重复 `ARCHITECTURE.md` 的设计原则。归属分两类：**运行时**（本仓库可改）、**平台侧**（云电脑宿主提供，运行时只做隔离适配）。

## 1. 已覆盖（不再重复规划）

| 能力 | 现状证据 |
| --- | --- |
| 三层记忆 + 检索 | `working_state` / `lessons·observations·deviations` / `documents·chunks`；知识库同步+异步检索，向量失败退回关键词混合 |
| 上下文压缩 | `core/agent.py` `_trim_messages`：保留 system + 最近消息，对齐 user 边界；被裁早期 user 要点并入 system 注记 |
| 调度 / 并行 / 续跑 / 重放 | `schedules` cron；`parallel` 子任务；检查点 + `resume`；配方 `replay` 零模型重放 |
| 权限与审计 | 风险 L0–L3；`auto / guarded / manual`；审计追加式哈希链 JSONL，凭据字段递归脱敏 |
| 预算与修正 | `max_calls` / `max_tokens` / `repair_rounds`；超限以 `incomplete` 收尾 |
| 注入防护 | 工具/网页/文件输出用不可信头脚包裹，其中指令不得执行（`_UNTRUSTED_HEADER/FOOTER`） |
| 守护自愈 | `daemon` 子进程退出按退避重启，SIGTERM 优雅收尾 |
| 自进化 | `skill_usage` 用量台账、active→stale→archived 生命周期、pin 保护、配方蒸馏草案 |
| 触发与并发 | 控制台 `POST /api/run`、`/api/webhook`（令牌/HMAC 签名）、`/api/parallel`、`/api/schedules`；全局 `Semaphore(3)` 并发闸 |
| 离线评估集 | `eval/golden/skills.json` 规则样例 + `eval/runner.py` 判定器，纳入 pytest 闸门；模型端到端样例占位 |
| 多模态输入 | `run --image`（可重复，单张 ≤5MB，OpenAI `image_url`） |
| 可靠性与可观测 | `tool_runs` 同会话幂等只读/可逆工具按参数指纹回放（L2/L3 永不重放）；`spans` 记录模型/工具耗时与成败，`metrics` 与 `GET /api/metrics`、`/api/sessions/{id}/trace` 聚合；`audit export` 先校验哈希链再导出；记忆 `feedback` 命中成功加分/失败降权、`decay` 半衰期衰减（偏好不衰减）；知识库 `domain=auto` 关键词路由；`parallel --review` 审阅者合并结果并标注冲突 |

## 2. 运行时缺口

### P0 — 决定能否长期无人值守且越用越稳

| # | 缺口 | 现状证据 | 验收方向 | 状态 |
| --- | --- | --- | --- | --- |
| G1 | **事件驱动触发** | 已有 cron 定时 + 手动/API 调起；`POST /api/webhook` 支持令牌或 HMAC-SHA256 签名入站触发；`watch_enabled` 开启后轮询目录 mtime 快照，新增/修改/删除广播 `file_change`，同路径同类事件在轮询窗口内防抖合并，配 `watch_task` 模板才触发运行 | 文件变更 / 外部回调均可起任务；入站触发带鉴权或签名，不裸奔 | **已落地**：webhook 验签 + 文件变更监听（含删除事件与防抖）已接入 |
| G2 | **进化评估集** | `eval/golden/skills.json` 离线样例 + `eval/runner.py` 判定器，纳入 pytest 闸门；`include_model` 时按当前配置构造真实 provider 跑端到端样例并断言，mock/无端点环境记跳过，离线闸门不发网络请求 | 沉淀标准任务样例与成败判定；技能/提示词迭代前先跑评估集，退化即拦截 | **已落地**：离线规则样例 + 在线模型样例均可判定 |
| G3 | **摘要式上下文压缩** | `_compact` 保留 system + 最近消息、对齐 user 边界，早期 user 要点与助手结论并入 system 注记；`context_model_summary` 开关开启时，对被裁早期消息调 provider 浓缩成模型摘要，失败回落规则要点；默认关，不额外产生模型调用 | 截断前把早期结论/决策沉淀进工作记忆或长期记忆，再按窗口裁剪 | **已落地**：规则要点 + 可选模型摘要（默认关） |

### P1 — 安全与成本

| # | 缺口 | 现状证据 | 验收方向 |
| --- | --- | --- | --- |
| G4 | **凭据加密保管** | 审计参数字段递归脱敏；长期记忆落库前对 Bearer / api_key= / JWT / ak·sk 前缀模式打码；`.env.local` 600 权限且 gitignore 排除；仍无 vault/keyring 加密存储，落盘为明文 | 凭据不入日志/记忆/产物；敏感落盘可加密，或明确由平台密钥保管注入 | **部分落地**：出口脱敏已做；加密保管依赖平台 vault，运行时无主密钥来源不做假加密 |
| G5 | **工具结果缓存与细粒度限流** | 幂等只读（L0）工具结果已按参数进程内缓存（`tool_cache_enabled`/`tool_cache_ttl`，replay 绕过缓存）；registry 按通道信号量并发上限 `tool_channel_concurrency`（0 不限），超限排队；另挂 `RateLimiter` 按通道 60 秒滑窗速率上限 `tool_rate_limits`，超限返回 `rate limited`；幂等工具遇瞬时错误按 `tool_retries` 指数退避重试；另有全局 `Semaphore(3)` | 幂等工具重复调用命中缓存；按通道设并发/速率上限，超限排队或拒绝而非压垮进程；瞬时失败可退避重试 | **已落地**：结果缓存 + 按通道并发闸 + 速率限流 + 工具瞬时重试已做 |
| G6 | **provider 异常兜底** | `_complete` 已按 `provider_retries` 重试（401/403/404 不重试）；`provider_fallback=mock` 切内置 mock；配置 `model_fallback` 块装配备用真实 provider；主端点失败切备用并带 `degraded` 标注；默认不降级 | 模型端点失败按策略重试/降级到备用 provider 或 mock，并在结果中显式标注降级 | **已落地**：重试 + mock 兜底 + 备用真实 provider 通道 + 降级标注 |

### P2 — 感知与协作

| # | 缺口 | 现状证据 | 验收方向 |
| --- | --- | --- | --- |
| G7 | **多模态输入面窄** | 仅图片；音频/摄像头/屏录在源码零命中 | 接入语音转写、屏幕/图像理解后，内容能进入规划上下文 |
| G8 | **系统感知专用工具** | 只读 `system.status`（标准库采集 CPU 核数/内存/磁盘/本进程 PID/线程/RSS，L0 幂等，不依赖 shell 与宿主桥）；`system.net_check` 异步 TCP 连通探测（host/port/timeout，建连即关，不可达返回原因） | 注册表提供只读系统指标工具，供规划判断资源与可行性 | **已落地**：主机资源指标 + 网络连通/进程级探测已做 |
| G9 | **跨 Agent 协议** | 仅本地 `parallel` 进程内拆分；无 A2A 协议、跨进程协作与共享记忆契约 | 多 Agent 间有统一消息/结果契约，可跨进程编排并回写共享记忆 |
| G10 | **桌面 GUI 视觉后端** | 浏览器 9 + 桌面 8 兜底工具目录在；执行依赖宿主桥，沙箱内真实截图/键鼠后端未适配 | 经宿主通道完成一次非 Web 客户端的截图、定位、点击；协议不符时有明确降级提示 |

## 3. 平台侧底座（运行时不承担，需向宿主确认归属）

| # | 事项 | 说明 |
| --- | --- | --- |
| P1 | 中文输入法 / 字体 | 渲染中文、输入中文的隐形基础，运行时不安装系统组件 |
| P2 | 宿主 ↔ 云电脑文件互传与剪贴板互通 | 产物回传、素材上传的通道在平台层 |
| P3 | 断连后会话保留 | 云电脑断连/切后台/被回收后的会话与进程保留策略由平台决定 |
| P4 | 快照 / 回滚 | 整机状态快照与恢复能力，非运行时职责 |
| P5 | 资源配额与端口现实 | CPU / 内存 / GPU 配额、端口占用、出口代理规则 |

## 4. 建议优先级

运行时可落地项已收口：G1 webhook 验签 + 文件变更监听（含删除事件与防抖）、G2 离线规则样例 + 在线模型质量回归、G3 规则式早期要点 + 可选模型摘要（默认关）、G4 记忆出口脱敏、G5 只读工具结果缓存 + 按通道并发闸、G6 provider 重试 + mock 兜底 + 备用真实 provider 通道、G8 主机资源指标 + 网络/进程级探测，均已落地并纳入 pytest 闸门。剩余归属平台/宿主侧：G4 平台侧加密 vault（运行时无主密钥来源，不做假加密）、G7 音频/摄像头/屏录、G9 A2A 跨进程协议、G10 桌面 GUI 视觉后端，以及 P1–P5 平台底座；这些按宿主能力开放情况再排期。
