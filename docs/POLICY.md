# 策略层（Policy）：网络通信 / 模型等待可配置

一句话：**超时、重试、退避、轮询、连接池不再散落在 10 个模块里写死，而是收成一份 `policy` 配置；改配置即提速，不用重启，也不用改代码。**

代码：`src/doubao_agent/core/netpolicy.py`（22 个可调项：`net` 12 项 + `model` 10 项）
配置：`agent.yaml` 的 `policy:` 段
命令：`doubao-agent policy show | set | preset | suggest | stats`
基准：`scripts/bench_policy.py`（本地 mock 端点，不联网、不花钱）

## 1. 两个族

| 族 | 管什么 | 关键项 |
|---|---|---|
| `net` | 一次网络请求怎么等、失败怎么重来 | `connect_timeout` `read_timeout` `retries` `backoff_base` `backoff_max` `jitter` `retry_status` `retry_on_timeout` `keepalive` `max_connections` `http2` |
| `model` | 一次模型调用怎么等、异步任务怎么轮询 | `timeout` `first_byte` `retries` `backoff_base` `backoff_max` `jitter` `poll_interval` `poll_max` `poll_timeout` |

三个预设：

- `turbo`：追延迟。建连 1.5s / 读 15s / 首字节 8s / 单请求重试 2 / 退避 0.08 起 / 轮询 0.4s / 连接池 40。
- `balanced`：默认，等于内置默认值。
- `careful`：追成功率。建连 6s / 读 90s / 首字节 60s / 重试 3 / 退避 0.6 起 / 轮询 2s。

优先级：**内置默认 < preset < `agent.yaml` 的 `policy` 段 < 环境变量 `AGENT_POLICY_<族>_<字段>`**
（例：`AGENT_POLICY_MODEL_RETRIES=3`）。`policy show` 会逐项标出值来自 `default` / `preset:x` / `agent.yaml` / `env`。

## 2. 怎么改

```bash
# 看现状（每项带来源）
doubao-agent policy show

# 单点改：立即生效，并写回 agent.yaml 的 policy 段
doubao-agent policy set model.retries=3 net.read_timeout=20

# 整档切
doubao-agent policy preset turbo

# 按实测延迟给建议（只建议，不替你改）
doubao-agent policy suggest
```

Agent 自身也可以用同一套 API 改策略：`Policy.set()` / `set_many()` / `preset()`；provider、Agent、HTTP 源、A2A 客户端每次调用都读 `snapshot()`，**运行中改完下一次请求就用新值**。

即时生效的项：超时、重试次数、退避与抖动、轮询间隔、重试状态码。
需要重建连接才生效的项（httpx 在构造客户端时固化）：`keepalive`、`max_connections`、`http2`。这三项改完重启一次。

## 3. 落到哪些模块

| 位置 | 改造前 | 改造后 |
|---|---|---|
| `providers/openai_compat.py` | `timeout=120.0` 写死；零重试 | 超时按 `model.httpx_timeout()`；429/5xx/连接错按 `net.retries` 重试，退避走策略；池大小/keepalive/http2 按 `net`；每次调用记录延迟 |
| `core/agent.py` `_complete()` | `provider_retries` + 线性 `retry_delay*(n+1)` | 有策略时用 `model.retries` + 指数退避 + 上限 + 抖动 |
| `core/a2a_client.py` `wait()` | `poll_interval=2.0` 写死 | 取 `model.poll_interval`，前 6 轮固定快查、之后放大到 `poll_max`；`poll_timeout` 作总时限 |
| `sources/http_api.py` | `timeout=60.0` 写死 | 每次请求按 `net` 超时 |
| `sources/bridge.py` | invoke 120s / 能力探测 10s 写死 | invoke 按 `model`、探测按 `net.read_timeout` |
| `core/model_channels.py` | `30.0` 写死 | 按 `net` 建连+读；`probe()` 可传策略 |
| `config.py` | 无配置路径字段 | 记 `config_path`，供策略层定位同一份 yaml |

兼容：`provider_retries` / `provider_retry_delay` 老字段仍在，没策略对象时按老逻辑跑；老 `agent.yaml` 不加 `policy` 段也能启动。

## 4. 提速证据（实跑）

`python3 scripts/bench_policy.py`，同一份代码只换 policy，打本地 mock 端点。原始输出见 `docs/policy-bench.txt`。

**① 端点卡死（请求发出后永不到）：多久才判死**

| 配置 | 单请求判死 | 3 个并发整批释放 |
|---|---|---|
| 基线（写死 120s，`read` 无独立上限） | 90.00s（服务端 90s 断连才被动结束；不断则 120s） | 90.14s |
| balanced（首字节 20s） | 20.01s | 20.16s |
| turbo（首字节 8s） | **8.01s** | **8.14s** |

**② 异步任务轮询（任务 1.2s 后完成）：多久发现完成**

| 配置 | 发现完成 | 轮询次数 |
|---|---|---|
| 基线（写死 2.0s） | 2.00s | 2 |
| balanced（1.0s） | 2.00s | 3 |
| turbo（0.4s） | **1.20s** | 4 |

**③ 抖动端点（每 3 个请求 1 个 503，共 12 次调用）**

| 配置 | 成功 | 失败 | 总耗时 |
|---|---|---|---|
| 基线（零重试） | 8/12 | 4 | 2.46s |
| balanced | **12/12** | 0 | 4.16s |
| turbo | **12/12** | 0 | 3.88s |

**④ 退避总量（同样 4 次尝试共等）：旧线性 1.20s → turbo 0.48s**

口径说明（不掺水）：

- ①② 是机制的实测比值，不是模型真的变快了：策略压的是「白等」。
- ③ 不是零成本。补 4 次失败多花约 1.4s 墙钟，换来的是 4 个本来拿不到的结果；一次失败的整任务重跑是分钟级，这笔账划算但不等于「变快」。
- ④ 的对比对象是改造前 `agent.py` 的线性退避公式；数字由两侧公式直接算出，未做网络实测。
- `turbo` 在弱网下更容易撞超时，适合内网/低延迟端点；公网不稳就 `balanced`，宁可慢也别断。

## 5. 一条顺手修掉的坑

卡死端点如果按原来「TransportError 一律重试」的写法，turbo（重试 2 次）会把 8s 的等待变成 24s——重试反而更慢。所以 `net.retry_on_timeout` 默认 **false**：超时意味着端点已经不可用，重试只是把等待翻倍；只对连接错误、429、5xx 重试。

## 6. 测试

`tests/test_policy.py`（12 条）：

- 默认值、yaml 覆盖 preset、环境变量优先级、非法值拒绝；
- `set()` 只动 `policy` 段、其它键原样保留、重新加载仍生效；
- 改文件后按 mtime 热重载、版本号自增；
- 退避指数+上限精确断言；
- Provider：无策略时 1 次 503 直接失败；turbo 时 3 次尝试穿透 2 次 503；
- Agent：`model.retries=2` 时确实调了 3 次；
- A2A：轮询间隔真吃 `model.poll_interval`。

回归：`PYTHONPATH=src pytest -q` → **360 passed, 1 failed**；唯一失败是 `tests/test_voice.py` 缺 `av` 模块（本机环境缺依赖，与本次改动无关）。
