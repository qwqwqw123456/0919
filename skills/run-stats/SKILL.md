---
name: run-stats
description: 任务级运行统计——跨会话成功率、步数、耗时分布、降级次数、工具调用量（补 metrics 只到 span 层的缺口）
tags: [可观测, 统计, 台账]
version: 0.1
runtime: python
entry: main.py
risk: L0
timeout: 30
input_schema:
  type: object
  properties:
    db_path:
      type: string
      description: 会话库路径，缺省读 AGENT_DB_PATH 环境变量，再退 ~/agent-state/agent.db
    runs_dir:
      type: string
      description: 审计目录，缺省读 AGENT_RUNS_DIR 环境变量，再退 ~/agent-state/runs
    recent:
      type: integer
      description: 只看最近 N 个会话（按更新时间倒序），0 表示全部
      default: 0
---

# run-stats

`metrics` 只聚合到 span 层（工具调用耗时/失败），看不到"任务成没成"。本技能补这一层：

- 任务数、状态分布（completed / failed / running）
- 成功率 = completed ÷（completed + failed），running 单列不计入分母
- 平均步数、耗时 p50 / p95
- 降级会话数（审计里出现 `degraded` 事件）
- 工具调用总量（审计 `tool_call`）+ 每任务均值

只读 SQLite 与 `audit.jsonl`，不写库、不改文件。返回里带 `sources` 说明实际读了哪些路径，缺库时如实报不存在。
