---
name: task-triage
description: 任务轻重分级：给一句话任务，判轻/标准/重度，给出步骤预算、预估耗时与 token 档，并列出触发升降级的信号
tags: [编排, 分级, 预算, 计划]
version: 0.1
runtime: python
entry: main.py
risk: L0
timeout: 20
input_schema:
  type: object
  properties:
    task:
      type: string
      description: 用户任务原话，一句话或一段话
  required: [task]
---

# task-triage

接活第一步先分级，再决定花多少步、要不要拆分。判定口径移植自 v1.62 轻/重双引擎的委派分级与权重表，去掉与本运行时无关的部分。

输出四件事：

- **分级**：L1-轻量 / L2-标准 / L3-委派（需要拆子任务）
- **步骤预算**：轻 ≤5 步 / 标准 ≤12 步 / 重度 ≤20 步
- **预估**：耗时与 token 档（提示是否要分片）
- **信号**：命中了哪几个实体、多主线词、疑问词，以及依据的意图权重

判定规则：

- 实体 ≥3 或 多主线信号 ≥3 或 疑问词 ≥4 → L3
- 实体 ≥2 或（多主线 + 疑问词）≥2 → L2
- 其余 → L1
- 意图权重 ≥3（如逆向/调研类）直接把复杂度抬到重度
