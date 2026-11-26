---
name: output-density
description: 交付物密度检测——扫 markdown 成品里的占位符/数字/链接/bullet，拦住骨架式交付
tags: [质量闸, 交付校验, 文本]
version: 0.1
runtime: python
entry: main.py
risk: L0
timeout: 20
input_schema:
  type: object
  properties:
    path:
      type: string
      description: 要检查的 markdown 文件绝对路径（与 text 二选一）
    text:
      type: string
      description: 直接传入正文（与 path 二选一，适合交付前内存里校验）
    thresholds:
      type: object
      description: 阈值覆盖，缺省用默认值
      properties:
        placeholder_max: {type: integer, description: 允许的占位符上限, default: 3}
        min_digits: {type: integer, description: 具体数字下限, default: 5}
        min_links: {type: integer, description: URL 下限（仅长文校验）, default: 2}
        min_bullets: {type: integer, description: bullet 下限（仅长文校验）, default: 3}
      required: []
---

# output-density

交付前自检用：数占位符（TODO/TBD/待定/XXX/待补充/待完善/FIXME/占位）、具体数字、URL、bullet。

判定规则（与旧框架 `check-output-density.py` 一致）：
- 占位符超过阈值 → 告警（像骨架交付）
- 具体数字少于下限 → 告警（可能是空洞描述）
- 正文超过 500 字符时，URL 少于下限 → 告警（长文没有来源支撑）
- 正文超过 300 字符时，bullet 少于下限 → 告警（长文没有结构化）

返回 `ok`、四项计数、`warnings` 清单。纯本地，不联网，不改文件。
