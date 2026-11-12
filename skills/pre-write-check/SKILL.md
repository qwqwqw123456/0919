---
name: pre-write-check
description: 写操作前的风险扫描：把"要做的事"先过一遍 D/S/P/A 四类（删除/对外发送/付费/权限迁移），识别否定语境避免误杀，命中即要求先确认再动手
tags: [护栏, 风险, 写操作, 确认]
version: 0.1
runtime: shell
entry: check.sh
risk: L0
timeout: 15
input_schema:
  type: object
  properties:
    text:
      type: string
      description: 你准备执行的操作描述，一句话说清"对什么、做什么"
  required: [text]
---

# pre-write-check

写操作动手前调用一次，把动作描述传进来。工具会判断这次动作是否命中高风险四类：

- **D-删除**：删除/覆盖/移动/git 破坏性操作/清空
- **S-对外发送**：发消息、邮件、群发、审批提交、发布
- **P-付费**：付款、下单、充值、兑换、开订阅
- **A-权限/迁移**：权限变更、数据迁移、批量导出上传

判定口径：

- 按标点与转折词（但是/不过/然而/否则）**先切句**，避免"前半句否定、后半句肯定"被整句豁免
- 风险词紧邻否定词（不要/禁止/避免/无需/勿…）时**豁免**该词，不算命中
- 命中 → 输出 `VERDICT=block` 与类别；未命中 → 输出 `VERDICT=pass` 并附一句 scope creep 自检提醒
- 退出码恒为 0：判定结果看文本，不看退出码（本运行时把非 0 当工具故障）

命中时不要继续写，先把对象、范围、可逆性摆给用户确认；确认后按 `safety-checklist` 技能留痕。
