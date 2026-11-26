# 工具与技能总账（2026-11-26 实盘核对，含旧框架整合）

> 口径：本文件所有数字来自本机实跑（`doubao-agent tools` / `doubao-agent skills` / `scripts/sync_skills.py`），
> 不是估计值。技能目录 = `/home/user/agent-state/skills`（`agent.yaml: skills_dir`）。

## 一、技能：212 个已装，全部可解析

按来源去重后装进技能目录，宿主 79MB 的只读技能库用**符号链接**接入，不复制、不占持久盘：

| 来源 | 路径 | 装入 | 说明 |
| --- | --- | --- | --- |
| 宿主平台 | `/runtime/skills`（`.skills` 指向此） | 123 | 平台自带，只读；pdf、word、ppt、sheet、lark-* 等 |
| 个人技能 | `~/.doubao/agent_mode/workspace/.user_skills` | 31 | 例如 ab-experiment-analysis、deep-research |
| 主权技能 | `~/.config/sovereign-skill/skills` | 9 | 13 个里有 5 个与上面同名，已跳过 |
| hermes | `~/.hermes/skills`（按分类嵌套） | 43 | apple/media/devops 等 |
| 仓库示例 | `<repo>/skills` | 6 | 原有 4 个 + 本轮整合 output-density、run-stats |

- 合计发现 217 个技能目录，**同名跳过 5 个**，装配 212 个，解析失败 0。
- 同名策略：按上表顺序先到先得（宿主版优先），避免"同名两份、说不清用哪份"。
- **可执行技能 5 个**：`pre-write-check`（L0 风险扫描）、`task-triage`（L0 分级）、`text-stats`（L0 统计）、`output-density`（L0 交付密度）、`run-stats`（L0 任务级统计）。旧框架整合裁决见 `docs/INTEGRATION.md`。
  其余 205 个是知识型技能，通过 `skill.read` 按需读正文，不进工具列表，所以不会把提示词撑爆。
- 抽查实跑：`skill.read(task-triage)` 396 字、`skill.read(pdf)` 10,124 字、`skill.read(security-threat-review)` 13,515 字，均返回正文与资源清单。

## 二、工具：注册表 25 个可用，0 个报错

| 来源 | 数量 | 风险 | 说明 |
| --- | --- | --- | --- |
| `host`（宿主 MCP） | 11 | L0–L2 | 浏览器/桌面/沙箱类；**比早期探到的 36 个缩水到 11 个**，宿主侧变化，不是本地配置问题 |
| `skill` | 6 | L0 | `skill.read` + 5 个可执行技能（新增 output-density、run-stats） |
| `a2a` | 3 | L0/L1 | `a2a.discover` / `a2a.send` / `a2a.cancel`，出站白名单默认回环 |
| `knowledge` | 2 | L0 | 检索 / 入库 |
| `script` | 2 | L1 | `script.run_python` / `script.run_shell` |
| `system` | 2 | L0 | `system.status` / `system.net_check` |
| `native` | 0（DOWN） | - | 宿主通用 REST 桥未配置，符合预期 |

旧框架那批白帽工具箱（54 项）本地已无副本：全盘搜过，只剩别会话里一个 `binary-reverse-engineering-skill.zip`。
现在的 208 个里已含安全类（security-threat-review、reverse-toolkit 等），但不是"54 项"那一套，不冒充。

## 三、重跑与维护

```bash
python scripts/sync_skills.py                     # 预览：装什么、跳过什么
python scripts/sync_skills.py --apply             # 建链接（幂等，已存在跳过）
python scripts/sync_skills.py --apply --repair    # 指向不对的链接重建
python scripts/sync_skills.py --apply --manifest /home/user/agent-state/skills/.manifest.json
doubao-agent --config agent.yaml skills          # 复核加载数与可执行数
doubao-agent --config agent.yaml sources         # 复核来源健康（skill UP 会报 N skills）
```

清单明细（每个技能名、来源、目录、是否可执行）落在 `agent-state/skills/.manifest.json`。

## 四、已知边界

- 宿主技能是只读的（root 所有），能读能用，不能改；要给平台技能打补丁得另存为个人技能。
- hermes 里 apple/imessage 类依赖 macOS，本机 Linux 只能读流程，跑不了。
- 知识型技能只吃 `skill.read`，不产生工具调用；想让某个知识技能变成"一句话就能跑"的工具，得给它补 entry 与 input_schema。
