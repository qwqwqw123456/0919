# 自进化层（evolve）

本文件记录「技能自进化」这一层怎么来的、做了什么、还差什么。机制源自 Hermes Agent v0.23.1 的
curator（skill 用量台账 + 生命周期 + 配方沉淀），按本仓库既有结构改写，不引入新依赖。

## 为什么需要

v1.62 之前的技能只增不减：装进去的技能没有任何使用记录，既不知道哪个真被用过，也没有
"多久没用"的时间戳，于是"该留、该调、该退"全靠人记。结果是技能库越来越厚、命中率越来越低。

## 五件套对照

| Hermes 机制 | 本仓库落点 | 状态 |
| --- | --- | --- |
| 技能用量四计数（use / view / patch / 最后使用） | `core/evolution.py` 的 `EvolutionStore`，表 `skill_usage` | 已落地 |
| 生命周期 active → stale(14d) → archived(30d) | `evolve audit`（默认预演，`--apply` 才改） | 已落地 |
| pin 保护 + 复用即复活 | `evolve pin/unpin`；`use`/`patch` 命中归档项自动回 active | 已落地 |
| 程序性记忆（把验证过的流程写成技能） | `evolve distill <配方号>` → `skills/_drafts/<名>/SKILL.md` | 已落地（草案） |
| 后台 nudge / LLM 合并审查 | — | 未接（见下） |

## 数据落点

- 表 `skill_usage`，与 `recipes`、`knowledge` 等共用 `agent.db`，但彼此独立，不改任何既有表结构。
- 只写状态位，不动文件：归档不等于删除，脚本路径引用不会断。
- 配置在 `agent.yaml` 的 `evolution.enabled / stale_days / archive_days`，也可用 `AGENT_EVOLUTION=0` 整体关掉。

## 记账时机

| 事件 | 触发点 | 含义 |
| --- | --- | --- |
| view | `SkillManager.knowledge_context` 注入正文、`skill.read` | 技能被看到，不等于被用 |
| use | `SkillSource._run` 执行完（成败都记） | 技能真的跑过 |

记账是旁路：`EvolutionStore.record` 抛异常会被吞掉，不拖垮技能执行（有测试覆盖）。

## 命令

```bash
doubao-agent evolve status            # 用量排行 + 冷技能榜
doubao-agent evolve audit             # 预演流转
doubao-agent evolve audit --apply     # 执行流转（只改状态位）
doubao-agent evolve pin <技能名>       # 钉住，不被自动流转
doubao-agent evolve distill <配方号>   # 配方 → 技能草案（写入 _drafts，不自动加载）
doubao-agent evolve report            # 生态周报
```

## 边界

- 草案进 `skills/_drafts/`，扫描只认 `skills/*/SKILL.md`，所以草案天然不会被加载；要生效必须人工
  `skills install`。自进化不自动改线上技能，避免"跑一次就学会一个坏习惯"。
- `distill` 生成的步骤会标注曾失败的步（`需人工复核`），不把失败步骤洗成"已验证"。

## 还没接线

- **nudge**：Hermes 在会话里按轮数触发"要不要记一条"，本仓库没有常驻对话循环，等价能力目前只能靠
  调度任务或任务收尾人工触发（`evolve report` 可作为定时任务的固定动作）。
- **patch 计数**：`skills install` 只允许新装、拒绝覆盖，所以没有"改已有技能"的入口，`patch_count`
  暂时恒为 0。
- **成本归因**：`UsageMeter` 只记进程级调用与 token，没有按技能切分，"哪个技能烧钱"还查不出来。
