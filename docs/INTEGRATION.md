# 旧框架能力整合台账（2026-11-26）

> 来源：`~/work/archive/20261126-de-os/` 里 v1.62 的 24 个脚本 + 1 个清单（25 件）。
> 判据：新引擎已有对应实现且契约更完整的，不迁（避免两套口径打架）；新引擎确实没有的，迁成可执行技能并补测试。
> 证据字段全部指向本仓库实际文件，未推测。

## 一、已整合（2 项，本轮新增）

| 新技能 | 源脚本 | 契约 | 验证 |
| --- | --- | --- | --- |
| `output-density` | `check-output-density.py`（128 行） | stdin JSON → stdout JSON，输入 `path`/`text`，阈值可覆盖 | `tests/test_integrated_skills.py` 4 条；实跑空稿判不通过、实稿判通过 |
| `run-stats` | `stats.py`（139 行） | 读会话库 + `audit.jsonl`，出任务级成功率/步数/耗时/降级/工具量 | 同测试 2 条；实数据跑出 16 任务、成功率 0.625 |

装配结果：`python scripts/sync_skills.py --apply` 后 `doubao-agent skills` 显示 212 个技能、5 个可执行（原 208 / 3）。

## 二、不迁（19 项，逐项给对应件）

| 源脚本 | 新引擎对应件 | 判定理由 |
| --- | --- | --- |
| `reasoning-engine.py` | `core/planner.py` + `skills/task-triage` | task-triage 的 SKILL.md 已注明"移植 v1.62 reasoning-engine 的委派分级" |
| `task-router.py` | `skills/task-triage` + Planner | 轻/重分级已由 task-triage 承担 |
| `tool-scaffold.py` | `registry.py` + `core/permissions.py` | 受控工具登记与风险分级已内建 |
| `checkpoint.py` | `core/session.py` 的 `checkpoints` 表 | 断点表已在会话库里 |
| `memory-rag.py` | `core/longterm.py` + `core/memory.py` | 长期记忆 + 召回已落库，向量检索口径不同但能力重叠 |
| `start-recall.py` | `Agent` 启动召回（`core/agent.py`） | 启动注入已内建 |
| `warm-start.py` | `core/working.py` + `core/memory.py` | 热记忆由工作记忆承担 |
| `memory-graph.py` | `core/longterm.py` 三表 | 节点关系有表可查，无需独立图 CLI |
| `task-closeout.py` | `core/session.py` + `core/journal.py` | 收尾改为会话 finish + 日志，台账机制不再存在 |
| `backfill_type.py` | —— | 台账已不存在，无字段可补 |
| `stats.py` | 已整合为 `run-stats` | 见上表 |
| `health-check.sh` | `pytest` 全量 + `docs/REVIEW.md` | 旧体检口径依赖 assets/skills.yaml 结构 |
| `system-status.sh` | `system.status` 工具 | 只读系统指标已是注册工具 |
| `regen-skill-index.py` | `scripts/sync_skills.py` + `.manifest.json` | 技能索引改为清单文件，不再生成 md |
| `gen-capability-grades.py` | 旧体系专属 | 依赖 capability-assurance.md 矩阵，新引擎按 risk 分级 |
| `check-repro-numbers.py` | —— | 依赖 `repro-scan-config.json`，该配置在 v1.62 恢复时丢失且 git 历史无记录，无法忠实还原（不编造阈值） |
| `competitor-pricing.py` | —— | 依赖 `safe-web-fetch.py` 整条链，且只覆盖 Dify 一个站；迁入成本高于收益 |
| `safe-web-fetch.py` | 宿主 MCP 浏览器/抓取通道 | 抓取由平台通道承担 |
| `policy-brief.py` / `rag-daemon.py` / `pipeline.py` / `trace.py` | —— | 旧体系常驻件/待核件，未读通不迁 |

## 三、仍差的两处（如实记录）

1. **复现数字扫描**没迁：配置丢失，需要你给一份基准数字清单才能重建规则。
2. **竞品定价结构化**没迁：要接抓取通道并重做解析，属于新功能而非迁移，等你点名再做。
