# 技能树回写（Tree Ledger）

把每次 Agent 运行的证据写回「网络安全技能树 · 29 类依赖图谱」的 A/P/B/C，并在跨类共现时自动生成依赖边。目标是让图谱跟着真实跑过的任务长，而不是靠人手工标注。

## 等级口径（只升不降）

| 等级 | 含义 | 判定 |
|---|---|---|
| A 已验 | 本机可复现 | 有执行类工具成功（`script.run_shell` / `script.run_python` / `host.sandbox_execute_bash` / `host.sandbox_execute_code` / `host.browser_gui_execute_action` / `toolbox.run`）且会话收敛 |
| P 部分验 | 有分析无执行 | 分类命中但只做了推理/检索 |
| B 待验 | 未自验 | 会话 failed/incomplete，或执行类工具全失败 |
| C 边界 | 被拦/无结论 | 权限拒绝且没有任何执行 |

- 证据只能抬高等级：A 记录记 `ok`，B/C 记录记 `fail` 并置 `regressed`，但不会把已验类别打回去。
- 分类不中的会话不写账，避免噪声污染图谱。

## 分类

- 词法来自 `data/sec-tree-v55.json`（29 类 / 93 子类 / 2667 词条）+ `data/sec-tree-alias.json`（每类 6–15 个中英别名）。
- 权重：显式 `#N` +6、类名分词 +3、短名/子类 +2、词条 +1、别名 +3；阈值 `MIN_SCORE=3.0`，取 top3。
- 自动边要求两边都强命中（`EDGE_MIN_SCORE=5.0`），且不在 39 条手绘边里。

## 落点

- 证据台账：`<workspace>/tree/evidence.jsonl`（一行一次会话：ts/session/task/level/status/cats/tools_ok/tools_fail/denied/spans/note）。
- 状态每次由证据重算，不手改：`core/skilltree.py` 的 `state()`。
- 运行钩子：`Runtime.start_run()` 的 `finally` 调 `ingest_tree(session_id)`，异常只打 warning，不影响任务。
- 幂等：同一会话重复回写直接跳过。

## 接口

- CLI：`doubao-agent tree status | ingest --session ID [--scan --limit N] | edges | export --out FILE`
- HTTP：`GET /api/tree`、`POST /api/tree/ingest`、`POST /api/tree/scan`、`GET /api/tree/export`
- 控制台「图谱」视图：29 类按 T0–T4 分层展示等级与 ok/fail，右侧列有证据类别、自动边、导出。

## 导出格式

与图谱页同构：`{meta, D, EV, edges, auto_edges}`，`EV` 每类带 `lv` 与 `note`，可直接替换页面 meta 做本地预览；**不回写 CDN**（外发动作需确认）。

## 实测

- `3b0893ee067a`（UPX 加壳 + 逆向脱壳）：A，命中 #4(9.0)、#5(9.0)，自动长出 `#4 → #5`，手绘边里没有这条。
- `cd2d7d7e6024`（失败反馈）：B，只命中 #5，给 #5 记一次 fail，等级保持 A。
- `021a9f17694d` / `bae46920ccef`：A，命中 #4。

## 已知边界

- 别名表是人工补的，冷门说法仍可能漏召回；漏召回只影响记账，不影响任务。
- 自动边目前只做「同任务共现」，不把「工具→类别」当边（噪声太大）。
- 图谱 `EV` 的 note 直接取会话结论摘要前 220 字，不做二次改写。
