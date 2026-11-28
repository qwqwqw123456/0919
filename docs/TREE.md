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
- 控制台「图谱」视图：原图谱页整页嵌入（`static/tree.html?embed=1`），数据优先取本地 `/api/tree/export`，失败才回退远端 pointer；自动边在页面上画成青色虚线。

## 新鲜度（P 候选队列）

- `doubao-agent tree fresh --days 30 --limit 50 [--file 离线.json]`
- `POST /api/tree/fresh`，工作台「抓 CVE → P」按钮同源。
- 源：CISA KEV 公开 JSON（无需 key）。只入 P 队列，`fresh.jsonl` 按 CVE 去重；转 A 必须本机沙箱复现，不做自动升 A。
- 噪声门槛：分类分 ≥5 才入队；纯数字词条（年份）不进词法，英文别名按词边界匹配，避免 `proper` 命中 `rop`、`cve-2026-...` 命中 `2026`。
- 实测：45 天窗口、limit 20 → 入队 5 条（MiCollab 命令注入 / MONAI 路径穿越 / Barracuda 命令注入 / Apple 认证缺陷 / Collabora 命令注入）；同日 `XML Injection` 类弱匹配被门槛挡掉。

## 边的门槛与可解释

- 共现边：同一对类别 ≥2 次、且跨 ≥2 个会话才画线（单次任务不连线，避免刷图）。每条边带 `kind="cooccur"`、`reason`（命中词）、`explain`（证据会话），页面 `path` 上有 `<title>` 提示。
- 工具边：模板 → 类别需 ≥3 次且跨 ≥2 会话，`kind="tool"`，只在 A 级任务里计；`tree tools` 可查，页面暂不画端点。

## 快照与对比

- 每次 `export`/回写会自动落 `tree/snapshots/<时间>.json`（内容不变则不写，保留最近 60 份）。
- `doubao-agent tree diff` 输出：证据增量、新边/消失边、新工具边、等级变化；`GET /api/tree/diff` 同源，工作台图谱条显示「对比上份快照」。
- 回归告警：`tree status` 打「⚠ 回归」行，`/api/tree` 的 `regressed` 计数在工作台显示橙标。

## 新鲜度转 A 的入口

- `doubao-agent tree triage [--id CVE-x]` 生成 `tree/triage-YYYYMMDD.md`：资产确认 → 版本核对 → 配置核对 → 隔离验证 → 回写，全部只读步骤，不含利用载荷；候选状态改 `triaged`。
- `POST /api/tree/triage` 返回 markdown 全文。

## 每日抓取

- 服务内定时：`serve` 启动后每 20 小时判断一次，到期抓 KEV（`tree/fresh-last.json` 记账），失败只打日志。
- 系统兜底：`scripts/tree-fresh-daily.sh`，crontab 每 04:17 跑一次，日志 `tree/fresh-cron.log`。系统盘重启会丢 crontab，重装命令：
  `(crontab -l; echo "17 4 * * * /home/user/Doubao/chats/1413030229010434/doubao-agent/scripts/tree-fresh-daily.sh") | crontab -`

## 词条检索（Agent 索引）

- `doubao-agent tree lookup 提权 --limit 10` 与 `GET /api/tree/lookup?q=&limit=`：在 2667 条索引里按名称/正文子串查找，名称命中优先，返回类/子类/状态/正文摘要。
- 工作台图谱条右侧输入框同源，输入即查（260ms 防抖，最多 8 条），结果面板显示类与子类、正文前 120 字。

## 发布通道（已接通）

- 指针仓库：`qwqwqw123456/sec-tree-go` 的 `pointer.json`（页面里 `PTR` 三源：jsDelivr ×2 + gh-proxy 直读 raw，按 `t` 取最新）。
- 发布四步：
  1. `tree export --out tree-export.json`；
  2. 上传 JSON 得到 `meta` 地址（CDN 直链）；
  3. `python3 scripts/publish_cdn.py --meta <meta地址> --out tree-shell.html` 生成独立壳（`shellV=s15x`，首屏先问指针，避免吃旧缓存；下载按钮指回原始 CDN 直链）；
  4. 上传壳得到 `url`，改 `pointer.json` 的 `url/shellV/meta/v/t`，push 到 `main`。
- 旧入口 `aka.doubaocdn.com/s/J1fvwP4Q3v` 会因 `shellV` 变化自动 `location.replace` 到新壳，不用换链接；实测两个入口都落到同一新壳、29 节点、A 25。
- 缓存：jsDelivr `@main` 可能滞后数小时，三源按 `t` 比新旧，gh-proxy 直读 raw 通常 1–2 分钟内生效。

## 工具维度

- 证据里的成功/失败工具分别聚合进 `state.cats[no].tools` 与 `tools_fail`（tool → 次数），只做软关联，不生成依赖边；工具挂节点前必须先按 `docs/TOOL-ATTRIBUTION.md` 的命令模板归属规范归一。

## 词条库与 Agent 索引

- 两份原始 JSON 已落到仓库：`data/sec-words.json`（2667 条词条，136 KB）、`data/sec-agent-index.json`（类/子类/词条三级 + status + 六段正文，1.4 MB，可直接做 RAG）。
- 本地直出：`GET /api/tree/download?which=words|agent`，图谱页两个下载按钮已指向本地，不再依赖远端 CDN。
- 索引与 `data/sec-tree-v55.json` 的差别：v55 只给分类与边，agent 索引带每个词条的正文，可用于后续检索/生成，不参与分类打分。

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
