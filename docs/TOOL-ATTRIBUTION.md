# 工具维度：命令模板归属规范

工具暂不作为图谱端点。证据只记 `tools_ok` / `tools_fail`，`state()` 汇总成 `cats[no].tools` 与 `cats[no].tools_fail`，用于「这类活常靠什么工具干成/干砸」，不生成任何边。

原因：工具直挂节点会按参数爆炸（`nmap -sV` / `nmap -sS -p-` 会变成两个节点），噪声边很快超过 39 条手绘边，图谱就失去可读性。要动端点，先按下面这份规范把命令归一成模板。

## 1. 模板是什么

模板 = 工具名 + 归一化后的命令主体，标识为 `<tool>#<bin>:<sub>`。

| 执行工具 | 取什么 | 示例 |
|---|---|---|
| `host.sandbox_execute_bash` / `script.run_shell` | 命令第一个可执行名 + 其后的子命令（最多两级），flag 与参数全丢 | `strings /tmp/a` → `#strings`；`upx -d x` → `#upx:-d`；`nmap -sV -p- h` → `#nmap` |
| `host.sandbox_execute_code` / `script.run_python` | 首个第三方 import 的包名；没有则入口函数/脚本名 | `import scapy` → `#python:scapy`；纯标准库 → `#python:stdlib` |
| `host.browser_gui_execute_action` | 动作动词 | `#browser:navigate` |
| `toolbox.run` | 工具注册名 | `#toolbox:upx` |
| 只分析类工具 | 不产模板 | 不计 |

归一化规则：去引号内容、去绝对路径与变量、去 `-`/`--` 开头的 flag、去重排序后只留主体。同一命令换路径/换目标不应产生新模板。

## 2. 允许成为端点前的四条门槛

1. 模板必须能绑定至少一个类别：该会话里这个类别 `score ≥ 5`（与自动边同门槛）。
2. 同一模板在同一类别累计 ≥ 3 次，且跨 ≥ 2 个会话，才允许画边；同会话重复只计 1 次。
3. 通用底座模板永不成为端点，只保留软关联：`bash/sh/python/curl/wget/git/pip/npm/ls/cat/grep/chmod/cp/mv/rm/tar/ssh/scp`。
4. 单会话最多贡献 2 条工具边，防止一次长任务把图刷乱。

## 3. 落库与验收

- 现在：`evidence.jsonl` 的 `tools_ok` / `tools_fail`；`state.cats[no].tools` / `tools_fail`。不出边，不导出工具节点。
- 未来：模板抽取结果先写 `tree/tool-templates.jsonl`（模板 id、类别、会话、次数），过完四条门槛再进 `auto_edges`，并带 `kind="tool"` 标记，与共现边分开统计。
- 验收标准：模板抽取覆盖 ≥80% 的执行类证据；抽样 20 条同命令换参数的记录，模板数不增长；在 3 个会话内产出工具边为 0（说明门槛真的在挡）。
