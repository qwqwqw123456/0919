---
name: trae-daily-checkin
description: Trae 每日积分自动领取：读取本机 Trae 登录态，查询并调用官方签到接口领取每日积分，已签到自动跳过（幂等）
tags: [trae, 签到, 积分, 自动化]
version: 0.1
runtime: python
entry: main.py
risk: L1
timeout: 180
input_schema:
  type: object
  properties:
    action:
      type: string
      enum: [checkin, status]
      description: checkin 领取积分（默认）；status 只查询今日签到状态
    dry_run:
      type: boolean
      description: 只查状态，不领取
    storage:
      type: string
      description: 手动指定 storage.json 路径（一般不需要）
  required: []
---

# trae-daily-checkin

自动领取 Trae 客户端的每日签到积分：本机读取登录态 → 查询签到状态 → 未签到则调用官方接口领取。
幂等设计：今日已签到直接返回，同一天重复执行不会重复领取。

## 前置条件

- 必须运行在**已安装并登录过 Trae 客户端**的机器上（目前对接 Trae CN 的签到接口，
  同样适配 TRAE SOLO CN 等安装目录），脚本要从客户端的 `storage.json` 读取登录态。
- 远程容器 / 云电脑读不到你本机桌面端的登录态；在那种环境里请改用环境变量方式（见下）。

## 用法

在 agent 中作为技能调用：

```json
{"action": "checkin"}
```

直接命令行运行：

```bash
python3 skills/trae-daily-checkin/main.py            # 领取（已签到则跳过）
python3 skills/trae-daily-checkin/main.py --status   # 只查状态
python3 skills/trae-daily-checkin/main.py --dry-run  # 只查不领
```

成功输出示例：

```json
{
  "ok": true,
  "action": "checkin",
  "date": "2026-11-20",
  "source": "/Users/me/Library/Application Support/Trae CN/User/globalStorage/storage.json",
  "device_id_present": true,
  "checked_in": true,
  "credits": 150,
  "extra_credits": 50,
  "enable": true,
  "result": "claimed",
  "message": "签到成功"
}
```

`result` 取值：`claimed`（本次领取成功）、`already_checked_in`（今日已签到）、
`not_checked_in`（仅查询）、`disabled`（活动未启用）。
退出码：`0` 成功 / 已签到，`1` 接口失败，`2` 未找到登录态。

## 定时执行

脚本幂等，建议每天多个时间点兜底（电脑开机即补签）。

crontab（macOS / Linux）：

```bash
5 9,12,15,18,21 * * * cd /path/to/repo && python3 skills/trae-daily-checkin/main.py >> /tmp/trae-checkin.log 2>&1
```

Windows 任务计划程序：

```powershell
schtasks /Create /TN TraeDailyCheckin /TR "python C:\path\to\repo\skills\trae-daily-checkin\main.py" /SC DAILY /ST 09:00 /F
```

本仓库自带调度器（需 daemon 常驻）：

```bash
doubao-agent schedule add "5 9 * * *" "调用 trae-daily-checkin 技能领取 Trae 每日积分" --name trae-daily-checkin
```

## 登录态来源

按顺序查找（找到第一个有效的即用）：

1. 环境变量 `TRAE_TOKEN`（配合 `TRAE_DEVICE_ID`，适合读不到本地文件的远程环境）
2. `TRAE_STORAGE` 指定的 `storage.json`
3. 各平台默认安装目录：
   - macOS：`~/Library/Application Support/<应用名>/User/globalStorage/storage.json`
   - Windows：`%APPDATA%\<应用名>\User\globalStorage\storage.json`
   - Linux：`~/.config/<应用名>/User/globalStorage/storage.json`
   - 应用名依次尝试：`TRAE SOLO CN`、`Trae CN`、`TRAE SOLO`、`Trae`

设备标识（`x-device-id`，领取接口必需）来源：
`storage.json` 的 `iCubeAuthInfo://icube-dc:*` → `aha/TinyStorage` 中的
`aha.device.device_id` → `telemetry.devDeviceId` → `SHA-256(machineId)`。

可选环境变量：`TRAE_USER_ID`、`TRAE_USER_REGION`（对应请求头，缺失也能用）。

## 接口

- 状态：`POST https://api.trae.cn/trae/api/v2/ug/checkin_credits/status`
- 领取：`POST https://api.trae.cn/trae/api/v2/ug/checkin_credits/claim`
- 认证：`Authorization: Cloud-IDE-JWT <token>`；领取需 `x-device-id`

## 排错

| 现象 | 处理 |
| --- | --- |
| `未找到 Trae 本地登录态` | 确认本机装并登录过 Trae；或在目标机设置 `TRAE_TOKEN` / `TRAE_DEVICE_ID` |
| `code=9004` | 设备标识无效：打开 Trae 客户端刷新登录态后重试 |
| `code=1001` / `HTTP 401/403` | 登录态过期（token 约 14 天），打开 Trae 客户端重新登录 |
| `code=9074` | 服务端限流，脚本已自动退避重试；仍失败则稍后再跑 |
| `解密失败 / 完整性校验失败` | storage.json 加密格式变更或文件损坏，更新客户端后重试 |

## 安全与边界

- 令牌只在内存中使用，不打印、不写日志、不落盘；日志只应保存输出 JSON。
- 只访问 `api.trae.cn` 官方接口，不经过任何第三方。
- 仅用于**本人账号**的每日签到；请遵守 Trae 用户协议，不要用于批量账号或刷分。
- 签到积分额度（当前约 150/160 一天）与签到开关以服务端返回为准。
