#!/usr/bin/env bash
# 每日抓一次 CISA KEV 进 P 队列（无网络时失败退出，不影响其他任务）
set -u
REPO="/home/user/Doubao/chats/1413030229010434/doubao-agent"
LOG="/home/user/agent-state/tree/fresh-cron.log"
cd "$REPO" || exit 0
PYTHONPATH=src python3 -m doubao_agent.cli tree fresh --days 1 --limit 40 >> "$LOG" 2>&1
