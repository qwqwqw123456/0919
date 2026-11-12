#!/usr/bin/env bash
# 换机 / 整机重建后恢复 doubao-agent 运行时。
# 幂等：可重复执行；已存在的 venv、配置、密钥文件不会被覆盖。
# 注意：本脚本不读取、不打包、不打印 .env.local 里的密钥；密钥需在新环境手动填写。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "未找到 python3，请先安装 Python >= 3.11" >&2
  exit 1
fi

# 1. 虚拟环境（已存在则复用）
if [ ! -d ".venv" ]; then
  echo "[1/4] 创建虚拟环境 .venv"
  "$PY" -m venv .venv
else
  echo "[1/4] 复用已有 .venv"
fi

# 2. 可编辑安装依赖（含 dev 测试依赖）
echo "[2/4] 安装依赖（editable, extras=dev）"
.venv/bin/pip install -e ".[dev]"

# 3. 配置文件（缺失才从模板复制，不覆盖本机已有配置）
if [ ! -f "agent.yaml" ]; then
  echo "[3/4] 从 agent.example.yaml 生成 agent.yaml"
  cp agent.example.yaml agent.yaml
else
  echo "[3/4] 保留已有 agent.yaml"
fi

# 4. 密钥占位提示（不代填、不读取已有文件内容）
echo "[4/4] 密钥检查"
if [ ! -f ".env.local" ]; then
  cat <<'EOF'
  缺少 .env.local。请手动写入模型密钥（该文件已被 .gitignore 排除，勿入库）：

    cat > .env.local <<'ENV'
    OPENAI_API_KEY='你的密钥'
    ENV

  其余可选变量见 agent.example.yaml 注释。
EOF
else
  echo "  已存在 .env.local，跳过（脚本不读取其内容）"
fi

# 自检：列出已装配工具，确认运行时可启动
echo "--- 自检 ---"
.venv/bin/doubao-agent tools

cat <<'EOF'

恢复完成。下一步：
  .venv/bin/doubao-agent run "任务描述"     # 跑一个任务
  .venv/bin/python -m pytest tests -q       # 跑回归
  .venv/bin/doubao-agent serve              # 起控制台（默认 8765）
EOF
