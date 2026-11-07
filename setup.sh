#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

echo "1. 更新套件清單並修復依賴..."
apt-get update --fix-missing
apt-get upgrade -y

echo "2. 安裝基礎編譯與網路工具..."
apt-get install -y \
    curl \
    wget \
    git \
    sudo \
    ca-certificates \
    gnupg \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    net-tools \
    iputils-ping

echo "3. 檢查並安裝 Node.js LTS (v20)..."
if ! command -v node >/dev/null 2>&1 || [[ $(node -v | cut -d'.' -f1 | tr -d 'v') -lt 18 ]]; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi

echo "4. 檢查並安裝 Claude Code CLI..."
if ! command -v claude >/dev/null 2>&1; then
    npm install -g @anthropic-ai/claude-code
fi

echo "5. 安裝 Aider CLI 與股票數據分析庫..."
pip3 install --upgrade pip
pip3 install aider-chat yfinance pandas numpy requests matplotlib

echo "6. 檢查並安裝 NordVPN..."
if ! command -v nordvpn >/dev/null 2>&1; then
    sh <(curl -sSf https://downloads.nordcdn.com/apps/linux/install.sh) || true
fi

echo "7. 配置 NordVPN 內網與 SSH 白名單..."
nordvpn whitelist add subnet 192.168.2.0/24 2>/dev/null || true
nordvpn whitelist add port 22 2>/dev/null || true

echo "8. 預埋多 AI 供應商環境變數架構..."
if ! grep -q "AI 供應商設定架構" ~/.bashrc; then
cat << 'EOF' >> ~/.bashrc

# ================= AI 供應商設定架構 =================
# 1. 官方 Anthropic Claude (填入哥哥提供的 API Key)
export ANTHROPIC_API_KEY=""

# 2. DeepSeek (備用方案，香港直連)
export DEEPSEEK_API_KEY=""

# 3. 自訂中轉平台 (如需要使用 OpenRouter 時取消註解)
# export ANTHROPIC_BASE_URL="https://openrouter.ai/api/v1"
# ===================================================
EOF
fi

source ~/.bashrc
echo "✅ 所有環境與多 AI 設定已全部安裝並配置完成！"
