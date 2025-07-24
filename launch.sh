#!/bin/bash

# 智能果园检测系统启动脚本

echo "🍊 智能果园检测系统启动"
echo "=========================="

# 激活conda环境
source /home/hllqk/miniconda3/etc/profile.d/conda.sh
conda activate deeplearn

# 检查环境
echo "✅ 使用conda环境: deeplearn"
echo "✅ Python版本: $(python --version)"

# 启动系统
echo "🚀 启动Streamlit应用..."
echo "📱 系统将在浏览器中打开"
echo "🛑 按 Ctrl+C 停止系统"
echo "=========================="

streamlit run main.py
