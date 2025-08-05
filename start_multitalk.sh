#!/bin/bash

# MultiTalk 精简版启动脚本

echo "🎭 MultiTalk 精简版启动脚本"
echo "================================"

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python3"
    exit 1
fi

# 检查必要的目录
echo "📁 检查目录结构..."
mkdir -p ckpts
mkdir -p weights
mkdir -p output
mkdir -p temp

# 检查模型文件
if [ ! -f "ckpts/multitalk-wan2gp-14B.pth" ]; then
    echo "⚠️  警告: 未找到 MultiTalk 模型文件"
    echo "请将模型文件放置在: ckpts/multitalk-wan2gp-14B.pth"
fi

# 检查 Wav2Vec2 模型
if [ ! -d "ckpts/chinese-wav2vec2-base" ] && [ ! -d "ckpts/wav2vec" ]; then
    echo "⚠️  警告: 未找到 Wav2Vec2 模型"
    echo "请将 Wav2Vec2 模型放置在: ckpts/chinese-wav2vec2-base 或 ckpts/wav2vec"
fi

# 检查 Kokoro TTS 模型（如果使用 TTS）
if [ ! -d "weights/Kokoro-82M" ]; then
    echo "⚠️  警告: 未找到 Kokoro TTS 模型"
    echo "如果需要使用 TTS 功能，请将模型放置在: weights/Kokoro-82M"
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一个 GPU，可根据需要修改
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo ""
echo "🚀 启动 MultiTalk 界面..."
echo "访问地址: http://localhost:7860"
echo ""

# 启动应用
python3 multitalk_app.py