#!/bin/bash
set -e

echo "=== MultiTalk Container Startup ==="

# Restore application files if needed (handles volume mount scenario)
if [ ! -f "/workspace/Wan2gp-Multitalk/multitalk_app.py" ]; then
    echo "Restoring application files..."
    mkdir -p /workspace/Wan2gp-Multitalk
    rsync -a /opt/wan2gp_source/ /workspace/Wan2gp-Multitalk/
    echo "Application files restored"
else
    echo "Application files already present"
fi

# Add Warp terminal integration if not already present
if ! grep -q "SourcedRcFileForWarp" ~/.bashrc 2>/dev/null; then
    echo "Adding Warp terminal integration to ~/.bashrc..."
    echo 'printf '"'"'\eP$f{"hook": "SourcedRcFileForWarp", "value": { "shell": "bash"}}\x9c'"'"'' >> ~/.bashrc
    echo "Warp terminal integration added"
fi

# Change to application directory
cd /workspace/Wan2gp-Multitalk

# Check necessary directories
echo "📁 Checking directory structure..."
mkdir -p ckpts
mkdir -p weights
mkdir -p output
mkdir -p temp

# Check model files
if [ ! -f "ckpts/multitalk-wan2gp-14B.pth" ]; then
    echo "⚠️  WARNING: MultiTalk model file not found"
    echo "Please place model file at: ckpts/multitalk-wan2gp-14B.pth"
fi

# Check Wav2Vec2 model
if [ ! -d "ckpts/chinese-wav2vec2-base" ] && [ ! -d "ckpts/wav2vec" ]; then
    echo "⚠️  WARNING: Wav2Vec2 model not found"
    echo "Please place Wav2Vec2 model at: ckpts/chinese-wav2vec2-base or ckpts/wav2vec"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU, can be modified as needed
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Start our application in the background
echo "Starting MultiTalk application in background..."

# Gradio runs on localhost:7860 for nginx to proxy
SERVER_NAME="127.0.0.1"
SERVER_PORT="7860"

echo "Starting MultiTalk on $SERVER_NAME:$SERVER_PORT"
nohup python3 multitalk_app.py --server-name $SERVER_NAME --server-port $SERVER_PORT --no-browser > /workspace/multitalk.log 2>&1 &
echo "MultiTalk started on port $SERVER_PORT, logs in /workspace/multitalk.log"
echo ""
echo "🚀 Application directly accessible on port $SERVER_PORT"
echo ""

echo "Starting RunPod services..."
if [ -f "/start.sh" ]; then
    /start.sh
else
    echo "No /start.sh found, keeping container alive by monitoring log for debugging."
    tail -f /workspace/multitalk.log
fi