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

# Change to application directory
cd /workspace/Wan2gp-Multitalk

# Check necessary directories
echo "📁 Checking directory structure..."

# Check model files
echo "🔍 Checking Vace MultiTalk model files..."
VACE_MULTITALK_MODEL="ckpts/Wan14BT2VFusioniX_quanto_bf16_int8.safetensors"
VAE_MODEL="ckpts/Wan2.1_VAE.safetensors"

if [ ! -f "$VACE_MULTITALK_MODEL" ]; then
    echo "⚠️  WARNING: Vace MultiTalk FusioniX model not found"
    echo "Please place model file at: $VACE_MULTITALK_MODEL"
else
    echo "✅ Vace MultiTalk FusioniX model found"
fi

if [ ! -f "$VAE_MODEL" ]; then
    echo "⚠️  WARNING: VAE model not found"
    echo "Please place model file at: $VAE_MODEL"
else
    echo "✅ VAE model found"
fi

# Check MultiTalk module files
MULTITALK_MODULE="ckpts/fantasy_proj_model.safetensors"
VACE_MODULE="ckpts/wan2.1_Vace_14B_module_quanto_mbf16_int8.safetensors"
MULTITALK_CORE="ckpts/wan2.1_multitalk_14B_quanto_mbf16_int8.safetensors"

if [ ! -f "$MULTITALK_MODULE" ]; then
    echo "⚠️  WARNING: MultiTalk fantasy module not found"
    echo "Please place model file at: $MULTITALK_MODULE"
else
    echo "✅ MultiTalk fantasy module found"
fi

if [ ! -f "$VACE_MODULE" ]; then
    echo "⚠️  WARNING: Vace 14B module not found"
    echo "Please place model file at: $VACE_MODULE"
else
    echo "✅ Vace 14B module found"
fi

if [ ! -f "$MULTITALK_CORE" ]; then
    echo "⚠️  WARNING: MultiTalk 14B core module not found"
    echo "Please place model file at: $MULTITALK_CORE"
else
    echo "✅ MultiTalk 14B core module found"
fi

# Check Wav2Vec2 model
if [ ! -d "ckpts/chinese-wav2vec2-base" ] && [ ! -d "ckpts/wav2vec" ]; then
    echo "⚠️  WARNING: Wav2Vec2 model not found"
    echo "Please place Wav2Vec2 model at: ckpts/chinese-wav2vec2-base or ckpts/wav2vec"
else
    echo "✅ Wav2Vec2 model found"
fi

# Check text encoder models
TEXT_ENCODER_FILE="ckpts/umt5-xxl/models_t5_umt5-xxl-enc-quanto_int8.safetensors"
XLM_ROBERTA_DIR="ckpts/xlm-roberta-large"

if [ ! -f "$TEXT_ENCODER_FILE" ]; then
    echo "⚠️  WARNING: UMT5 text encoder not found"
    echo "Please place text encoder at: $TEXT_ENCODER_FILE"
else
    echo "✅ UMT5 text encoder found"
fi

if [ ! -d "$XLM_ROBERTA_DIR" ]; then
    echo "⚠️  WARNING: XLM-RoBERTa encoder not found"
    echo "Please place encoder at: $XLM_ROBERTA_DIR"
else
    echo "✅ XLM-RoBERTa encoder found"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU, can be modified as needed
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Start our application in the background
echo "Starting MultiTalk application in background..."

# Gradio runs on localhost:7860 for nginx to proxy
SERVER_NAME="0.0.0.0"
SERVER_PORT="7860"

echo "Starting MultiTalk on $SERVER_NAME:$SERVER_PORT"
# python3 multitalk_app.py --server-name 0.0.0.0 --server-port 7860 --no-browser
python3 multitalk_app.py --server-name $SERVER_NAME --server-port $SERVER_PORT --no-browser
