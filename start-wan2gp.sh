#!/bin/bash
set -e

echo "=== Wan2gp-Multitalk Container Startup ==="

# Restore application files if needed (handles volume mount scenario)
if [ ! -f "/workspace/Wan2gp-Multitalk/wgp.py" ]; then
    echo "Restoring application files..."
    mkdir -p /workspace/Wan2gp-Multitalk
    rsync -a /opt/wan2gp_source/ /workspace/Wan2gp-Multitalk/
    echo "Application files restored"
else
    echo "Application files already present"
fi

# Start our application
echo "Starting Wan2gp-Multitalk application in background..."
cd /workspace/Wan2gp-Multitalk

# Make sure the code is up to date
# git checkout docker
git pull

# Gradio runs on all interfaces for external access
SERVER_NAME="0.0.0.0"
SERVER_PORT="7860"

echo "Starting Wan2gp-Multitalk on $SERVER_NAME:$SERVER_PORT"
nohup python3 wgp.py --server-name $SERVER_NAME --server-port $SERVER_PORT > /workspace/wan2gp.log 2>&1 &
echo "Wan2gp-Multitalk started on port $SERVER_PORT, logs in /workspace/wan2gp.log"
echo ""
echo "🚀 Application directly accessible on port 7860"
echo ""

echo "Starting RunPod services..."
if [ -f "/start.sh" ]; then
    # Start RunPod services in background to avoid blocking
    /start.sh &
    # Keep our application logs visible
    echo "RunPod services started in background"
    tail -f /workspace/wan2gp.log
else
    echo "No /start.sh found, keeping container alive by monitoring log for debugging."
    tail -f /workspace/wan2gp.log
fi 