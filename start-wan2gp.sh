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

# Add Warp terminal integration if not already present
if ! grep -q "SourcedRcFileForWarp" ~/.bashrc 2>/dev/null; then
    echo "Adding Warp terminal integration to ~/.bashrc..."
    echo 'printf '"'"'\eP$f{"hook": "SourcedRcFileForWarp", "value": { "shell": "bash"}}\x9c'"'"'' >> ~/.bashrc
    echo "Warp terminal integration added"
fi

# Start our application in the background
echo "Starting Wan2gp-Multitalk application in background..."
cd /workspace/Wan2gp-Multitalk

# Direct access without authentication
SERVER_NAME="0.0.0.0"
SERVER_PORT="7860"

echo "Starting Wan2gp-Multitalk on $SERVER_NAME:$SERVER_PORT"
nohup python3 wgp.py --server-name $SERVER_NAME --server-port $SERVER_PORT > /workspace/wan2gp.log 2>&1 &
echo "Wan2gp-Multitalk started on port $SERVER_PORT, logs in /workspace/wan2gp.log"
echo ""
echo "🚀 Application available on port $SERVER_PORT"
echo ""

echo "Starting RunPod services..."
if [ -f "/start.sh" ]; then
    /start.sh
else
    echo "No /start.sh found, keeping container alive by monitoring log for debugging."
    tail -f /workspace/wan2gp.log
fi 