#!/bin/bash
set -e

# Load RDMA module if available
if lsmod | grep -q rdma_rxe; then
    echo "RDMA module already loaded"
else
    modprobe rdma_rxe 2>/dev/null || echo "Warning: Could not load rdma_rxe module"
fi

# Create Soft-RoCE device if it doesn't exist
if ! rdma link show | grep -q "eth0-rxe"; then
    # Find first available network interface
    IFACE=$(ip -o link show | awk -F': ' '{print $2}' | grep -v lo | head -1)
    if [ -n "$IFACE" ]; then
        echo "Creating Soft-RoCE device on $IFACE"
        rdma link add eth0-rxe type rxe netdev "$IFACE" 2>/dev/null || true
    fi
fi

# Show RDMA status
echo "RDMA devices:"
rdma link show 2>/dev/null || echo "No RDMA devices found"

# Set defaults
ROLE=${ROLE:-prefill}
MODEL_PATH=${MODEL_PATH:-/model}
PREFILL_HOST=${PREFILL_HOST:-127.0.0.1}
PREFILL_PORT=${PREFILL_PORT:-8050}
DECODE_HOST=${DECODE_HOST:-127.0.0.1}
DECODE_PORT=${DECODE_PORT:-8060}

echo "Starting PD server..."
echo "  Role: $ROLE"
echo "  Model: $MODEL_PATH"
echo "  Prefill: $PREFILL_HOST:$PREFILL_PORT"
echo "  Decode: $DECODE_HOST:$DECODE_PORT"

# Run the server
python3 -c "
import sys
sys.path.insert(0, '/opt/pd_disagg')

from pd_server_v2 import PDServerV2
from pd_config import PDConfig

config = PDConfig(
    role='$ROLE',
    model_path='$MODEL_PATH',
    prefill_host='$PREFILL_HOST',
    prefill_port=$PREFILL_PORT,
    decode_host='$DECODE_HOST',
    decode_port=$DECODE_PORT,
)

server = PDServerV2(config)
server.start()
"
