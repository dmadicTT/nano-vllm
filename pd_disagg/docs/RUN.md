# How to Run PD Disaggregation

## Quick Start (Single Machine)

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    cmake \
    build-essential \
    linux-modules-extra-$(uname -r) \
    rdma-core \
    ibverbs-utils \
    perftest

# Python dependencies
pip install torch transformers numpy
```

### 1. Set Up Soft-RoCE (RDMA over Ethernet)

```bash
# Load kernel module
sudo modprobe rdma_rxe

# Create Soft-RoCE device on your network interface (e.g., eth0)
sudo rdma link add eth0-rxe type rxe netdev eth0

# Verify it's active
rdma link show
# Should show: eth0-rxe state ACTIVE physical_state LINK_UP

# Test bandwidth
# Terminal 1:
ib_write_bw

# Terminal 2:
ib_write_bw localhost
# Should show ~500 MB/s bandwidth
```

### 2. Build Mooncake

```bash
# Clone Mooncake
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake

# Build with CPU-only support (no CUDA)
mkdir build && cd build
cmake .. -DUSE_CUDA=OFF -DBUILD_TEST=OFF
make -j$(nproc)

# Verify libraries exist
ls mooncake-transfer-engine/src/libtransfer_engine.so
ls mooncake-integration/engine.cpython*.so
```

### 3. Download Model

```bash
# Download a small model for testing
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'Qwen/Qwen3-0.6B'
AutoTokenizer.from_pretrained(model_name).save_pretrained('/tmp/qwen3-0.6b')
AutoModelForCausalLM.from_pretrained(model_name).save_pretrained('/tmp/qwen3-0.6b')
"
```

### 4. Run End-to-End Test

```bash
cd pd_disagg/src

# Run the automated test (starts prefill + decode + client)
python3 run_pd_e2e_final.py
```

Expected output:
```
============================================================
PD Disaggregation End-to-End Test
============================================================
[1/5] Starting prefill server on port 8050...
[2/5] Starting decode server on port 8060...
[3/5] Sending prefill request...
  Request ID: req_1
  Prompt tokens: 7
  Block table: [0]
[4/5] Sending decode request with KV cache transfer...
  [Prefill] RDMA transfer to 127.0.0.1:xxxxx...
  [Prefill] RDMA send successful
  [Decode] KV cache received via RDMA for 1 blocks
  [Decode] Decode complete. Generated 20 tokens
[5/5] SUCCESS!
  Tokens generated: 20
```

## Manual Mode (Separate Terminals)

### Terminal 1: Prefill Server
```bash
cd pd_disagg/src
python3 -c "
from pd_server_v2 import PDServerV2
from pd_config import PDConfig

config = PDConfig(
    role='prefill',
    model_path='/tmp/qwen3-0.6b',
    prefill_host='127.0.0.1',
    prefill_port=8050,
    decode_host='127.0.0.1',
    decode_port=8060,
)
server = PDServerV2(config)
server.start()
"
```

### Terminal 2: Decode Server
```bash
cd pd_disagg/src
python3 -c "
from pd_server_v2 import PDServerV2
from pd_config import PDConfig

config = PDConfig(
    role='decode',
    model_path='/tmp/qwen3-0.6b',
    prefill_host='127.0.0.1',
    prefill_port=8050,
    decode_host='127.0.0.1',
    decode_port=8060,
)
server = PDServerV2(config)
server.start()
"
```

### Terminal 3: Client
```bash
cd pd_disagg/src
python3 test_client.py
```

## Docker Mode (Separate Containers)

### Build Image
```bash
cd pd_disagg/docker
docker build -t nano-vllm-pd .
```

### Run with Host Network (for RDMA)
```bash
# Prefill container
docker run -d --name prefill \
    --network host \
    --privileged \
    -v /tmp/qwen3-0.6b:/model \
    -e ROLE=prefill \
    -e PREFILL_HOST=127.0.0.1 \
    -e PREFILL_PORT=8050 \
    -e DECODE_HOST=127.0.0.1 \
    -e DECODE_PORT=8060 \
    nano-vllm-pd

# Decode container
docker run -d --name decode \
    --network host \
    --privileged \
    -v /tmp/qwen3-0.6b:/model \
    -e ROLE=decode \
    -e PREFILL_HOST=127.0.0.1 \
    -e PREFILL_PORT=8050 \
    -e DECODE_HOST=127.0.0.1 \
    -e DECODE_PORT=8060 \
    nano-vllm-pd
```

### Run with Separate Networks (Multi-Node)
```bash
# Create network
docker network create pd-network

# Prefill container (Node 1)
docker run -d --name prefill \
    --network pd-network \
    --privileged \
    -v /tmp/qwen3-0.6b:/model \
    -e ROLE=prefill \
    -e PREFILL_HOST=prefill \
    -e PREFILL_PORT=8050 \
    -e DECODE_HOST=decode \
    -e DECODE_PORT=8060 \
    nano-vllm-pd

# Decode container (Node 2)
docker run -d --name decode \
    --network pd-network \
    --privileged \
    -v /tmp/qwen3-0.6b:/model \
    -e ROLE=decode \
    -e PREFILL_HOST=prefill \
    -e PREFILL_PORT=8050 \
    -e DECODE_HOST=decode \
    -e DECODE_PORT=8060 \
    nano-vllm-pd
```

Note: For multi-node RDMA, you'll need:
- InfiniBand/RDMA-capable network between nodes
- RDMA device passthrough (`--device /dev/infiniband/rdma_cm`)
- Or use host networking with physical interfaces

## Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `ROLE` | Server role: `prefill` or `decode` | Required |
| `MODEL_PATH` | Path to HuggingFace model | `/tmp/qwen3-0.6b` |
| `PREFILL_HOST` | Prefill server hostname | `127.0.0.1` |
| `PREFILL_PORT` | Prefill server TCP port | `8050` |
| `DECODE_HOST` | Decode server hostname | `127.0.0.1` |
| `DECODE_PORT` | Decode server TCP port | `8060` |
| `MOONCAKE_PROTOCOL` | Transfer protocol: `rdma` or `tcp` | `rdma` |
| `RDMA_DEVICE` | RDMA device name | `eth0-rxe` |

### PDConfig Parameters
```python
from pd_config import PDConfig

config = PDConfig(
    role='prefill',              # 'prefill' or 'decode'
    model_path='/tmp/qwen3-0.6b', # Model directory
    prefill_host='127.0.0.1',    # Prefill server address
    prefill_port=8050,           # Prefill server port
    decode_host='127.0.0.1',     # Decode server address
    decode_port=8060,            # Decode server port
    max_tokens=256,              # Max generation length
    temperature=0.7,             # Sampling temperature
)
```

## Troubleshooting

### "No RDMA devices found"
```bash
# Check if module is loaded
lsmod | grep rdma_rxe

# Check if device exists
rdma link show

# Recreate device
sudo rdma link delete eth0-rxe
sudo rdma link add eth0-rxe type rxe netdev eth0
```

### "Transfer Engine does not support overlapped memory region"
This happens when trying to register memory that overlaps with existing registration.

**Fix**: Use `allocate_managed_buffer()` instead of numpy arrays. Managed buffers are pre-registered.

### "Failed to get segment descriptor"
The remote node doesn't know about the buffer address.

**Fix**: 
1. Ensure buffer is allocated before transfer
2. Add `time.sleep(1)` after allocation for propagation
3. Verify both nodes show the buffer in `transfer_metadata_dump`

### "libtransfer_engine.so: cannot open shared object file"
```bash
# Set library path
export LD_LIBRARY_PATH="/path/to/Mooncake/build/mooncake-transfer-engine/src:/path/to/Mooncake/build/mooncake-common:$LD_LIBRARY_PATH"

# Or use ldconfig
sudo echo "/path/to/Mooncake/build/mooncake-transfer-engine/src" > /etc/ld.so.conf.d/mooncake.conf
sudo ldconfig
```

### Low RDMA Bandwidth
Soft-RoCE is limited by kernel TCP/IP stack. For production:
- Use real InfiniBand hardware
- Enable jumbo frames (MTU 9000)
- Tune kernel network parameters

## Performance Tuning

### Increase Block Size
Larger blocks = fewer transfers but more memory waste:
```python
# In cpu_model_runner.py
self.block_size = 32  # Default 16
```

### Batch Multiple Sequences
Process multiple prompts together on prefill node:
```python
# In pd_server_v2.py, modify _do_prefill to accept batch
```

### Use FP16/INT8 KV Cache
Reduce transfer size by 2-4x:
```python
# Change dtype in cpu_model_runner.py
kv_cache = np.zeros(..., dtype=np.float16)  # Instead of float32
```

## Next Steps

1. Replace TinyLM with real model (DeepSeek, Llama, etc.)
2. Add GPU support via CUDA
3. Implement continuous batching
4. Add metrics and monitoring
5. Deploy on Kubernetes with RDMA
