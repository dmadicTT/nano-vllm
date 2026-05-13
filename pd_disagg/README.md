# nano-vllm PD Disaggregation

Prefill-Decode (PD) Disaggregation for nano-vllm using Mooncake RDMA-based KV cache migration.

## What is This?

This project implements **PD Disaggregation** - splitting LLM inference into two phases:
- **Prefill nodes**: Process prompts and compute KV caches
- **Decode nodes**: Generate tokens using transferred KV caches

KV cache is migrated between nodes using **RDMA** (Remote Direct Memory Access) via [Mooncake](https://github.com/kvcache-ai/Mooncake).

## Architecture

```
┌─────────┐     ┌─────────────┐     ┌───────────┐     ┌───────────┐
│ Client  │────►│ Proxy Server│────►│  Prefill  │────►│  Decode   │
│         │     │  (Router)   │     │  Server   │ RDMA│  Server   │
└─────────┘     └─────────────┘     └───────────┘     └───────────┘
                                           │                   │
                                           │  KV Cache         │
                                           │  Transfer         │
                                           └───────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design.

## Quick Start

### 1. Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-pip git cmake build-essential \
    linux-modules-extra-$(uname -r) rdma-core ibverbs-utils

pip install torch transformers numpy
```

### 2. Set Up RDMA (Soft-RoCE)

```bash
sudo modprobe rdma_rxe
sudo rdma link add eth0-rxe type rxe netdev eth0
rdma link show  # Verify eth0-rxe is ACTIVE
```

### 3. Build Mooncake

```bash
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake && mkdir build && cd build
cmake .. -DUSE_CUDA=OFF && make -j$(nproc)
```

### 4. Run Test

```bash
cd pd_disagg/src
python3 run_pd_e2e_final.py
```

Expected output:
```
[1/5] Starting prefill server on port 8050...
[2/5] Starting decode server on port 8060...
[3/5] Sending prefill request...
[4/5] Sending decode request with KV cache transfer...
  [Prefill] RDMA send successful
  [Decode] KV cache received via RDMA for 1 blocks
  [Decode] Decode complete. Generated 20 tokens
[5/5] SUCCESS!
```

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design and data flow
- [LEARNINGS.md](docs/LEARNINGS.md) - Key technical learnings and pitfalls
- [RUN.md](docs/RUN.md) - Detailed run instructions and troubleshooting

## Project Structure

```
pd_disagg/
├── src/                    # Source code
│   ├── cpu_model_runner.py      # CPU-based LLM engine
│   ├── mooncake_kv_transfer.py  # Mooncake RDMA adapter
│   ├── pd_config.py             # Configuration
│   ├── pd_engine.py             # Extended scheduler
│   ├── pd_server_v2.py          # Main server (prefill/decode)
│   ├── proxy_server.py          # HTTP proxy
│   └── run_pd_e2e_final.py      # End-to-end test
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md
│   ├── LEARNINGS.md
│   └── RUN.md
├── docker/                 # Docker deployment
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
└── tests/                  # Test scripts
```

## Docker Deployment

```bash
cd pd_disagg/docker

# Build
docker build -t nano-vllm-pd ..

# Run with docker-compose
docker-compose up -d

# Or manually
docker run -d --name prefill --network host --privileged \
    -e ROLE=prefill -e MODEL_PATH=/model nano-vllm-pd

docker run -d --name decode --network host --privileged \
    -e ROLE=decode -e MODEL_PATH=/model nano-vllm-pd
```

## Key Features

- ✅ **RDMA-based KV transfer** via Mooncake
- ✅ **Soft-RoCE support** (no InfiniBand hardware needed)
- ✅ **CPU-only** proof of concept (no GPU required)
- ✅ **Standalone** modules (no nano-vllm imports)
- ✅ **End-to-end** automated test
- ✅ **Docker** deployment ready

## Limitations

- Uses TinyLM (random weights) for POC - replace with real model for production
- CPU-only - add CUDA support for GPU inference
- Single-machine - extend for multi-node deployment
- No batching - process one request at a time

## Performance

| Metric | Value |
|--------|-------|
| RDMA Bandwidth (Soft-RoCE) | ~500 MB/s |
| RDMA Bandwidth (InfiniBand) | 50-100 GB/s |
| KV Cache Transfer (1 block) | ~1 ms |
| Token Generation (CPU) | ~10-50 tok/s |

## References

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) - Minimal vLLM implementation
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - RDMA transfer engine
- [vLLM PD Disaggregation](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)

## License

Same as nano-vllm (Apache 2.0)
