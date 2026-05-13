# PD Disaggregation Architecture

## Overview

This project implements **Prefill-Decode (PD) Disaggregation** for nano-vllm using **Mooncake** for RDMA-based KV cache migration.

## What is PD Disaggregation?

In standard LLM serving, both prefill (processing the prompt) and decode (generating tokens) run on the same GPU. PD disaggregation splits these phases:

- **Prefill nodes**: Process prompts, compute KV caches, serve multiple requests
- **Decode nodes**: Generate tokens using transferred KV caches, focus on low-latency

This separation allows independent scaling and better resource utilization.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT                                          │
│                         (HTTP Request)                                       │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROXY SERVER                                        │
│                    (Routes prefill → decode)                                 │
│  1. Receives request with prompt                                             │
│  2. Sends to prefill node                                                    │
│  3. Gets block_table + num_cached_tokens                                   │
│  4. Sends to decode node with prefill RPC port                               │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
              ▼                                         ▼
┌─────────────────────────────┐           ┌─────────────────────────────┐
│      PREFILL SERVER         │           │       DECODE SERVER         │
│      (Port 8050)            │           │       (Port 8060)           │
│                             │           │                             │
│  ┌─────────────────────┐    │           │    ┌─────────────────────┐  │
│  │   CPULLMEngine      │    │           │    │   CPULLMEngine      │  │
│  │   (TinyLM Model)    │    │           │    │   (TinyLM Model)    │  │
│  └─────────────────────┘    │           │    └─────────────────────┘  │
│           │                 │           │           │                 │
│           ▼                 │           │           ▼                 │
│  ┌─────────────────────┐    │           │    ┌─────────────────────┐  │
│  │   KV Cache (numpy)  │    │  RDMA KV  │    │   KV Cache (numpy)  │  │
│  │   [2, L, B, S, H, D]│◄───┼──Transfer─┼──►│   [2, L, B, S, H, D]│  │
│  └─────────────────────┘    │           │    └─────────────────────┘  │
│           │                 │           │           │                 │
│           ▼                 │           │           ▼                 │
│  ┌─────────────────────┐    │           │    ┌─────────────────────┐  │
│  │  Mooncake Transfer  │◄───┼──RDMA─────┼──►│  Mooncake Transfer  │  │
│  │  Engine (RDMA)      │    │           │    │  Engine (RDMA)      │  │
│  │  Port: auto         │    │           │    │  Port: auto         │  │
│  └─────────────────────┘    │           │    └─────────────────────┘  │
│                             │           │                             │
│  Role: Process prompts      │           │  Role: Generate tokens      │
│  - Tokenize input           │           │  - Receive KV via RDMA      │
│  - Run prefill              │           │  - Run decode steps         │
│  - Return block_table       │           │  - Return completion        │
└─────────────────────────────┘           └─────────────────────────────┘
```

## Data Flow

### 1. Prefill Phase
```
Client → Proxy: POST /v1/completions {prompt: "Hello..."}
Proxy → Prefill: TCP {action: "prefill", prompt_token_ids: [...]}
Prefill:
  - Runs CPULLMEngine.prefill()
  - Allocates blocks in KV cache
  - Fills KV cache with computed values
  - Returns {request_id, block_table, num_cached_tokens}
```

### 2. KV Cache Transfer (RDMA)
```
Decode → Prefill: TCP {action: "get_kv_cache", block_ids, decode_rpc_port, decode_buf_addrs}
Prefill:
  - Allocates managed buffer
  - Copies KV cache blocks to buffer
  - Calls Mooncake batch_transfer_sync_write() to decode's buffer
  - Frees managed buffer
Decode:
  - Allocated managed buffer (address sent to prefill)
  - Waits for RDMA write to complete
  - Copies received data to numpy KV cache
  - Frees managed buffer
```

### 3. Decode Phase
```
Decode:
  - Creates Sequence with received KV cache
  - Runs CPULLMEngine.decode() in loop
  - Generates tokens until completion
  - Returns {completion, tokens_generated}
Proxy → Client: Response with generated text
```

## Key Components

### CPULLMEngine
CPU-based LLM engine that replaces nano-vllm's GPU-dependent engine:
- Uses `TinyLM` (tiny transformer) for CPU inference
- Manages KV cache as numpy arrays
- Supports prefill and decode phases
- Compatible with nano-vllm's Sequence/Scheduler API

### MooncakeKVTransfer
Adapter for Mooncake Transfer Engine:
- Initializes RDMA transport on `eth0-rxe` (Soft-RoCE)
- Uses managed buffers for zero-copy transfers
- Handles segment descriptor propagation
- No explicit memory registration needed (managed buffers are pre-registered)

### PDServerV2
TCP server that handles prefill or decode role:
- **Prefill role**: Processes prompts, serves KV cache via RDMA
- **Decode role**: Receives KV cache, generates completions
- Communicates via pickle-over-TCP for control messages
- Uses Mooncake RDMA for data plane

## RDMA Transfer Details

### Why Managed Buffers?
Mooncake's `allocate_managed_buffer()` returns memory from a pre-registered pool. This avoids:
- Memory overlap issues with numpy arrays
- Explicit `batch_register_memory()` calls
- Segment descriptor synchronization problems

### Transfer Protocol
1. Decode allocates recv_buf, sends its address to prefill
2. Prefill allocates send_buf, copies KV data
3. Prefill calls `batch_transfer_sync_write(remote, [send_buf], [recv_buf], [size])`
4. Decode waits, then reads from recv_buf
5. Both free their buffers

### Soft-RoCE Setup
For environments without InfiniBand hardware:
```bash
# Load kernel module
modprobe rdma_rxe

# Create Soft-RoCE device on eth0
rdma link add eth0-rxe type rxe netdev eth0

# Verify
ibstat        # Shows eth0-rxe device
ib_write_bw   # Test bandwidth (~500 MB/s on typical cloud VMs)
```

## Design Decisions

### 1. CPU-Only Proof of Concept
- No GPU/CUDA required
- Uses TinyLM (random weights) for architecture validation
- Real models can be swapped in by replacing `TinyLM` with actual model

### 2. Standalone Modules
- All nano-vllm imports removed from runtime path
- Inlined `Config`, `SamplingParams`, `Sequence` to avoid `flash_attn` dependency
- Clean separation from upstream nano-vllm

### 3. Pickle over TCP for Control Plane
- Simple, synchronous request/response
- Exchanges block IDs, RPC ports, buffer addresses
- RDMA used only for data plane (KV cache)

### 4. Block-Level Transfer
- KV cache divided into blocks (default: 16 tokens/block)
- Only used blocks transferred (not entire cache)
- Block table tracks which blocks belong to which sequence

## Performance Considerations

### Current (POC)
- Model: TinyLM (2 layers, 64 hidden dim)
- Throughput: ~10-50 tokens/sec (CPU-bound)
- RDMA bandwidth: ~500 MB/s (Soft-RoCE limited)

### Production Targets
- Model: DeepSeek-R1 or similar
- Prefill: Batch multiple prompts for GPU efficiency
- Decode: Continuous batching for throughput
- RDMA: 50-100 GB/s with real InfiniBand
- KV cache compression: 2-4x via quantization

## Files

| File | Purpose |
|------|---------|
| `cpu_model_runner.py` | CPU-based LLM engine with TinyLM |
| `mooncake_kv_transfer.py` | Mooncake RDMA adapter |
| `pd_config.py` | Configuration dataclass |
| `pd_engine.py` | Extended Sequence/Scheduler classes |
| `pd_server_v2.py` | Main prefill/decode server |
| `proxy_server.py` | HTTP proxy routing requests |
| `run_pd_e2e_final.py` | End-to-end test orchestrator |

## Next Steps

1. **GPU Support**: Replace TinyLM with real GPU model
2. **Multi-Node**: Run prefill/decode on separate machines
3. **Docker**: Containerize with RDMA device passthrough
4. **Benchmarks**: Measure TTFT, TBT, throughput vs colocated
5. **Production Hardening**: Error handling, retries, metrics
