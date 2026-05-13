# Key Learnings from Building PD Disaggregation

## 1. Mooncake RDMA with CPU Memory

### Discovery: Managed Buffers Are Pre-Registered
**Problem**: Initial attempts to register numpy arrays with `batch_register_memory()` failed with "overlapped memory region" errors.

**Root Cause**: Mooncake's allocator pool is already registered. When we allocated managed buffers and tried to register them again, they overlapped with existing registrations.

**Solution**: Use `allocate_managed_buffer()` which returns pre-registered memory. No explicit `batch_register_memory()` needed.

```python
# ❌ Wrong - causes overlap
buf = engine.allocate_managed_buffer(size)
engine.batch_register_memory([buf], [size])  # Fails with -7

# ✅ Correct - already registered
buf = engine.allocate_managed_buffer(size)
# Use directly for transfer
```

### Discovery: Cross-Process Address Exchange Required
**Problem**: RDMA transfer failed because prefill didn't know decode's buffer addresses.

**Root Cause**: `batch_transfer_sync_write()` needs destination addresses on the remote node. Different processes have different virtual address spaces.

**Solution**: Decode allocates buffer first, sends address to prefill via TCP, then prefill writes to that address.

```python
# Decode side
recv_buf = engine.allocate_managed_buffer(total_size)
recv_addrs = [recv_buf + i * block_size for i in range(num_blocks)]

# Send recv_addrs to prefill via TCP

# Prefill side  
send_buf = engine.allocate_managed_buffer(total_size)
engine.batch_transfer_sync_write(
    remote_session,
    src_ptrs=[send_buf + i * block_size for i in range(num_blocks)],
    dst_ptrs=recv_addrs,  # Decode's addresses!
    lengths=[block_size] * num_blocks
)
```

### Discovery: Buffer Lifecycle Matters
**Problem**: Transfer failed because prefill freed buffer before decode could read.

**Root Cause**: Segment descriptors are updated when buffers are freed. If prefill frees send_buf before decode reads, the address is no longer valid.

**Solution**: Keep buffers allocated until transfer completes. Use synchronous `batch_transfer_sync_write()` which blocks until done.

## 2. Soft-RoCE Setup

### Kernel Module Loading
```bash
# Install required packages
apt-get install -y linux-modules-extra-$(uname -r) rdma-core ibverbs-utils

# Load Soft-RoCE module
modprobe rdma_rxe

# Create device
rdma link add eth0-rxe type rxe netdev eth0

# Verify
rdma link show        # Should show eth0-rxe state ACTIVE
ibstat                # Should list eth0-rxe device
ib_write_bw           # Test bandwidth
```

### Bandwidth Expectations
- Soft-RoCE over TCP: ~500 MB/s (limited by kernel networking)
- Real InfiniBand: 50-100 GB/s
- For KV cache transfer, even Soft-RoCE is sufficient for POC

## 3. nano-vllm Architecture

### No Native PD Support
nano-vllm (unlike upstream vLLM) has no built-in PD disaggregation. Key gaps:
- No KV cache transfer mechanism
- No prefill/decode role separation
- No distributed scheduler

### Flash Attention Dependency
nano-vllm unconditionally imports `flash_attn` at package load, which requires CUDA.

**Workaround**: Inline standalone copies of required classes:
```python
# Instead of: from nanovllm.engine.sequence import Sequence, SequenceStatus
# We define our own:
class SequenceStatus(Enum):
    WAITING = 0
    RUNNING = 1
    # ...

class Sequence:
    # Minimal implementation
```

### Scheduler Design
nano-vllm's scheduler is simple but effective:
- `waiting` queue: Sequences ready for prefill
- `running` queue: Sequences in decode phase
- `block_manager`: Allocates KV cache blocks

For PD disaggregation, we extended this to:
- Track which blocks need transfer
- Support external KV cache injection
- Handle cross-node request routing

## 4. CPU Model Runner Design

### TinyLM for Architecture Validation
Instead of loading a real model (which requires GBs of memory and GPU), we use a tiny transformer:
```python
class TinyLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)
```

This validates:
- Prefill/decode phase separation
- KV cache allocation and transfer
- Token generation loop
- Without GPU dependency

### KV Cache Shape
Standard shape: `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`
- `2`: Key and Value
- `num_layers`: Transformer layers
- `num_blocks`: Allocated blocks
- `block_size`: Tokens per block (default 16)
- `num_kv_heads`: KV attention heads
- `head_dim`: Dimension per head

For CPU POC, we simplified to `num_kv_heads=1` and small dimensions.

## 5. Testing Strategy

### Iterative Development
We created 5+ versions of the e2e test (`run_pd_e2e_v1.py` through `run_pd_e2e_final.py`):
1. v1: Basic TCP communication
2. v2: Added Mooncake integration
3. v3: Fixed numpy serialization
4. v4: Added managed buffers
5. v5: Fixed address exchange
6. final: Working RDMA transfer

**Lesson**: When debugging distributed systems, iterate quickly with simple tests before building full pipeline.

### Verification Points
At each stage, verify:
1. **Mooncake initialization**: Both nodes discover RDMA device
2. **Buffer registration**: Managed buffers appear in segment descriptors
3. **Direct transfer**: Simple test with known data patterns
4. **End-to-end**: Full prefill → transfer → decode pipeline

## 6. Common Pitfalls

### Pitfall: Import Order
Mooncake's shared libraries must be loaded before importing the Python module:
```python
# Must do this FIRST
ctypes.CDLL("libtransfer_engine.so", mode=ctypes.RTLD_GLOBAL)

# Then this
import engine as mooncake_engine
```

### Pitfall: LD_LIBRARY_PATH
The transfer engine library has dependencies that need to be found:
```python
os.environ["LD_LIBRARY_PATH"] = (
    f"{MOONCAKE_BUILD}/mooncake-transfer-engine/src:"
    f"{MOONCAKE_BUILD}/mooncake-common:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)
```

### Pitfall: Segment Descriptor Propagation
After allocating/registering memory, there's a delay before other nodes see it:
```python
# Allocate buffer
buf = engine.allocate_managed_buffer(size)

# Wait for propagation
time.sleep(1)  # Required!

# Now transfer will work
engine.batch_transfer_sync_write(...)
```

### Pitfall: Process Isolation
When testing with multiprocessing, each process needs its own Mooncake engine instance. Sharing engines across processes causes crashes.

## 7. Performance Insights

### KV Cache Size Calculation
For a typical model:
```
bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * 2  # fp16
bytes_per_block = bytes_per_token * block_size
```

Example for DeepSeek-R1:
```
num_layers = 61
num_kv_heads = 1
head_dim = 512
block_size = 16

bytes_per_token = 2 * 61 * 1 * 512 * 2 = 125,952 bytes (~123 KB)
bytes_per_block = 125,952 * 16 = 2,015,232 bytes (~1.9 MB)
```

For 4096 tokens (256 blocks): ~500 MB to transfer

### Transfer Time Estimates
| Setup | Bandwidth | 500MB Transfer Time |
|-------|-----------|---------------------|
| Soft-RoCE | 500 MB/s | ~1 second |
| InfiniBand HDR | 25 GB/s | ~20 ms |
| InfiniBand NDR | 100 GB/s | ~5 ms |

For low-latency serving, real InfiniBand is essential.

## 8. Production Readiness Checklist

- [ ] GPU support (replace TinyLM)
- [ ] Real model weights loading
- [ ] FP8/INT8 KV cache quantization
- [ ] Multi-node deployment (not just localhost)
- [ ] Error handling and retry logic
- [ ] Metrics and monitoring
- [ ] Load balancing across prefill nodes
- [ ] Continuous batching on decode nodes
- [ ] PagedAttention integration
- [ ] Docker/Kubernetes deployment
- [ ] Security (TLS, auth)

## 9. References

- **nano-vllm**: https://github.com/GeeeekExplorer/nano-vllm
- **Mooncake**: https://github.com/kvcache-ai/Mooncake
- **vLLM PD Disaggregation**: https://docs.vllm.ai/en/latest/features/disagg_prefill.html
- **RDMA over Converged Ethernet**: https://en.wikipedia.org/wiki/RDMA_over_Converged_Ethernet
