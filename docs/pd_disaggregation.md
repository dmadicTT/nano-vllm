# PD Disaggregation in nano-vllm

This document explains how nano-vllm is adapted to run with **disaggregated
prefill and decode** workers that exchange KV cache via the full
**Mooncake** stack (master service + transfer engine + distributed store).

It also captures the things that surprised me on the way there. If you only
want to *run* the demo, jump to [Quick run](#quick-run).

---

## 1. What changes — and where

The integration touches three layers of the engine plus a new
transport adapter.

| File | Change |
|---|---|
| `nanovllm/config.py` | adds `role` (`'prefill'`/`'decode'`/`'colocated'`) and a small set of `mooncake_*` fields. The original `device` / `gpu_memory_utilization` / `enforce_eager` fields are gone — this fork is CPU-only. |
| `nanovllm/layers/attention.py` | rewritten to use `F.scaled_dot_product_attention` + an `index_copy_`-based paged store. flash-attn / triton / triton.jit are removed entirely. |
| `nanovllm/engine/model_runner.py` | the CUDA branch (NCCL, CUDA graphs, `torch.cuda.*`, pin_memory, warmup) is deleted. The runner uses a single-rank `gloo` process group so the `Linear` / `Embedding` layers' `dist.get_world_size()` calls still resolve. |
| `nanovllm/engine/llm_engine.py` | wires `KVTransfer` in when `role != 'colocated'`, adds `run_prefill_and_publish` / `run_decode_from_handoff` / `_prefetch_from_store`. |
| `nanovllm/engine/kv_transfer.py` (new) | wraps a Mooncake Store client tied to a specific paged KV cache tensor. |
| `nanovllm/engine/pd_server.py` (new) | HTTP+JSON worker (`POST /prefill`, `/decode`, `GET /stats`, `POST /shutdown`). |
| `examples/pd_*.py` (new) | demo orchestrators — see the [README](../README.md). |

---

## 2. Architecture

```
   ┌──────────────────┐
   │ orchestrator /   │   length-prefixed pickle TCP
   │ pd_demo client   │ ─────┐                  ┌───────────────────┐
   └──────────────────┘      │                  │ Mooncake master   │
                             │                  │  (rpc :50051,     │
                             ▼                  │   metrics :9013)  │
                  ┌─────────────────────┐       └─────────┬─────────┘
                  │ prefill worker      │                 │
                  │  LLMEngine          │                 │  Mooncake control plane
                  │  role='prefill'     │ ◄───────────────┤  (TCP/IP)
                  │  ─ Qwen3-0.6B (CPU) │                 │
                  │  ─ KV cache tensor  │                 │
                  │  ─ KVTransfer       │                 │
                  │     ─ store client  │                 │
                  │     ─ xfer engine   │ ◄───────────────┘
                  └─────────┬───────────┘
                            │  (1) prefill returns {request_id, block_hashes,
                            │       first_token, num_cached_tokens, sp...}
                            │
                            │  (2) prefill pushes 1 Mooncake key per paged
                            │      block: nanovllm/req/<id>/blk/<i>
                            ▼
                  ┌─────────────────────┐
                  │ Mooncake Store /     │ ◄── HTTP metadata server :8081
                  │  TransferEngine     │     (service discovery only —
                  │  ─ global segment   │      no data flows through it)
                  │  ─ keyed by string  │
                  └─────────┬───────────┘
                            │
                            │  (3) decode pulls each key via Mooncake
                            │      get(); bytes traverse the TransferEngine
                            │      (RDMA if available, else TCP)
                            ▼
                  ┌─────────────────────┐
                  │ decode worker       │
                  │  LLMEngine          │
                  │  role='decode'      │
                  │  ─ Qwen3-0.6B (CPU) │
                  │  ─ KV cache tensor  │
                  │  ─ KVTransfer       │
                  └─────────────────────┘
                            │
                            ▼
                       generated reply
```

### Wire protocol between orchestrator and worker

Plain HTTP+JSON. Each worker exposes:

* `POST /prefill` — body `{prompt_token_ids, request_id, temperature, max_tokens, ignore_eos}` →
  `{ok, descriptor}`. The descriptor is what the decode side needs to
  reconstruct the sequence and pull the right Mooncake keys.
* `POST /decode` — body `{descriptor}` → `{ok, completion_token_ids, completion_text, request_id}`.
* `GET  /stats` — worker's authoritative push/pull byte counters
  (handy for "did anything actually transfer?" checks).
* `POST /shutdown` — clean exit.

Errors come back as `{"ok": false, "error": "..."}` with HTTP 4xx/5xx.
The control plane is one `requests`-based call per turn; the data plane
(KV bytes) goes through Mooncake's TransferEngine and never touches HTTP.

### KV layout and key scheme

The KV cache stays in nanovllm's native layout:

```
[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

A single paged "block" (across all layers, both K and V) is **not contiguous**
in this layout — the third dim is the block index. Rather than rearrange the
layout (which would slow down the colocated GPU path), we stage one block at
a time into a contiguous scratch tensor and ship its bytes. With Qwen3-0.6B
this is ~28 MiB / block; the gather copy is cheap relative to the RPC.

One Mooncake key per paged block, **content-addressed**:

```
nanovllm/kv/<xxh64-of-tokens-up-to-this-block>
```

The hash is the same chain hash nano-vllm's local `BlockManager.compute_hash`
uses (`xxh64(prev_block_hash || block_tokens)`), so two requests that share a
prefix produce the same hashes and therefore reuse the same Mooncake keys.
The producer-side `push` does an `is_exist` probe first and skips the put on
a cache hit; that turns Mooncake into a **cross-request prefix cache**. Keys
are not tied to a request id; we never call `remove` — Mooncake's lease /
eviction reclaims them. Local block ids (which differ between prefill and
decode because each side runs its own block manager — see gotcha 4 below) are
only the address the bytes land into on each side, never part of the key.

### Sequence state handoff

After `run_prefill_and_publish`:

* `num_cached_tokens = num_prompt_tokens`  — the prompt's KVs are in cache
* `num_tokens = num_prompt_tokens + 1`     — plus the first generated token
* `last_token = first_token`               — but its KV is **not yet** cached
* `status = RUNNING`

The descriptor carries the list of `block_hashes` (one xxh64 per paged block
of the prompt). The decode worker reconstructs the seq state, allocates fresh
local blocks (bypassing the prefix-hash logic, see gotcha 4), and
`pull(block_hashes, decode_local_block_ids)`s the prompt's KV bytes into
them, then runs its scheduler from the decode branch.

### Prefill-side prefetch (Mooncake as a remote prefix-cache tier)

Before `run_prefill_and_publish` even adds the seq to the scheduler, it
walks the prompt block-by-block and:

* on a local cache hit (`hash_to_block_id`), leaves the block alone — the
  existing nano-vllm prefix cache already covers it;
* on a Mooncake hit (`store.is_exist(key) == 1`), peeks at a free block,
  pulls the bytes into it, registers the block's hash + tokens, and leaves
  it in the free list. The scheduler's subsequent `allocate()` finds it via
  the cached-block path and ref-counts it normally;
* on a miss, stops — only a contiguous prefix is honored.

`scheduler.schedule()` then sees a seq whose `num_cached_tokens` already
covers the prefetched prefix, and the model forward only runs on the
suffix. This works the same way whether the prefix was filled by a previous
turn on this node, an earlier turn on a sibling prefill node, or a decode
node's pull from a third node — because the key namespace is content-only.

Counters exposed on `/stats`:
* `blocks_pushed`         — `put`s that actually hit the wire
* `blocks_skipped_push`   — `put`s that short-circuited via `is_exist`
* `blocks_pulled`         — every `get` (prefetch + decode)
* `blocks_prefetched`     — subset of `blocks_pulled` issued by prefetch

### Multiple prefill workers

The architecture supports a fleet of prefill nodes with no engine-level
changes — each one runs its own `LLMEngine(role='prefill')` and talks to the
same Mooncake master. Content-addressed keys mean two nodes computing
the same prompt produce the same key, and the prefetch path means the
second-mover doesn't recompute.

`pd_demo.py` takes `--num-prefill N` to spawn N workers on ports
`[prefill_port, prefill_port+N)` and round-robins requests across them.
`pd_prefetch_demo.py` is a tighter focused demo: it sends the same prompt
to two nodes in series and prints the stats so you can see node 1 hit
Mooncake instead of recomputing.

The first generated token's KV gets written on the decode side during its
first decode step. So exactly the prompt is what crosses the wire.

---

## 3. Quick run

### Prerequisites

```bash
# nanovllm + Mooncake + libcudart (everything needed for the disaggregated
# CPU path is declared in pyproject.toml).
pip install .

# A model file (Qwen3-0.6B in the example)
huggingface-cli download Qwen/Qwen3-0.6B \
    --local-dir models/Qwen3-0.6B --local-dir-use-symlinks False
```

### Run the demo

```bash
python examples/pd_demo.py --max-tokens 40
```

What you should see:

* The metadata server and master come up.
* Two worker processes load Qwen3-0.6B on CPU (~30s each — first call into
  attention through `torch.compile` is the slow bit).
* 5 conversation turns across 2 interleaved conversations get answered.
* A summary line at the end:

```
Mooncake KV transfer summary (in-process counters):
  [prefill]  pushed: 5 blocks / 140.0 MiB   pulled: 0 blocks / 0.0 MiB
  [decode]   pushed: 0 blocks / 0.0 MiB     pulled: 5 blocks / 140.0 MiB
  [master log] last metric line: PutStart=5/5, Get=5/5, keys=0
```

`PutStart=5/5, Get=5/5` on the master log is the cross-check: the master
recorded the same number of put/get operations the workers think they did.

### Trying RDMA

```bash
sudo modprobe rdma_rxe
sudo rdma link add rxe0 type rxe netdev eth0
rdma link show               # should show 'state ACTIVE'

python examples/pd_demo.py --protocol rdma --rdma-devices rxe0
```

(This worked on an earlier environment of mine but failed in the
container I built this on. See gotcha 1.)

---

## 4. Learnings / gotchas

### 1. Soft-RoCE (rdma_rxe) needs more than just the kernel module

To get a userspace RDMA path on a host without InfiniBand hardware, you do
need the kernel module:

```bash
sudo modprobe rdma_rxe
sudo rdma link add rxe0 type rxe netdev eth0
rdma link show   # eth0/rxe0 state ACTIVE
```

But that's only half the story. You also need:

* `librxe-rdmav34.so` (Ubuntu: `ibverbs-providers` — usually already present)
* the `/etc/libibverbs.d/rxe.driver` provider mapping (same package)
* `/dev/infiniband/uverbs0` (auto-created by udev — *but the container I
  developed in had no udev, so I had to `mknod` it myself*)
* enough effective capabilities to actually open that device

The container I had didn't grant any capabilities (`CapEff=0`), so opening
`/dev/infiniband/uverbs0` returned EPERM regardless of file permissions.
Mooncake then prints `"No available RNIC"` and refuses to initialize. The
demo falls back to `--protocol tcp` in that case; the rest of the stack
(master, store, transfer engine) is unchanged.

### 2. `mooncake-transfer-engine` is a CUDA wheel

`pip install mooncake-transfer-engine` resolves to a wheel built against
CUDA 12 — even though we use it on a CPU-only host. The shared object
imports `libcudart.so.12` on load. If that isn't on `LD_LIBRARY_PATH` or
preloaded, `from mooncake.store import MooncakeDistributedStore` blows up
with `ImportError: libcudart.so.12: cannot open shared object file`.

`KVTransfer` calls `ctypes.CDLL(libcudart, RTLD_GLOBAL)` before importing
the Mooncake Python module. CUDA itself never runs — we only need its
symbols resolved.

### 3. `put` copies into the segment, `put_from` doesn't

`MooncakeDistributedStore.put(key, value)` copies the bytes into Mooncake's
mounted segment, so the source buffer can be reused immediately. I verified
this by issuing two `put`s back-to-back with the same `bytearray` (mutated
between calls) and reading both keys back — each returned its own snapshot.

`put_from(key, ptr, size)` is the zero-copy variant. The source buffer must
stay alive at least until every consumer has finished `get_into` — re-using
or freeing it before that races. We sidestep that by using the copying `put`
API; the staging tensor can be reused for the next block immediately.

### 4. The block-manager hash logic gets in the way of migration

nanovllm's `BlockManager.can_allocate(seq)` hashes the seq's content to
opportunistically reuse cached blocks. On the **decode** node we don't want
that — we want fresh blocks to land the migrated KV bytes into.

`run_decode_from_handoff` calls `block_manager._allocate_block()` directly
in a loop instead of going through the hash path. The seq's `block_table`
is filled with fresh decode-local ids, and the seq is shoved straight into
the running queue with `status = RUNNING` and `is_prefill = False`.

### 5. KV block layout vs. zero-copy

Native layout: `[2, L, B, S, H, D]`. A single block "across all layers"
(i.e. `kv_cache[:, :, b, :, :, :]`) is a strided view, *not* a contiguous
slab. To do true zero-copy transfers with Mooncake we would have either
(a) re-laid-out the cache to `[B, 2, L, S, H, D]` and accepted strided K
and V views inside the attention layer, or (b) chopped each block into
`2*L` per-(k|v, layer) keys and used `batch_put_from_multi_buffers`.

For now we pay one staged copy per block (~28 MiB for Qwen3-0.6B). That's
cheap relative to even loopback TCP at this scale, and it keeps the
attention layer untouched. The first option is the right one if/when
we move this to real per-node RDMA at scale.

### 6. `bf16` doesn't go through numpy

`torch.bfloat16` (the native dtype for Qwen3) has no numpy equivalent, so
`staging.numpy().tobytes()` throws `Got unsupported ScalarType BFloat16`.
We grab raw bytes via `ctypes.string_at(t.data_ptr(), t.nbytes)` and on
the receiver side use `torch.frombuffer(bytearray(data), dtype=...)`
which *does* accept bf16. (frombuffer of `bytes` is read-only; we wrap
in `bytearray` so the resulting tensor is writeable for the subsequent
`copy_` into the KV cache.)

### 7. nanovllm's GPU path was intertwined enough that the cleanest fix was to delete it

Originally upstream's `ModelRunner` did `torch.cuda.set_device`,
`torch.set_default_device("cuda")`, `dist.init_process_group("nccl", ...)`,
captured CUDA graphs after a warmup pass, and called into flash-attn /
triton kernels for attention. We initially kept a CPU branch alongside,
but the GPU branch wasn't testable on this host and complicated every
file it touched. In this fork the CUDA branch is **gone**:

* `model_runner.py` has a single path — `gloo` process group, plain
  `torch.empty` for the paged KV cache, no graphs, no warmup, no
  `torch.cuda.*` anywhere.
* `attention.py` has a single path — `F.scaled_dot_product_attention`
  plus an `index_copy_`-based paged store. `triton` and `flash_attn`
  are no longer imported even lazily; they're not deps.
* `Config` has lost `device`, `gpu_memory_utilization`, and
  `enforce_eager` — they were only ever inputs to the deleted GPU path.

The CPU `Linear` / `Embedding` layers still need `dist.get_rank()` /
`dist.get_world_size()` to resolve, so the runner still opens a single-
rank `gloo` group on `127.0.0.1:<role-hash>`; `tensor_parallel_size` is
asserted to `== 1`.

### 8. Periodic master metrics are emitted on a slow timer

The mooncake master logs an admin metrics line on its own schedule
(default ~5 sec quiet, longer when idle). A fast demo can finish before
the master emits a single metric line, leaving you wondering whether
anything happened. `KVTransfer` keeps its own `bytes_pushed` /
`bytes_pulled` counters that the demo dumps at shutdown — those are the
ground truth.

### 9. The master admin server collides with itself across runs

`mooncake_master` exposes its admin/metrics server on port `9003` by
default. If a previous instance didn't shut down cleanly (or another
master is around), startup logs the friendly error `bind port: 9003 error:
Address already in use` — but the master itself still runs on the RPC
port. We pass `--metrics_port=9013` from the demo to dodge the typical
clash.

---

## 5. What's intentionally not covered

* **Real RDMA over the wire.** The integration is RDMA-capable
  (`--protocol rdma --rdma-devices <dev>`) but the development environment
  only let me exercise `tcp`. The control plane and data plane go through
  the same Mooncake objects either way.
* **Continuous batching of prefill + decode handoffs.** The current PD
  worker handles one outstanding request at a time. Real PD systems pipeline
  multiple requests across both workers.
* **Cross-request prefix caching across nodes.** Each turn re-pushes the
  entire prompt's KV; a smarter design would deduplicate against keys the
  store already has.
* **Multi-node deployment.** The demo runs both workers on `127.0.0.1`.
  Switching to a second host is purely a matter of `mooncake_local_hostname`
  + master address; no engine changes are needed.

---

## 6. Useful pointers

* Mooncake project — https://github.com/kvcache-ai/Mooncake
* The original nano-vllm — https://github.com/GeeeekExplorer/nano-vllm
