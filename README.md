<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** — Comparable inference speeds to vLLM
* 📖 **Readable codebase** — Clean implementation in ~ 1,200 lines of Python code
* 🖥  **CPU-only build** — Runs Qwen3-0.6B on plain CPU through `torch.nn.functional.scaled_dot_product_attention`; no `flash-attn`, no `triton`, no CUDA. (The upstream CUDA path has been removed in this fork.)
* 🛰  **PD disaggregation via Mooncake** — Run prefill and decode on separate workers that share KV cache through the full Mooncake stack (master + distributed store + transfer engine, RDMA-or-TCP)

## Installation

```bash
git clone https://github.com/dmadicTT/nano-vllm.git
cd nano-vllm
python3.10 -m venv .venv && source .venv/bin/activate
pip install .
```

`pip install .` pulls the runtime deps the disaggregated CPU path needs:
`torch`, `transformers`, `tqdm`, `xxhash`, `requests`,
`mooncake-transfer-engine`, `nvidia-cuda-runtime-cu12`. The CUDA runtime
is only there so the Mooncake wheel's dynamic loader can resolve
`libcudart.so.12` — nothing in nano-vllm itself uses CUDA in this fork.

## Model

```bash
huggingface-cli download Qwen/Qwen3-0.6B \
    --local-dir models/Qwen3-0.6B --local-dir-use-symlinks False
```

The examples below default to `./models/Qwen3-0.6B` relative to the repo root.

## PD disaggregation with Mooncake

There are four demo scripts under `examples/`, each scripted around a
different point in the design.

### `pd_demo.py` — full end-to-end demo

Spawns `mooncake_master` + `mooncake_http_metadata_server` + one prefill
worker + one decode worker, then drives a multi-turn chat with two
interleaved conversations.

```bash
python examples/pd_demo.py
```

Useful flags:

* `--num-prefill N` — spawn N prefill workers (round-robin LB)
* `--verbose-mooncake` — enable Mooncake's C++ glog and stream master /
  metadata stderr to the terminal
* `--protocol rdma --rdma-devices rxe0` — try real RDMA (Soft-RoCE)
  instead of the TCP transport

### `pd_two_prefill_chat.py` — focused two-prefill / one-decode story

One user, two turns. Turn 1 routes to `prefill@A`, turn 2 (same
conversation, growing context) routes to `prefill@B`. The system prompt
is padded so the first KV block is fully covered by it, which means
`prefill@B`'s `_prefetch_from_store` finds the shared prefix in
Mooncake (put there by `prefill@A`) and skips recomputing it.

```bash
python examples/pd_two_prefill_chat.py
```

Output is a curated INFO-only narrative (Tokenization → Hashing → local
hit count → store hit count → Prefilling → Pushing → decode-side
equivalents). On exit it prints `/tmp/pd_two_prefill_master.log`'s path
and the last 10 lines of the Mooncake master's audit log.

### `pd_prefetch_demo.py` — minimal cross-node prefetch validation

Two prefill workers, same 300-token prompt sent to each in turn. Prints
each worker's `KVTransfer` counters at the end:

```
node0 after its request:  pushed=2  skipped-put=0  pulled=0  prefetched=0
node1 after its request:  pushed=0  skipped-put=2  pulled=1  prefetched=1
```

`prefetched=1` on node 1 proves the shared full block was served from
Mooncake instead of recomputed.

### `pd_show_protocol.py` — dump the HTTP+JSON wire protocol

Runs two prefill/decode round-trips and prints every request and response
dict that crosses the worker socket. Useful when you want to see the
exact `{prompt_token_ids, request_id, ...}` body the orchestrator sends
and the `{descriptor: {block_hashes, first_token, ...}}` reply.

```bash
python examples/pd_show_protocol.py
```

### Design notes / gotchas

[docs/pd_disaggregation.md](docs/pd_disaggregation.md) covers the
architecture, key naming scheme, sequence-state handoff, prefetch flow,
multi-prefill orchestration, and an annotated list of nine things that
went wrong on the way to a working integration (Soft-RoCE container
restrictions, the libcudart loader trick, `put` vs `put_from` semantics,
etc.).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)