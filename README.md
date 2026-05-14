<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## PD disaggregation with Mooncake (this fork)

This fork adds support for running prefill and decode in **separate workers**
that exchange KV cache through the full Mooncake stack (master service,
distributed store, and transfer engine, with RDMA-or-TCP transport). The
integration is in-tree — see `nanovllm/engine/kv_transfer.py`,
`nanovllm/engine/pd_server.py`, and the CPU paths added to
`nanovllm/layers/attention.py` and `nanovllm/engine/model_runner.py`.

End-to-end multi-turn chat demo (CPU-only):

```bash
pip install mooncake-transfer-engine nvidia-cuda-runtime-cu12
huggingface-cli download Qwen/Qwen3-0.6B \
    --local-dir models/Qwen3-0.6B --local-dir-use-symlinks False
python examples/pd_demo.py
```

Design notes and gotchas: [docs/pd_disaggregation.md](docs/pd_disaggregation.md).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)