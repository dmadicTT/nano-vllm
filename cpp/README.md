# nano-vllm C++ engine

This directory contains a **C++ rebuild of the scheduling and block-management core** of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — a lightweight vLLM-style implementation. The design (prefill vs decode batching, KV cache blocks, chunked hashing) follows the Python engine; the C++ code is inference-backend agnostic and uses token IDs only.

## Build

From the repository root, using a build directory inside `cpp`:

```bash
cd cpp
mkdir -p build
cd build
cmake ..
cmake --build .
```

Requirements: **CMake 3.16+** and a **C++17** toolchain.

- **Debug logs** (scheduler, block_manager, model_runner, llm_engine):  
  `cmake -DNANOVLLM_DEBUG=ON ..` then rebuild.
- **compile_commands.json** is generated for clangd; a symlink is created in `cpp/` when you configure from `cpp/`.

## Run the demo

From `cpp/build`:

```bash
./engine_demo
```

The demo creates an `LLMEngine` with a **stub model runner** (no real model): it runs multiple requests with different `max_tokens`, so you can see scheduling, block allocation, and per-request completion. Output is to stdout (and to stderr if built with `NANOVLLM_DEBUG=ON`).

## Layout

| Path | Description |
|------|-------------|
| `include/nanovllm/` | Public headers: `config.hpp`, `sampling_params.hpp`, `engine/*.hpp` |
| `src/engine/` | Engine implementation (sequence, block_manager, scheduler, model_runner, llm_engine) |
| `src/demo.cpp` | Example: engine + stub runner, multiple requests with different `max_tokens` |

The engine expects token IDs; for prompt tokenization you can use the Python utility (see the main [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) repo and the `nanovllm.utils.tokenizer` / `cpp/scripts/tokenize_prompts.py` if present).

## Model runner

`IModelRunner::run(seqs, is_prefill)` returns one token per sequence. The default **stub** always returns a non-EOS token so sequences finish only when they hit `max_tokens`. To use a real model, implement `IModelRunner` (e.g. with your CUDA/ONNX/ggml backend) and plug it in via your own factory or engine setup.
