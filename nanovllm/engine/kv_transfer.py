"""Mooncake-based KV cache transport for nano-vllm PD disaggregation.

This module wires a nanovllm `ModelRunner.kv_cache` into the Mooncake Store +
Master + TransferEngine stack so that prefill and decode workers can hand
KV blocks to each other.

Layout assumption: `kv_cache` is a contiguous CPU tensor with shape
    [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
where dim 0 is K/V and dim 2 is the paged block id. A single block (across
all layers, both K and V) is NOT contiguous in this layout — the slice
`kv_cache[:, :, block_id, :, :, :]` has the same shape across requests but
strides into the larger tensor. We pay one staging copy per block when
moving data through Mooncake; that copy is cheap relative to the network /
TransferEngine RPC cost we are exercising.
"""
from __future__ import annotations

import ctypes
import glob
import logging
import os
import time
from typing import List, Optional

import torch

log = logging.getLogger("nanovllm.kv_transfer")


# Preload libcudart (needed because the upstream Mooncake wheel was built
# against CUDA). On CPU-only hosts this is harmless — we never call cudaMalloc.
def _preload_cuda():
    # First try the canonical names — works if libcudart is already on
    # LD_LIBRARY_PATH or installed system-wide.
    for name in ("libcudart.so.12", "libcudart.so"):
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            pass
    # Otherwise look inside the nvidia-cuda-runtime-cu12 pip wheel for the
    # current Python environment. `nvidia.cuda_runtime` is a PEP 420 namespace
    # package (no __init__.py / no __file__), so we resolve through __path__.
    try:
        import nvidia.cuda_runtime  # type: ignore
        for base in list(nvidia.cuda_runtime.__path__):
            lib_dir = os.path.join(base, "lib")
            if not os.path.isdir(lib_dir):
                continue
            for entry in sorted(os.listdir(lib_dir)):
                if entry.startswith("libcudart.so"):
                    try:
                        ctypes.CDLL(os.path.join(lib_dir, entry), mode=ctypes.RTLD_GLOBAL)
                        return
                    except OSError:
                        pass
    except (ImportError, OSError, FileNotFoundError):
        pass
    # Fall through: if Mooncake's shared object can resolve cudart itself,
    # the import will still succeed; otherwise the next line will raise the
    # real ImportError so the user knows what's missing.


_preload_cuda()

from mooncake.store import MooncakeDistributedStore  # noqa: E402


class KVTransfer:
    """Wraps a Mooncake Store client tied to a specific KV cache tensor.

    Workflow:
      * Both prefill and decode workers create a `KVTransfer` against their
        local `kv_cache` tensor, connecting to a shared Mooncake master.
      * After prefill finishes, it calls `push(request_id, block_ids)` to
        publish the request's KV cache.
      * Decode calls `pull(request_id, block_ids, num_blocks)` to fetch the
        block bytes into its local `kv_cache` slots.
      * Decode calls `remove(request_id, num_blocks)` once it has the data
        and the keys can be reclaimed.

    Block IDs are *local* to each side — prefill and decode each run their
    own block manager and their index spaces differ. We content-address every
    block by an `xxh64` chain hash over the tokens it covers (matching
    nano-vllm's local BlockManager.compute_hash). Two requests that share a
    prefix produce the same hashes and therefore reuse the same Mooncake
    keys — Mooncake naturally becomes a cross-request prefix cache. Keys are
    not tied to a request id; we never call `remove`.
    """

    KEY_FMT = "nanovllm/kv/{h:016x}"

    def __init__(self, kv_cache: torch.Tensor, *, role: str,
                 local_hostname: str, metadata_server: str, master_addr: str,
                 protocol: str = "tcp", rdma_devices: str = "",
                 global_segment_size: int = 1 << 30,
                 local_buffer_size: int = 256 << 20):
        assert kv_cache.is_cpu, "KVTransfer only supports CPU kv_cache for now"
        assert kv_cache.ndim == 6, (
            f"kv_cache expected shape [2, L, B, S, H, D]; got {tuple(kv_cache.shape)}"
        )
        self.kv_cache = kv_cache
        self.role = role
        _, self.num_layers, self.num_blocks, self.block_size, self.num_kv_heads, self.head_dim = kv_cache.shape
        self.dtype = kv_cache.dtype
        self.block_shape = (2, self.num_layers, self.block_size, self.num_kv_heads, self.head_dim)
        self.block_numel = 2 * self.num_layers * self.block_size * self.num_kv_heads * self.head_dim
        self.bytes_per_block = self.block_numel * kv_cache.element_size()

        # Mooncake Store client
        self.store = MooncakeDistributedStore()
        rc = self.store.setup(
            local_hostname=local_hostname,
            metadata_server=metadata_server,
            global_segment_size=global_segment_size,
            local_buffer_size=local_buffer_size,
            protocol=protocol,
            rdma_devices=rdma_devices,
            master_server_addr=master_addr,
        )
        if rc != 0:
            raise RuntimeError(f"Mooncake Store setup failed: rc={rc}")

        # Persistent staging buffer for one block (re-used across calls).
        # Allocated as a contiguous CPU tensor with the same dtype as the
        # KV cache; gather/scatter copies bridge between the strided block
        # view and this buffer.
        self.staging = torch.empty(*self.block_shape, dtype=self.dtype, device="cpu")

        # Counters so callers can verify data actually flowed through Mooncake.
        self.bytes_pushed = 0
        self.bytes_pulled = 0
        self.blocks_pushed = 0
        self.blocks_pulled = 0
        # Blocks that we *would* have pushed but skipped because the key was
        # already in the store (cross-request prefix cache hit).
        self.blocks_skipped_push = 0
        # Subset of `blocks_pulled`: those pulled by prefill-side prefetch
        # rather than decode-side handoff. Bumped by the engine.
        self.blocks_prefetched = 0

        log.info(
            "KVTransfer[%s] connected: master=%s metadata=%s protocol=%s "
            "block=%.1fMiB (L=%d, B=%d, S=%d, H=%d, D=%d, dtype=%s)",
            role, master_addr, metadata_server, protocol,
            self.bytes_per_block / (1 << 20),
            self.num_layers, self.num_blocks, self.block_size,
            self.num_kv_heads, self.head_dim, self.dtype,
        )

    # -------- producer side (prefill) --------
    def push(self, block_hashes: List[int], local_block_ids: List[int]) -> None:
        """Publish each block's KV bytes under its content hash.

        For every (hash, local block id) pair: build a Mooncake key from the
        hash, skip the put if Mooncake already has the key (prefix cache hit),
        otherwise gather the strided block into staging and `put` the bytes.
        Source: the prefill node's KV cache.
        """
        assert len(block_hashes) == len(local_block_ids), (
            f"block_hashes/local_block_ids length mismatch: "
            f"{len(block_hashes)} vs {len(local_block_ids)}"
        )
        t0 = time.perf_counter()
        n_put = 0
        n_skip = 0
        for bhash, bid in zip(block_hashes, local_block_ids):
            key = self.KEY_FMT.format(h=bhash & ((1 << 64) - 1))
            # If the key is already in the store, some earlier request with
            # the same prefix put it there. Skip — Mooncake will serve the
            # decode-side get() from that existing entry.
            if self.store.is_exist(key) == 1:
                self.blocks_skipped_push += 1
                n_skip += 1
                log.info("push  %s  local_bid=%d  SKIP (already in store)", key, bid)
                continue
            # Gather the (possibly strided) block into staging — copy_() handles
            # the layout translation, then we ship the contiguous bytes.
            self.staging.copy_(self.kv_cache[:, :, bid, :, :, :])
            # `put` copies bytes into Mooncake's mounted segment, so we can
            # reuse the staging buffer immediately for the next block.
            # ctypes.string_at handles bfloat16 / fp16 which numpy doesn't.
            payload = ctypes.string_at(self.staging.data_ptr(), self.bytes_per_block)
            t_put = time.perf_counter()
            rc = self.store.put(key, payload)
            if rc != 0:
                raise RuntimeError(f"Mooncake put({key}) failed rc={rc}")
            self.bytes_pushed += self.bytes_per_block
            self.blocks_pushed += 1
            n_put += 1
            log.info(
                "push  %s  local_bid=%d  %.1f MiB  put=%.3fs",
                key, bid, self.bytes_per_block / (1 << 20),
                time.perf_counter() - t_put,
            )
        log.info(
            "push  %d block(s): %d new / %d cached  in %.3fs",
            len(block_hashes), n_put, n_skip, time.perf_counter() - t0,
        )

    # -------- consumer side (decode) --------
    def pull(self, block_hashes: List[int], local_block_ids: List[int],
             *, wait_timeout: float = 30.0, poll_interval: float = 0.02) -> None:
        """Populate local KV slots `local_block_ids` with bytes keyed by `block_hashes`.

        Blocks until all expected keys are available (or `wait_timeout` is hit).
        Order of `block_hashes` must match `local_block_ids` (hash[i] goes into
        the local kv_cache slice at block id local_block_ids[i]).
        """
        assert len(block_hashes) == len(local_block_ids), (
            f"block_hashes/local_block_ids length mismatch: "
            f"{len(block_hashes)} vs {len(local_block_ids)}"
        )
        t0 = time.perf_counter()
        deadline = time.monotonic() + wait_timeout
        for bhash, bid in zip(block_hashes, local_block_ids):
            key = self.KEY_FMT.format(h=bhash & ((1 << 64) - 1))
            # Spin until the producer has published this block.
            t_wait = time.perf_counter()
            while self.store.is_exist(key) != 1:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"KV block {key} not received within {wait_timeout}s")
                time.sleep(poll_interval)
            wait_s = time.perf_counter() - t_wait
            t_get = time.perf_counter()
            data = self.store.get(key)
            if not data:
                raise RuntimeError(f"Mooncake get({key}) returned empty payload")
            expected = self.bytes_per_block
            if len(data) != expected:
                raise RuntimeError(
                    f"Mooncake get({key}) returned {len(data)} bytes, expected {expected}"
                )
            get_s = time.perf_counter() - t_get
            # Materialize bytes into a typed tensor view of the returned buffer,
            # then scatter into the right block slot of the local kv_cache.
            tensor = torch.frombuffer(bytearray(data), dtype=self.dtype).view(*self.block_shape)
            self.kv_cache[:, :, bid, :, :, :].copy_(tensor)
            self.bytes_pulled += self.bytes_per_block
            self.blocks_pulled += 1
            log.info(
                "pull  %s  local_bid=%d  %.1f MiB  wait=%.3fs get=%.3fs",
                key, bid, self.bytes_per_block / (1 << 20), wait_s, get_s,
            )
        log.info(
            "pull  %d block(s) / %.1f MiB total in %.3fs",
            len(block_hashes),
            len(block_hashes) * self.bytes_per_block / (1 << 20),
            time.perf_counter() - t0,
        )

    def close(self) -> None:
        try:
            self.store.close()
        except Exception:
            pass
