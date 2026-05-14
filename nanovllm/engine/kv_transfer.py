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

    `block_ids` is the *local* block table — the index space differs between
    prefill and decode because each side runs its own block manager. The
    sequence-relative block index (0, 1, ...) is encoded in the Mooncake
    key so the receiver knows which slot to fill.
    """

    KEY_FMT = "nanovllm/req/{rid}/blk/{idx}"

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

        log.info(
            "KVTransfer[%s] connected: master=%s metadata=%s protocol=%s "
            "block=%.1fMiB (L=%d, B=%d, S=%d, H=%d, D=%d, dtype=%s)",
            role, master_addr, metadata_server, protocol,
            self.bytes_per_block / (1 << 20),
            self.num_layers, self.num_blocks, self.block_size,
            self.num_kv_heads, self.head_dim, self.dtype,
        )

    # -------- producer side (prefill) --------
    def push(self, request_id: str, block_ids: List[int]) -> None:
        """Publish each block of a request as its own Mooncake key.

        `block_ids` is the prefill node's local block ids; the receiver uses
        its own different block ids and just keys by sequence-relative index.
        """
        t0 = time.perf_counter()
        for seq_idx, bid in enumerate(block_ids):
            # Gather the (possibly strided) block into staging — copy_() handles
            # the layout translation, then we ship the contiguous bytes.
            self.staging.copy_(self.kv_cache[:, :, bid, :, :, :])
            key = self.KEY_FMT.format(rid=request_id, idx=seq_idx)
            # `put` copies bytes into Mooncake's mounted segment, so we can
            # reuse the staging buffer immediately for the next block.
            # Pull raw bytes out of the storage. This works for bfloat16 / fp16
            # which numpy doesn't natively support.
            payload = ctypes.string_at(self.staging.data_ptr(), self.bytes_per_block)
            t_put = time.perf_counter()
            rc = self.store.put(key, payload)
            if rc != 0:
                raise RuntimeError(f"Mooncake put({key}) failed rc={rc}")
            self.bytes_pushed += self.bytes_per_block
            self.blocks_pushed += 1
            log.info(
                "push  %s  local_bid=%d  %.1f MiB  put=%.3fs",
                key, bid, self.bytes_per_block / (1 << 20),
                time.perf_counter() - t_put,
            )
        log.info(
            "push  %s: %d block(s) / %.1f MiB total in %.3fs",
            request_id, len(block_ids),
            len(block_ids) * self.bytes_per_block / (1 << 20),
            time.perf_counter() - t0,
        )

    # -------- consumer side (decode) --------
    def pull(self, request_id: str, block_ids: List[int],
             *, wait_timeout: float = 30.0, poll_interval: float = 0.02) -> None:
        """Populate the local KV cache slots at `block_ids` with bytes for `request_id`.

        Blocks until all expected keys are available (or `wait_timeout` is hit).
        """
        t0 = time.perf_counter()
        deadline = time.monotonic() + wait_timeout
        for seq_idx, bid in enumerate(block_ids):
            key = self.KEY_FMT.format(rid=request_id, idx=seq_idx)
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
            # Materialize bytes into the staging tensor, then scatter into the
            # right block slot of the local kv_cache.
            tensor = torch.frombuffer(bytearray(data), dtype=self.dtype).view(*self.block_shape)
            self.kv_cache[:, :, bid, :, :, :].copy_(tensor)
            self.bytes_pulled += self.bytes_per_block
            self.blocks_pulled += 1
            log.info(
                "pull  %s  local_bid=%d  %.1f MiB  wait=%.3fs get=%.3fs",
                key, bid, self.bytes_per_block / (1 << 20), wait_s, get_s,
            )
        log.info(
            "pull  %s: %d block(s) / %.1f MiB total in %.3fs",
            request_id, len(block_ids),
            len(block_ids) * self.bytes_per_block / (1 << 20),
            time.perf_counter() - t0,
        )

    def remove(self, request_id: str, num_blocks: int) -> None:
        """Best-effort cleanup of a request's published blocks.

        Called by the producer once it knows the consumer is done.
        """
        for seq_idx in range(num_blocks):
            key = self.KEY_FMT.format(rid=request_id, idx=seq_idx)
            try:
                self.store.remove(key)
            except Exception:
                # The key may already have been evicted; ignore.
                pass

    def close(self) -> None:
        try:
            self.store.close()
        except Exception:
            pass
