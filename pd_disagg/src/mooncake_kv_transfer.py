"""Mooncake-based KV cache transfer for nano-vllm PD disaggregation.

This is a simplified adapter that uses Mooncake Transfer Engine
to move KV caches from prefill to decode nodes.
"""

import os
import sys
import time
import pickle
import struct
import threading
from typing import Optional, List, Dict, Tuple
import numpy as np

# Add Mooncake build to path
MOONCAKE_BUILD = "/root/.openclaw/workspace/Mooncake/build"
sys.path.insert(0, f"{MOONCAKE_BUILD}/mooncake-integration")

# Set LD_LIBRARY_PATH before importing
os.environ["LD_LIBRARY_PATH"] = (
    f"{MOONCAKE_BUILD}/mooncake-transfer-engine/src:"
    f"{MOONCAKE_BUILD}/mooncake-common:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

# Use ctypes to preload libraries
import ctypes
ctypes.CDLL(f"{MOONCAKE_BUILD}/mooncake-common/libasio.so", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(f"{MOONCAKE_BUILD}/mooncake-transfer-engine/src/libtransfer_engine.so", mode=ctypes.RTLD_GLOBAL)

import engine as mooncake_engine


class MooncakeKVTransfer:
    """Handles KV cache transfer between prefill and decode nodes."""
    
    def __init__(
        self,
        role: str,  # 'prefill' or 'decode'
        local_hostname: str = "127.0.0.1",
        protocol: str = "rdma",
    ):
        self.role = role
        self.local_hostname = local_hostname
        self.protocol = protocol
        
        self.engine = mooncake_engine.TransferEngine()
        # Use RDMA device eth0-rxe
        ret = self.engine.initialize(self.local_hostname, "P2PHANDSHAKE", protocol, "eth0-rxe")
        if ret != 0:
            raise RuntimeError(f"Mooncake Transfer Engine initialization failed: {ret}")
        
        self.rpc_port = self.engine.get_rpc_port()
        print(f"[Mooncake] {role} node initialized at {local_hostname}:{self.rpc_port}")
        
        # Track registered memory regions
        self.registered_addrs: List[int] = []
        self.registered_lens: List[int] = []
        self._lock = threading.Lock()
    
    def register_kv_cache(self, kv_cache: np.ndarray):
        """Register a KV cache tensor for remote transfer.
        
        Args:
            kv_cache: numpy array of shape [2, num_layers, num_blocks, block_size, ...]
        """
        base_addr = kv_cache.ctypes.data
        size = kv_cache.nbytes
        
        with self._lock:
            self.registered_addrs.append(base_addr)
            self.registered_lens.append(size)
            
            ret = self.engine.batch_register_memory([base_addr], [size])
            if ret != 0:
                raise RuntimeError(f"Failed to register KV cache memory: {ret}")
        
        print(f"[Mooncake] Registered KV cache: addr={base_addr}, size={size}")
        return base_addr
    
    def unregister_all(self):
        """Unregister all memory regions."""
        with self._lock:
            if self.registered_addrs:
                self.engine.batch_unregister_memory(self.registered_addrs)
                self.registered_addrs.clear()
                self.registered_lens.clear()
    
    def send_kv_cache(
        self,
        remote_host: str,
        remote_port: int,
        local_block_ids: List[int],
        remote_block_ids: List[int],
        block_size_bytes: int,
        remote_base_addr: int = None,
    ) -> bool:
        """Send KV cache blocks to remote decode node.
        
        Args:
            remote_host: Hostname of decode node
            remote_port: RPC port of decode node
            local_block_ids: Block IDs in local KV cache to send
            remote_block_ids: Corresponding block IDs in remote KV cache
            block_size_bytes: Size of each block in bytes
            remote_base_addr: Remote's registered memory base address (for same-process testing)
        """
        remote_session = f"{remote_host}:{remote_port}"
        
        if not self.registered_addrs:
            print("[Mooncake] No registered memory")
            return False
        
        local_base = self.registered_addrs[0]
        # For cross-process, Mooncake handles address translation via metadata
        # For same-process testing, we need the remote's actual address
        dst_base = remote_base_addr if remote_base_addr is not None else local_base
        
        # Build per-block transfer params
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        
        for lb, rb in zip(local_block_ids, remote_block_ids):
            src_ptrs.append(local_base + lb * block_size_bytes)
            dst_ptrs.append(dst_base + rb * block_size_bytes)
            lengths.append(block_size_bytes)
        
        print(f"[Mooncake] Sending {len(local_block_ids)} blocks to {remote_session}...")
        ret = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        
        if ret != 0:
            print(f"[Mooncake] Transfer failed: {ret}")
            return False
        
        print(f"[Mooncake] Sent {len(local_block_ids)} blocks to {remote_session}")
        return True
    
    def receive_kv_cache(
        self,
        remote_host: str,
        remote_port: int,
        local_block_ids: List[int],
        remote_block_ids: List[int],
        block_size_bytes: int,
        remote_base_addr: int = None,
    ) -> bool:
        """Receive KV cache blocks from remote prefill node.
        
        This is the decode-side counterpart to send_kv_cache.
        """
        remote_session = f"{remote_host}:{remote_port}"
        
        if not self.registered_addrs:
            print("[Mooncake] No registered memory")
            return False
        
        local_base = self.registered_addrs[0]
        src_base = remote_base_addr if remote_base_addr is not None else local_base
        
        # Build per-block transfer params (reverse direction for read)
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        
        for lb, rb in zip(local_block_ids, remote_block_ids):
            src_ptrs.append(src_base + rb * block_size_bytes)
            dst_ptrs.append(local_base + lb * block_size_bytes)
            lengths.append(block_size_bytes)
        
        print(f"[Mooncake] Receiving {len(local_block_ids)} blocks from {remote_session}...")
        ret = self.engine.batch_transfer_sync_read(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        
        if ret != 0:
            print(f"[Mooncake] Receive failed: {ret}")
            return False
        
        print(f"[Mooncake] Received {len(local_block_ids)} blocks from {remote_session}")
        return True
    
    def get_rpc_port(self) -> int:
        return self.rpc_port
    
    def __del__(self):
        self.unregister_all()


class SimpleKVCacheSerializer:
    """Serialize/deserialize KV cache metadata for transfer."""
    
    @staticmethod
    def serialize_transfer_request(
        request_id: str,
        local_block_ids: List[int],
        remote_block_ids: List[int],
        block_size_bytes: int,
        num_layers: int,
    ) -> bytes:
        """Serialize a transfer request."""
        data = {
            "request_id": request_id,
            "local_block_ids": local_block_ids,
            "remote_block_ids": remote_block_ids,
            "block_size_bytes": block_size_bytes,
            "num_layers": num_layers,
        }
        return pickle.dumps(data)
    
    @staticmethod
    def deserialize_transfer_request(data: bytes) -> dict:
        """Deserialize a transfer request."""
        return pickle.loads(data)
