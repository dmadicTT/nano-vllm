"""Prefill-Decode disaggregated engine for nano-vllm.

This extends nano-vllm's LLMEngine to support PD disaggregation
using Mooncake Transfer Engine for KV cache migration.
"""

import os
import sys
import pickle
import struct
import socket
import threading
from typing import Optional, List, Dict, Tuple
from dataclasses import fields

import numpy as np

# Add nano-vllm to path
sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm")

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

from pd_config import PDConfig
from mooncake_kv_transfer import MooncakeKVTransfer, SimpleKVCacheSerializer


class PDSequence(Sequence):
    """Extended Sequence with PD transfer tracking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transfer_complete = False
        self.remote_block_ids = []  # Block IDs assigned on decode node
        self.request_id = f"req_{self.seq_id}"


class PDScheduler(Scheduler):
    """Scheduler that handles PD handoff."""
    
    def __init__(self, config: Config, pd_config: PDConfig):
        super().__init__(config)
        self.pd_config = pd_config
        self.pending_prefill: List[PDSequence] = []  # Ready for transfer
        self.pending_decode: List[PDSequence] = []   # Received from prefill
    
    def schedule(self):
        """Override schedule to handle PD phases."""
        if self.pd_config.is_prefill():
            return self._schedule_prefill()
        else:
            return self._schedule_decode()
    
    def _schedule_prefill(self):
        """Prefill node: do prefill, mark for transfer when done."""
        seqs, is_prefill = super().schedule()
        
        # After prefill completes, sequences move to running
        # When prefill is fully done, move to pending_prefill for transfer
        if not is_prefill:
            for seq in list(self.running):
                if isinstance(seq, PDSequence) and not seq.transfer_complete:
                    if seq.num_cached_tokens >= seq.num_prompt_tokens:
                        seq.transfer_complete = True
                        self.running.remove(seq)
                        self.pending_prefill.append(seq)
        
        return seqs, is_prefill
    
    def _schedule_decode(self):
        """Decode node: process sequences that have been transferred."""
        # First, move any pending_decode sequences to running
        while self.pending_decode:
            seq = self.pending_decode.pop(0)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
        
        return super().schedule()
    
    def add_transferred_sequence(self, seq: PDSequence):
        """Add a sequence that was transferred from prefill node."""
        self.pending_decode.append(seq)


class PDModelRunner:
    """Wrapper around ModelRunner that adds KV cache access."""
    
    def __init__(self, model_runner: ModelRunner, pd_config: PDConfig):
        self.model_runner = model_runner
        self.pd_config = pd_config
        self.kv_cache = model_runner.kv_cache  # [2, num_layers, num_blocks, block_size, ...]
        
        # Calculate block size in bytes
        self.block_size_bytes = self.kv_cache.nbytes // self.kv_cache.shape[2]
        print(f"[PDModelRunner] KV cache shape: {self.kv_cache.shape}")
        print(f"[PDModelRunner] Block size: {self.block_size_bytes} bytes")
    
    def get_kv_cache_blocks(self, block_ids: List[int]) -> np.ndarray:
        """Extract KV cache blocks by ID."""
        return self.kv_cache[:, :, block_ids]
    
    def set_kv_cache_blocks(self, block_ids: List[int], kv_data: np.ndarray):
        """Set KV cache blocks by ID."""
        self.kv_cache[:, :, block_ids] = kv_data
    
    @property
    def num_blocks(self) -> int:
        return self.kv_cache.shape[2]
    
    @property
    def num_layers(self) -> int:
        return self.kv_cache.shape[1]


class PDEngine:
    """Prefill-Decode disaggregated engine."""
    
    def __init__(self, pd_config: PDConfig):
        self.pd_config = pd_config
        
        # Build nano-vllm config
        config_fields_set = {field.name for field in fields(Config)}
        config_kwargs = {
            k: v for k, v in pd_config.__dict__.items()
            if k in config_fields_set
        }
        self.config = Config(pd_config.model_path, **config_kwargs)
        
        # Override block size
        PDSequence.block_size = self.config.kvcache_block_size
        
        # Initialize model runner (CPU-only adaptation)
        self.model_runner = ModelRunner(self.config, 0, [])
        
        # Wrap with PD access
        self.pd_runner = PDModelRunner(self.model_runner, pd_config)
        
        # Initialize scheduler
        self.scheduler = PDScheduler(self.config, pd_config)
        
        # Initialize Mooncake transfer
        local_host = pd_config.prefill_host if pd_config.is_prefill() else pd_config.decode_host
        self.mooncake = MooncakeKVTransfer(
            role=pd_config.role,
            local_hostname=local_host,
            protocol=pd_config.mooncake_protocol,
        )
        
        # Register KV cache with Mooncake
        self.mooncake.register_kv_cache(self.pd_runner.kv_cache)
        
        # Prefill node: listen for transfer requests
        if pd_config.is_prefill():
            self._start_transfer_listener()
        
        print(f"[PDEngine] {pd_config.role} node ready")
    
    def _start_transfer_listener(self):
        """Start a thread to listen for KV transfer requests from decode node."""
        def listener():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.pd_config.prefill_host, self.pd_config.prefill_port + 100))
            sock.listen(5)
            print(f"[PDEngine] Transfer listener on port {self.pd_config.prefill_port + 100}")
            
            while True:
                conn, addr = sock.accept()
                try:
                    data = conn.recv(65536)
                    if not data:
                        continue
                    
                    request = SimpleKVCacheSerializer.deserialize_transfer_request(data)
                    self._handle_transfer_request(request, conn)
                except Exception as e:
                    print(f"[PDEngine] Transfer error: {e}")
                finally:
                    conn.close()
        
        self._listener_thread = threading.Thread(target=listener, daemon=True)
        self._listener_thread.start()
    
    def _handle_transfer_request(self, request: dict, conn: socket.socket):
        """Handle a KV transfer request from decode node."""
        request_id = request["request_id"]
        remote_block_ids = request["remote_block_ids"]
        
        # Find the sequence
        seq = None
        for s in self.scheduler.pending_prefill:
            if s.request_id == request_id:
                seq = s
                break
        
        if seq is None:
            conn.sendall(b"ERROR: Sequence not found")
            return
        
        # Send KV cache blocks
        success = self.mooncake.send_kv_cache(
            remote_host=self.pd_config.decode_host,
            remote_port=self.mooncake.get_rpc_port(),  # Will be overridden
            local_block_ids=seq.block_table,
            remote_block_ids=remote_block_ids,
            block_size_bytes=self.pd_runner.block_size_bytes,
        )
        
        if success:
            conn.sendall(b"OK")
            # Remove from pending
            self.scheduler.pending_prefill.remove(seq)
        else:
            conn.sendall(b"ERROR: Transfer failed")
    
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Add a new request."""
        if isinstance(prompt, str):
            # We need tokenizer - initialize it
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.model, use_fast=True)
            prompt = tokenizer.encode(prompt)
        
        seq = PDSequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.request_id
    
    def step(self):
        """Run one scheduling step."""
        seqs, is_prefill = self.scheduler.schedule()
        
        if not seqs:
            return [], 0
        
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens
    
    def is_finished(self):
        return self.scheduler.is_finished()
    
    def get_pending_transfers(self) -> List[PDSequence]:
        """Get sequences ready for transfer (prefill node only)."""
        return list(self.scheduler.pending_prefill)
    
    def transfer_sequence(self, request_id: str, decode_rpc_port: int) -> bool:
        """Transfer a sequence's KV cache to decode node.
        
        Args:
            request_id: ID of sequence to transfer
            decode_rpc_port: Mooncake RPC port of decode node
        """
        if self.pd_config.is_decode():
            raise RuntimeError("Only prefill node can initiate transfers")
        
        seq = None
        for s in self.scheduler.pending_prefill:
            if s.request_id == request_id:
                seq = s
                break
        
        if seq is None:
            print(f"[PDEngine] Sequence {request_id} not found for transfer")
            return False
        
        # Allocate blocks on decode node first (via control channel)
        # For simplicity, we'll use the same block IDs
        remote_block_ids = seq.block_table
        
        success = self.mooncake.send_kv_cache(
            remote_host=self.pd_config.decode_host,
            remote_port=decode_rpc_port,
            local_block_ids=seq.block_table,
            remote_block_ids=remote_block_ids,
            block_size_bytes=self.pd_runner.block_size_bytes,
        )
        
        if success:
            self.scheduler.pending_prefill.remove(seq)
            print(f"[PDEngine] Transferred {request_id} to decode node")
        
        return success
    
    def receive_sequence(
        self,
        request_id: str,
        prefill_rpc_port: int,
        block_table: List[int],
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
    ) -> bool:
        """Receive a sequence's KV cache from prefill node.
        
        Args:
            request_id: ID of sequence
            prefill_rpc_port: Mooncake RPC port of prefill node
            block_table: Block IDs allocated for this sequence
            prompt_token_ids: Original prompt tokens
            sampling_params: Sampling parameters
        """
        if self.pd_config.is_prefill():
            raise RuntimeError("Only decode node can receive transfers")
        
        # Receive KV cache blocks
        success = self.mooncake.receive_kv_cache(
            remote_host=self.pd_config.prefill_host,
            remote_port=prefill_rpc_port,
            local_block_ids=block_table,
            remote_block_ids=block_table,
            block_size_bytes=self.pd_runner.block_size_bytes,
        )
        
        if not success:
            return False
        
        # Create sequence with received KV cache
        seq = PDSequence(prompt_token_ids, sampling_params)
        seq.block_table = block_table
        seq.num_cached_tokens = len(prompt_token_ids)
        seq.transfer_complete = True
        seq.status = SequenceStatus.RUNNING
        
        self.scheduler.add_transferred_sequence(seq)
        print(f"[PDEngine] Received {request_id} from prefill node")
        return True
    
    def generate(self, prompts: List[str], sampling_params: SamplingParams) -> List[Dict]:
        """Generate completions (for standalone testing)."""
        from tqdm.auto import tqdm
        from time import perf_counter
        
        pbar = tqdm(total=len(prompts), desc="Generating")
        request_ids = []
        for prompt in prompts:
            rid = self.add_request(prompt, sampling_params)
            request_ids.append(rid)
        
        outputs = {}
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            
            # Handle transfers on prefill node
            if self.pd_config.is_prefill():
                for seq in self.get_pending_transfers():
                    # In real setup, decode node would request this
                    pass
            
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        
        pbar.close()
        return outputs
