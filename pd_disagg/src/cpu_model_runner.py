"""CPU-compatible model runner for nano-vllm PD testing.

This is a simplified version that doesn't require CUDA/flash-attn.
It uses a tiny random model just to test the PD pipeline.
"""

import os
import sys
import pickle
import numpy as np

sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm")

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig

# Avoid importing nano-vllm modules that require flash-attn
# from nanovllm.config import Config
# from nanovllm.sampling_params import SamplingParams
# from nanovllm.engine.sequence import Sequence

# Minimal copies to avoid flash_attn dependency
from dataclasses import dataclass, field
from typing import List

@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 256
    ignore_eos: bool = False

class SequenceStatus:
    WAITING = 0
    RUNNING = 1
    FINISHED = 2

class Sequence:
    block_size = 256
    _counter = 0
    
    def __init__(self, token_ids: list, sampling_params=None):
        Sequence._counter += 1
        self.seq_id = Sequence._counter
        self.status = SequenceStatus.WAITING
        self.token_ids = list(token_ids)
        self.last_token = token_ids[-1] if token_ids else 0
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature if sampling_params else 1.0
        self.max_tokens = sampling_params.max_tokens if sampling_params else 256
        self.ignore_eos = sampling_params.ignore_eos if sampling_params else False
    
    def __len__(self):
        return self.num_tokens
    
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED
    
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens
    
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]
    
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size
    
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size
    
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
    
    def block(self, i):
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1


class TinyLM(nn.Module):
    """Tiny language model for CPU testing."""
    
    def __init__(self, vocab_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed.weight
    
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor = None):
        x = self.embed(input_ids)
        
        # Add positional encoding
        if positions is not None:
            # Simple learned position embeddings
            pos_embed = self.embed(positions % self.embed.num_embeddings)
            x = x + pos_embed
        
        # Create causal mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        
        x = self.norm(x)
        return self.lm_head(x)


class CPUModelRunner:
    """CPU-based model runner for testing PD disaggregation."""
    
    def __init__(self, config: Config, rank: int = 0, events=None):
        self.config = config
        self.rank = rank
        self.block_size = config.kvcache_block_size
        
        # Load or create model
        self._init_model(config)
        
        # Allocate KV cache (simplified - just track shapes)
        self.kv_cache = self._allocate_kv_cache()
        
        # Track which blocks are used by which sequences
        self.block_to_seq = {}  # block_id -> seq_id
    
    def _init_model(self, config: Config):
        """Initialize the model."""
        try:
            # Try to load real config
            hf_config = AutoConfig.from_pretrained(config.model)
            vocab_size = hf_config.vocab_size
            hidden_size = getattr(hf_config, 'hidden_size', 64)
            num_layers = getattr(hf_config, 'num_hidden_layers', 2)
        except:
            # Fallback
            vocab_size = 32000
            hidden_size = 64
            num_layers = 2
        
        self.model = TinyLM(vocab_size, hidden_size, num_layers)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Move to CPU
        self.model.eval()
    
    def _allocate_kv_cache(self):
        """Allocate a mock KV cache."""
        # For CPU testing, we use a simplified KV cache
        # Shape: [2 (k/v), num_layers, num_blocks, block_size, hidden_size]
        num_kv_heads = 1  # Simplified
        head_dim = self.hidden_size
        
        # Calculate num blocks based on memory
        block_bytes = 2 * self.num_layers * self.block_size * num_kv_heads * head_dim * 4  # float32
        max_blocks = 64  # Increased for longer prompts
        
        # Use numpy array for KV cache (will be registered with Mooncake separately)
        kv_cache = np.zeros(
            (2, self.num_layers, max_blocks, self.block_size, num_kv_heads, head_dim),
            dtype=np.float32
        )
        return kv_cache
    
    def get_kv_cache_bytes(self, block_ids):
        """Get raw bytes for specified block IDs."""
        blocks = []
        for block_id in block_ids:
            blocks.append(self.kv_cache[:, :, block_id, :, :, :].tobytes())
        return b''.join(blocks)
    
    def set_kv_cache_bytes(self, block_ids, data_bytes, block_size_bytes):
        """Set KV cache from raw bytes for specified block IDs."""
        for i, block_id in enumerate(block_ids):
            block_data = data_bytes[i * block_size_bytes:(i + 1) * block_size_bytes]
            block_array = np.frombuffer(block_data, dtype=np.float32)
            target_shape = self.kv_cache[:, :, block_id, :, :, :].shape
            self.kv_cache[:, :, block_id, :, :, :] = block_array.reshape(target_shape)
    
    def call(self, method_name, *args):
        """Call a method (compatibility with nano-vllm's multiprocessing)."""
        method = getattr(self, method_name, None)
        return method(*args)
    
    def run(self, seqs: list, is_prefill: bool):
        """Run model on sequences."""
        if not seqs:
            return []
        
        # Prepare inputs
        if is_prefill:
            return self._run_prefill(seqs)
        else:
            return self._run_decode(seqs)
    
    def _run_prefill(self, seqs: list):
        """Run prefill phase."""
        results = []
        
        for seq in seqs:
            # Get tokens to process
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            tokens = seq.token_ids[start:end]
            
            if not tokens:
                # No new tokens to process
                results.append(seq.last_token)
                continue
            
            # Run through model
            input_ids = torch.tensor([tokens], dtype=torch.long)
            positions = torch.tensor([list(range(start, end))], dtype=torch.long)
            
            with torch.no_grad():
                logits = self.model(input_ids, positions)
            
            # Store KV cache (simplified - just mark blocks as used)
            for i, block_id in enumerate(seq.block_table):
                self.block_to_seq[block_id] = seq.seq_id
                # Fill with dummy data based on block_id for verification
                self.kv_cache[:, :, block_id, :, :, :] = block_id
            
            # Return last token logits for sampling
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            results.append(next_token)
        
        return results
    
    def _run_decode(self, seqs: list):
        """Run decode phase."""
        results = []
        
        for seq in seqs:
            # Get last token
            token = seq.last_token
            pos = len(seq) - 1
            
            input_ids = torch.tensor([[token]], dtype=torch.long)
            positions = torch.tensor([[pos]], dtype=torch.long)
            
            with torch.no_grad():
                logits = self.model(input_ids, positions)
            
            next_token_logits = logits[0, 0, :]
            next_token = torch.argmax(next_token_logits).item()
            results.append(next_token)
        
        return results
    
    def exit(self):
        """Cleanup."""
        pass


class SimpleScheduler:
    """Minimal scheduler for CPU testing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.waiting = []
        self.running = []
        self.finished = []
        self.block_manager = SimpleBlockManager(config)
    
    def add(self, seq: Sequence):
        self.waiting.append(seq)
    
    def schedule(self):
        if not self.running and not self.waiting:
            return [], False
        
        # If there are waiting sequences, schedule them for prefill
        if self.waiting:
            seqs = []
            for seq in list(self.waiting):
                # Allocate at least 1 block
                num_blocks_needed = max(1, seq.num_blocks)
                block_ids = self.block_manager.allocate(num_blocks_needed)
                if block_ids is None:
                    break
                seq.block_table = block_ids
                seq.num_scheduled_tokens = seq.num_prompt_tokens - seq.num_cached_tokens
                seq.status = SequenceStatus.RUNNING
                seqs.append(seq)
                self.waiting.remove(seq)
                self.running.append(seq)
            return seqs, True
        
        # Otherwise schedule running sequences for decode
        seqs = []
        for seq in list(self.running):
            if seq.is_finished:
                self.running.remove(seq)
                self.finished.append(seq)
                continue
            seq.num_scheduled_tokens = 1
            seqs.append(seq)
        return seqs, False
    
    def postprocess(self, seqs, token_ids, is_prefill):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            seq.num_cached_tokens = seq.num_tokens
            
            # Check finish conditions
            if token_id == self.config.eos and not seq.ignore_eos:
                seq.status = SequenceStatus.FINISHED
            elif seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
            
            if seq.is_finished:
                self.running.remove(seq)
                self.finished.append(seq)
    
    def is_finished(self):
        return not self.waiting and not self.running


class SimpleBlockManager:
    """Simple block manager for CPU testing."""
    
    def __init__(self, config: Config):
        self.num_blocks = config.num_kvcache_blocks if config.num_kvcache_blocks > 0 else 1024
        self.free_blocks = list(range(self.num_blocks))
        self.block_size = config.kvcache_block_size
    
    def allocate(self, num_blocks: int):
        if len(self.free_blocks) < num_blocks:
            return None
        return [self.free_blocks.pop(0) for _ in range(num_blocks)]
    
    def free(self, block_ids: list):
        for bid in block_ids:
            if bid not in self.free_blocks:
                self.free_blocks.append(bid)


class CPULLMEngine:
    """CPU-based LLM engine for PD testing."""
    
    def __init__(self, model_path: str, **kwargs):
        self.config = Config(model_path, **kwargs)
        
        Sequence.block_size = self.config.kvcache_block_size
        
        self.model_runner = CPUModelRunner(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.config.eos = self.tokenizer.eos_token_id or 2
        
        self.scheduler = SimpleScheduler(self.config)
    
    def add_request(self, prompt, sampling_params):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq
    
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0
        
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens
    
    def is_finished(self):
        return self.scheduler.is_finished()
