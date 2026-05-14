"""CPU-compatible model runner for nano-vllm PD testing.

Uses real HuggingFace model weights for coherent text generation.
Falls back to TinyLM (random weights) if no real model is available.
"""

import os
import sys
import pickle
import numpy as np

sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm")

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# Minimal copies to avoid flash_attn dependency
from dataclasses import dataclass, field
from typing import List, Optional

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
        # Store past_key_values for real model inference
        self.past_key_values = None
    
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
    """Tiny language model for CPU testing (fallback when no real model)."""
    
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
        self.lm_head.weight = self.embed.weight  # Tie weights
    
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor = None):
        x = self.embed(input_ids)
        if positions is not None:
            pos_embed = self.embed(positions % self.embed.num_embeddings)
            x = x + pos_embed
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        x = self.norm(x)
        return self.lm_head(x)


class CPUModelRunner:
    """CPU-based model runner using real HF weights or TinyLM fallback."""
    
    def __init__(self, config: Config, rank: int = 0, events=None):
        self.config = config
        self.rank = rank
        self.block_size = config.kvcache_block_size
        self.use_real_model = False
        self.model = None
        self.tokenizer = None
        
        # Load or create model
        self._init_model(config)
        
        # Allocate KV cache (simplified - just track shapes)
        self.kv_cache = self._allocate_kv_cache()
        
        # Track which blocks are used by which sequences
        self.block_to_seq = {}  # block_id -> seq_id
    
    def _init_model(self, config: Config):
        """Initialize the model - try real weights first, fallback to TinyLM."""
        model_path = config.model
        
        # Try to load real model
        try:
            print(f"[CPUModelRunner] Attempting to load real model from: {model_path}")
            hf_config = AutoConfig.from_pretrained(model_path)
            
            # Check if model files exist (not just config)
            if os.path.exists(model_path):
                has_weights = any(f.endswith(('.bin', '.safetensors', '.pt', '.pth')) 
                                  for f in os.listdir(model_path))
            else:
                # Remote model - try loading
                has_weights = True
            
            if has_weights:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                )
                self.model.eval()
                self.use_real_model = True
                self.vocab_size = hf_config.vocab_size
                self.hidden_size = getattr(hf_config, 'hidden_size', 768)
                self.num_layers = getattr(hf_config, 'num_hidden_layers', 12)
                self.num_kv_heads = getattr(hf_config, 'num_key_value_heads', 
                                             getattr(hf_config, 'num_attention_heads', 12))
                self.head_dim = self.hidden_size // getattr(hf_config, 'num_attention_heads', 12)
                print(f"[CPUModelRunner] Loaded real model: {model_path}")
                print(f"[CPUModelRunner]  Layers: {self.num_layers}, Hidden: {self.hidden_size}, "
                      f"KV heads: {self.num_kv_heads}, Head dim: {self.head_dim}")
                return
        except Exception as e:
            print(f"[CPUModelRunner] Failed to load real model: {e}")
        
        # Fallback to TinyLM
        print(f"[CPUModelRunner] Falling back to TinyLM (random weights)")
        try:
            hf_config = AutoConfig.from_pretrained(model_path)
            vocab_size = hf_config.vocab_size
            hidden_size = getattr(hf_config, 'hidden_size', 64)
            num_layers = getattr(hf_config, 'num_hidden_layers', 2)
        except:
            vocab_size = 32000
            hidden_size = 64
            num_layers = 2
        
        self.model = TinyLM(vocab_size, hidden_size, num_layers)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_kv_heads = 1
        self.head_dim = hidden_size
        self.model.eval()
    
    def _allocate_kv_cache(self):
        """Allocate a mock KV cache for transfer verification."""
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim
        
        block_bytes = 2 * self.num_layers * self.block_size * num_kv_heads * head_dim * 4
        max_blocks = 64
        
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
    
    def serialize_past_kv(self, past_key_values) -> bytes:
        """Serialize past_key_values to bytes. Handles both tuple and DynamicCache."""
        if past_key_values is None:
            return b''
        # Handle DynamicCache (newer transformers)
        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
            serialized = []
            for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
                serialized.append((k.cpu().numpy(), v.cpu().numpy()))
            return pickle.dumps({"type": "dynamic_cache", "layers": serialized})
        # Handle tuple format (older transformers) - may have 3 elements (k, v, None)
        serialized = []
        for layer_kv in past_key_values:
            if len(layer_kv) >= 2:
                k, v = layer_kv[0], layer_kv[1]
                serialized.append((k.cpu().numpy(), v.cpu().numpy()))
        return pickle.dumps({"type": "tuple", "layers": serialized})
    
    def deserialize_past_kv(self, data: bytes):
        """Deserialize bytes back to past_key_values as DynamicCache."""
        if not data:
            return None
        parsed = pickle.loads(data)
        serialized = parsed["layers"]
        
        result = []
        for k_np, v_np in serialized:
            result.append((torch.from_numpy(k_np), torch.from_numpy(v_np)))
        
        # Always return DynamicCache for compatibility with newer transformers
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for i, (k, v) in enumerate(result):
            cache.update(k, v, i)
        return cache
    
    def call(self, method_name, *args):
        method = getattr(self, method_name, None)
        return method(*args)
    
    def run(self, seqs: list, is_prefill: bool):
        if not seqs:
            return []
        if is_prefill:
            return self._run_prefill(seqs)
        else:
            return self._run_decode(seqs)
    
    def _run_prefill(self, seqs: list):
        """Run prefill phase using real model or TinyLM."""
        results = []
        
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            tokens = seq.token_ids[start:end]
            
            if not tokens:
                results.append(seq.last_token)
                continue
            
            if self.use_real_model:
                # Use real model with generate
                input_ids = torch.tensor([tokens], dtype=torch.long)
                
                with torch.no_grad():
                    outputs = self.model(input_ids, use_cache=True)
                    logits = outputs.logits
                    # Store past_key_values for decode phase
                    seq.past_key_values = outputs.past_key_values
                
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
            else:
                # TinyLM path
                input_ids = torch.tensor([tokens], dtype=torch.long)
                positions = torch.tensor([list(range(start, end))], dtype=torch.long)
                
                with torch.no_grad():
                    logits = self.model(input_ids, positions)
                
                for i, block_id in enumerate(seq.block_table):
                    self.block_to_seq[block_id] = seq.seq_id
                    self.kv_cache[:, :, block_id, :, :, :] = block_id
                
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
            
            results.append(next_token)
        
        return results
    
    def _run_decode(self, seqs: list):
        """Run decode phase using real model or TinyLM."""
        results = []
        
        for seq in seqs:
            token = seq.last_token
            
            if self.use_real_model and seq.past_key_values is not None:
                # Use real model with past_key_values for efficient decode
                input_ids = torch.tensor([[token]], dtype=torch.long)
                
                with torch.no_grad():
                    outputs = self.model(
                        input_ids,
                        past_key_values=seq.past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits
                    # Update past_key_values for next step
                    seq.past_key_values = outputs.past_key_values
                
                next_token_logits = logits[0, -1, :]
                # Apply temperature sampling
                if seq.temperature > 0 and seq.temperature != 1.0:
                    next_token_logits = next_token_logits / seq.temperature
                # Apply repetition penalty
                for token_id in set(seq.token_ids):
                    next_token_logits[token_id] /= 1.2
                # Top-k sampling
                top_k = 50
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1).item()
                next_token = top_k_indices[next_token_idx].item()
            else:
                # TinyLM path or no past_key_values
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
        
        if self.waiting:
            seqs = []
            for seq in list(self.waiting):
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
