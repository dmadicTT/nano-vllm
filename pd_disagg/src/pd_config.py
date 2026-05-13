"""Configuration for PD disaggregation with nano-vllm and Mooncake."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PDConfig:
    """Config for prefill-decode disaggregation."""
    role: str  # 'prefill' or 'decode'
    model_path: str
    
    # Network config
    prefill_host: str = "127.0.0.1"
    prefill_port: int = 8010
    decode_host: str = "127.0.0.1"
    decode_port: int = 8020
    proxy_port: int = 8000
    
    # Mooncake config
    mooncake_protocol: str = "tcp"  # tcp or rdma
    
    # Model config
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = True  # Required for CPU
    
    # KV cache transfer config
    block_size: int = 256
    
    def is_prefill(self) -> bool:
        return self.role == "prefill"
    
    def is_decode(self) -> bool:
        return self.role == "decode"
