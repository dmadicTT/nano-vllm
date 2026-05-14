import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # PD disaggregation fields.
    role: str = "colocated"  # 'prefill' | 'decode' | 'colocated'
    # Where to reach the Mooncake stack (master + http metadata server). Only used
    # when role != 'colocated'.
    mooncake_master_addr: str = "127.0.0.1:50051"
    mooncake_metadata_server: str = "http://127.0.0.1:8081/metadata"
    mooncake_protocol: str = "tcp"  # 'rdma' or 'tcp'
    mooncake_rdma_devices: str = ""
    mooncake_local_hostname: str = "127.0.0.1:12001"
    mooncake_global_segment_size: int = 1 << 30  # 1 GiB
    mooncake_local_buffer_size: int = 256 << 20  # 256 MiB

    def __post_init__(self):
        assert os.path.isdir(self.model)
        # block_size is only a paging granularity for the CPU attention path;
        # any positive value works.
        assert self.kvcache_block_size > 0
        # CPU-only build: tensor parallelism across CPU workers isn't useful
        # (no per-shard speedup; would just duplicate the model). Keep the
        # field for compatibility with the dist-based linear layers, but
        # require it to stay at 1.
        assert self.tensor_parallel_size == 1, (
            "this CPU-only build supports tensor_parallel_size=1 only"
        )
        assert self.role in ("prefill", "decode", "colocated")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
