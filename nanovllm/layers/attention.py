import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.utils.context import get_context


def _lazy_import_cuda_kernels():
    """Import flash_attn / triton lazily — they're CUDA-only.

    The CPU disaggregated path never reaches these imports.
    """
    import triton
    import triton.language as tl
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    @triton.jit
    def _store_kvcache_kernel(
        key_ptr, key_stride,
        value_ptr, value_stride,
        k_cache_ptr, v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)

    def _store_kvcache_cuda(key, value, k_cache, v_cache, slot_mapping):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        _store_kvcache_kernel[(N,)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping, D,
        )

    return flash_attn_varlen_func, flash_attn_with_kvcache, _store_kvcache_cuda


# Filled in on first CUDA use to avoid importing triton/flash_attn at import time.
_CUDA_KERNELS = None


def _store_kvcache_cpu(
    key: torch.Tensor,        # [total_tokens, num_kv_heads, head_dim]
    value: torch.Tensor,
    k_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """CPU implementation of paged KV-cache store. Iterates with a vector index_put_."""
    N = key.shape[0]
    if N == 0:
        return
    flat_k = k_cache.view(-1, k_cache.shape[-2], k_cache.shape[-1])
    flat_v = v_cache.view(-1, v_cache.shape[-2], v_cache.shape[-1])
    valid = slot_mapping >= 0
    if valid.all():
        flat_k.index_copy_(0, slot_mapping.to(torch.long), key)
        flat_v.index_copy_(0, slot_mapping.to(torch.long), value)
    else:
        valid_slots = slot_mapping[valid].to(torch.long)
        flat_k.index_copy_(0, valid_slots, key[valid])
        flat_v.index_copy_(0, valid_slots, value[valid])


def _gqa_expand(k: torch.Tensor, n_repeat: int) -> torch.Tensor:
    """Repeat the kv heads so the head count matches q (grouped-query attention)."""
    if n_repeat == 1:
        return k
    # k shape: [..., num_kv_heads, head_dim]
    return k.repeat_interleave(n_repeat, dim=-2)


def _prefill_cpu(
    q: torch.Tensor,            # [total_tokens, num_heads, head_dim]
    k: torch.Tensor,            # [total_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Per-sequence variable-length causal attention with optional prefix cache."""
    cu_q = context.cu_seqlens_q.tolist()
    cu_k = context.cu_seqlens_k.tolist()
    block_tables = context.block_tables
    block_size = k_cache.shape[1] if k_cache.numel() else 0
    n_repeat = num_heads // num_kv_heads
    out = torch.empty_like(q)
    for i in range(len(cu_q) - 1):
        q_start, q_end = cu_q[i], cu_q[i + 1]
        k_start, k_end = cu_k[i], cu_k[i + 1]
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
        if seqlen_q == 0:
            continue
        q_i = q[q_start:q_end]  # [Sq, Hq, D]
        if block_tables is not None and seqlen_k > seqlen_q:
            # Prefix cache: gather all of K/V from k_cache for this sequence.
            bt = block_tables[i]
            bt = bt[bt >= 0].to(torch.long)
            gathered_k = k_cache.index_select(0, bt).view(-1, num_kv_heads, head_dim)[:seqlen_k]
            gathered_v = v_cache.index_select(0, bt).view(-1, num_kv_heads, head_dim)[:seqlen_k]
            k_i, v_i = gathered_k, gathered_v
        else:
            # No prefix cache: K/V are the fresh tokens we just computed.
            k_i = k[k_start:k_end]
            v_i = v[k_start:k_end]
        k_i = _gqa_expand(k_i, n_repeat)
        v_i = _gqa_expand(v_i, n_repeat)
        # Reshape to [1, H, S, D] for sdpa.
        q_b = q_i.transpose(0, 1).unsqueeze(0)
        k_b = k_i.transpose(0, 1).unsqueeze(0)
        v_b = v_i.transpose(0, 1).unsqueeze(0)
        # Build causal mask that anchors the last seqlen_q rows.
        # Position i (0..Sq-1) attends to keys 0..(Sk - Sq + i)
        if seqlen_k == seqlen_q:
            attn = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True, scale=scale)
        else:
            q_pos = torch.arange(seqlen_q).unsqueeze(1) + (seqlen_k - seqlen_q)
            k_pos = torch.arange(seqlen_k).unsqueeze(0)
            mask = (k_pos <= q_pos)  # bool, True == attend
            attn = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=mask, scale=scale)
        out[q_start:q_end] = attn.squeeze(0).transpose(0, 1).contiguous()
    return out


def _decode_cpu(
    q: torch.Tensor,            # [batch, num_heads, head_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    block_tables = context.block_tables
    context_lens = context.context_lens.tolist()
    n_repeat = num_heads // num_kv_heads
    out = torch.empty_like(q)
    for i in range(q.shape[0]):
        ctx_len = context_lens[i]
        bt = block_tables[i]
        bt = bt[bt >= 0].to(torch.long)
        gathered_k = k_cache.index_select(0, bt).view(-1, num_kv_heads, head_dim)[:ctx_len]
        gathered_v = v_cache.index_select(0, bt).view(-1, num_kv_heads, head_dim)[:ctx_len]
        gathered_k = _gqa_expand(gathered_k, n_repeat)
        gathered_v = _gqa_expand(gathered_v, n_repeat)
        # q[i]: [H, D] -> [1, H, 1, D] for SDPA
        q_b = q[i].unsqueeze(1).unsqueeze(0)
        # gathered_k/v: [Sk, H, D] -> [1, H, Sk, D]
        k_b = gathered_k.transpose(0, 1).unsqueeze(0)
        v_b = gathered_v.transpose(0, 1).unsqueeze(0)
        attn = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=False, scale=scale)
        # attn: [1, H, 1, D] -> [H, D]
        out[i] = attn.squeeze(0).squeeze(1)
    return out


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    """Dispatcher kept for backwards compatibility with the GPU path."""
    if key.is_cpu:
        _store_kvcache_cpu(key, value, k_cache, v_cache, slot_mapping)
    else:
        global _CUDA_KERNELS
        if _CUDA_KERNELS is None:
            _CUDA_KERNELS = _lazy_import_cuda_kernels()
        _, _, _store_cuda = _CUDA_KERNELS
        _store_cuda(key, value, k_cache, v_cache, slot_mapping)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if q.is_cpu:
            if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
                _store_kvcache_cpu(k, v, k_cache, v_cache, context.slot_mapping)
            if context.is_prefill:
                return _prefill_cpu(
                    q, k, v, k_cache, v_cache, context,
                    self.scale, self.num_heads, self.num_kv_heads, self.head_dim,
                )
            return _decode_cpu(
                q, k_cache, v_cache, context,
                self.scale, self.num_heads, self.num_kv_heads, self.head_dim,
            )

        # CUDA path — unchanged behavior.
        global _CUDA_KERNELS
        if _CUDA_KERNELS is None:
            _CUDA_KERNELS = _lazy_import_cuda_kernels()
        flash_attn_varlen_func, flash_attn_with_kvcache, _store_cuda = _CUDA_KERNELS
        if k_cache.numel() and v_cache.numel():
            _store_cuda(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            return flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale, causal=True, block_table=context.block_tables,
            )
        return flash_attn_with_kvcache(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=context.context_lens, block_table=context.block_tables,
            softmax_scale=self.scale, causal=True,
        )
