"""CPU-only paged attention.

This fork removed the original CUDA path entirely. The implementation
below uses `torch.nn.functional.scaled_dot_product_attention` for the
math and an `index_copy_`-based paged store for KV writes — no
flash-attn, no triton, no `triton.jit`, no `flash_attn_varlen_func`.
"""
import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.utils.context import get_context


def store_kvcache(
    key: torch.Tensor,        # [total_tokens, num_kv_heads, head_dim]
    value: torch.Tensor,
    k_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Paged KV-cache store via index_copy_ on a flat view."""
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


def _prefill(
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


def _decode(
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
        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            return _prefill(
                q, k, v, k_cache, v_cache, context,
                self.scale, self.num_heads, self.num_kv_heads, self.head_dim,
            )
        return _decode(
            q, k_cache, v_cache, context,
            self.scale, self.num_heads, self.num_kv_heads, self.head_dim,
        )
