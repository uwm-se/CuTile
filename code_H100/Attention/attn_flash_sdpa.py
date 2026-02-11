"""
FlashAttention-2 via the official flash_attn package (Tri Dao).

This is the canonical FlashAttention-2 implementation, using optimized
CUDA kernels with tiled IO, online softmax, and no N^2 memory
materialization. Directly calls flash_attn_func for the forward pass.

Package: flash-attn 2.8.3 (https://github.com/Dao-AILab/flash-attention)
"""

import torch
from flash_attn import flash_attn_func

# Native layout for flash_attn_func is [batch, seq_len, heads, head_dim]
NATIVE_LAYOUT = "BNHD"


def attn_flash_sdpa(Q, K, V, causal=False):
    """
    FlashAttention-2 (official Tri Dao implementation).

    Expects Q, K, V in [batch, seq_len, heads, head_dim] layout
    (native flash_attn layout). The benchmark should provide data
    in this layout to avoid transpose overhead in timing.

    Args:
        Q, K, V: [batch, seq_len, heads, head_dim] in BF16/FP16
        causal: whether to apply causal mask

    Returns:
        O: [batch, seq_len, heads, head_dim]
    """
    return flash_attn_func(Q, K, V, causal=causal)
