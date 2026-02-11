"""
cuDNN Attention via PyTorch SDPA (memory-efficient backend).

On Hopper, the mem_efficient backend dispatches to cuDNN's fused attention,
which uses Tensor Cores. This represents a vendor-optimized but different
algorithm path from FlashAttention-2.
"""

import torch
import torch.nn.functional as F


def attn_cudnn_sdpa(Q, K, V, causal=False):
    """
    cuDNN/Memory-Efficient Attention via PyTorch SDPA.

    Args:
        Q, K, V: [batch, heads, seq_len, head_dim] in BF16/FP16
        causal: whether to apply causal mask

    Returns:
        O: [batch, heads, seq_len, head_dim]
    """
    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
    ):
        return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
