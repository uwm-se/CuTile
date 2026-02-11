"""
Naive Unfused Attention — standard matmul-based SDPA.

Computes attention as separate steps:
  1. S = Q @ K^T * scale   (cuBLAS matmul)
  2. P = softmax(S)         (PyTorch softmax)
  3. O = P @ V              (cuBLAS matmul)

No fusion, no tiling, no memory optimization.
This is the baseline showing why FlashAttention and fused kernels exist.
Memory usage: O(N^2) for the attention matrix S.
"""

import torch
import math


def attn_naive(Q, K, V, causal=False):
    """
    Naive unfused scaled dot-product attention.

    Args:
        Q, K, V: [batch, heads, seq_len, head_dim] in BF16/FP16
        causal: whether to apply causal mask

    Returns:
        O: [batch, heads, seq_len, head_dim]
    """
    head_dim = Q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    # S = Q @ K^T * scale — materializes full N×N attention matrix
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Causal mask
    if causal:
        seq_len = Q.shape[2]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        S.masked_fill_(mask, float('-inf'))

    # Softmax
    P = torch.softmax(S.float(), dim=-1).to(Q.dtype)

    # O = P @ V
    O = torch.matmul(P, V)
    return O
