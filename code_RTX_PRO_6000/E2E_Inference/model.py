"""
LLaMA-style Transformer Block for End-to-End Inference Benchmarking.

Config: LLaMA-7B (dim=4096, 32 heads, d=128, FFN=11008, SwiGLU)
Supports multiple attention backends: naive, sdpa, flash_attn.
Supports prefill (full sequence) and decode (single token + KV cache).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# LLaMA-7B architecture constants
DIM = 4096
NUM_HEADS = 32
HEAD_DIM = DIM // NUM_HEADS  # 128
FFN_DIM = 11008
NUM_LAYERS = 4  # 4 layers for benchmark (per-layer metrics scale linearly)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA-style)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class LLaMABlock(nn.Module):
    """Single LLaMA transformer block with configurable attention backend."""

    def __init__(self, dim=DIM, num_heads=NUM_HEADS, ffn_dim=FFN_DIM, attn_mode='sdpa'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_mode = attn_mode

        # Attention
        self.attn_norm = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # FFN (SwiGLU)
        self.ffn_norm = RMSNorm(dim)
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, k_cache=None, v_cache=None):
        """
        Args:
            x: [B, S, D] input hidden states
            k_cache: [B, cache_len, num_heads, head_dim] or None
            v_cache: [B, cache_len, num_heads, head_dim] or None
        Returns:
            (output, new_k, new_v)
        """
        B, S, D = x.shape

        # Pre-attention norm
        h = self.attn_norm(x)

        # QKV projections -> [B, S, H, D_head]
        q = self.q_proj(h).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(B, S, self.num_heads, self.head_dim)

        # Append to KV cache
        is_decode = (k_cache is not None)
        if is_decode:
            k_full = torch.cat([k_cache, k], dim=1)
            v_full = torch.cat([v_cache, v], dim=1)
        else:
            k_full = k
            v_full = v

        # Attention dispatch
        if self.attn_mode == 'flash':
            from flash_attn import flash_attn_func
            # flash_attn: [B, S, H, D] layout, handles asymmetric q/kv lengths
            # causal=True always correct: decode maps q to last position
            attn_out = flash_attn_func(q, k_full, v_full, causal=True)

        elif self.attn_mode == 'sdpa':
            # SDPA: [B, H, S, D] layout
            q_t = q.transpose(1, 2)
            k_t = k_full.transpose(1, 2)
            v_t = v_full.transpose(1, 2)
            # is_causal=True only for prefill (q_len == k_len)
            # For decode (q_len=1 < k_len), use is_causal=False
            attn_out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=(not is_decode)
            )
            attn_out = attn_out.transpose(1, 2)

        elif self.attn_mode == 'naive':
            q_t = q.transpose(1, 2)  # [B, H, S, D_head]
            k_t = k_full.transpose(1, 2)
            v_t = v_full.transpose(1, 2)
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
            if not is_decode:  # Prefill: apply causal mask
                mask = torch.triu(
                    torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
                )
                scores.masked_fill_(mask[None, None], float('-inf'))
            attn_out = torch.softmax(scores.float(), dim=-1).to(x.dtype) @ v_t
            attn_out = attn_out.transpose(1, 2)

        attn_out = attn_out.reshape(B, S, D)
        x = x + self.o_proj(attn_out)

        # FFN (SwiGLU)
        h = self.ffn_norm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))

        return x, k_full, v_full


class LLaMAModel(nn.Module):
    """Stack of LLaMA blocks for inference benchmarking."""

    def __init__(self, num_layers=NUM_LAYERS, dim=DIM, num_heads=NUM_HEADS,
                 ffn_dim=FFN_DIM, attn_mode='sdpa'):
        super().__init__()
        self.layers = nn.ModuleList([
            LLaMABlock(dim, num_heads, ffn_dim, attn_mode)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.num_layers = num_layers

    def prefill(self, x):
        """Prefill forward pass. Returns output and KV caches."""
        kv_caches = []
        for layer in self.layers:
            x, k, v = layer(x)
            kv_caches.append((k, v))
        return self.norm(x), kv_caches

    def decode_step(self, x, kv_caches):
        """Single decode step with KV caches. Returns output and updated caches."""
        new_caches = []
        for i, layer in enumerate(self.layers):
            k_cache, v_cache = kv_caches[i]
            x, k, v = layer(x, k_cache, v_cache)
            new_caches.append((k, v))
        return self.norm(x), new_caches
