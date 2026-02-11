"""
CuTile Fused Multi-Head Attention (FMHA) - NVIDIA cuda.tile implementation.
Based on the official NVIDIA CuTile AttentionFMHA.py sample:
  https://github.com/NVIDIA/cutile-python/blob/main/samples/AttentionFMHA.py

NOTE: cuTile requires Blackwell GPUs (compute 10.x / 12.x) and CUDA 13.1+.
cuTile does NOT run on H100 (sm_90). This implementation is provided for when
you have access to B200, B100, or RTX 5090. On H100, the benchmark will skip.
"""

import torch
import math

HAS_CUTILE = False
NATIVE_LAYOUT = "BHND"  # CuTile expects standard [B, H, N, D] layout

try:
    import cuda.tile as ct
    import numpy as np
    from cuda.tile import RoundingMode as RMd

    INV_LOG_2 = 1.0 / math.log(2)
    ConstInt = ct.Constant[int]
    ConstBool = ct.Constant[bool]

    @ct.kernel(occupancy=2)
    def fmha_kernel(Q, K, V, Out,
                    qk_scale: float,
                    input_pos: int,
                    TILE_D: ConstInt,
                    H: ConstInt,
                    TILE_M: ConstInt,
                    TILE_N: ConstInt,
                    QUERY_GROUP_SIZE: ConstInt,
                    CAUSAL: ConstBool,
                    EVEN_K: ConstBool):
        """
        CuTile fused attention kernel with online softmax.
        Implements FlashAttention-2 algorithm using ct.mma for Tensor Core acceleration.
        """
        bid_x = ct.bid(0)
        bid_y = ct.bid(1)
        batch_idx = bid_y // H
        head_idx = bid_y % H
        off_kv_h = head_idx // QUERY_GROUP_SIZE

        qk_scale = qk_scale * INV_LOG_2

        # M-dimension offsets for query tile
        offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)
        offs_m += input_pos
        offs_m = offs_m[:, None]

        # N-dimension offsets for key/value tile
        offs_n_tile = ct.arange(TILE_N, dtype=np.int32)
        offs_n_tile = offs_n_tile[None, :]

        # Online softmax accumulators
        m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
        l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
        acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

        # Load query tile
        q = ct.load(
            Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D),
            latency=2,
        ).reshape((TILE_M, TILE_D))

        m_end = input_pos + (bid_x + 1) * TILE_M
        k_seqlen = K.shape[2]
        if CAUSAL:
            mask_start = (input_pos + bid_x * TILE_M) // TILE_N
            mask_start = min(mask_start, k_seqlen // TILE_N)
            Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
        else:
            Tc = ct.cdiv(k_seqlen, TILE_N)
            mask_start = k_seqlen // TILE_N

        # Loop over K/V blocks
        for j in range(0, Tc):
            # Load K block (transposed)
            k = ct.load(
                K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
                order=(0, 1, 3, 2),
                latency=2,
            )
            k = k.reshape((TILE_D, TILE_N))
            qk = ct.full((TILE_M, TILE_N), 0., dtype=np.float32)
            qk = ct.mma(q, k, qk)

            # Causal masking
            if (CAUSAL or not EVEN_K) and j >= mask_start:
                offs_n = j * TILE_N + offs_n_tile
                mask = ct.full((TILE_M, TILE_N), True, dtype=np.bool)
                if not EVEN_K:
                    mask = mask & (offs_n < k_seqlen)
                if CAUSAL:
                    mask = mask & (offs_m >= offs_n)
                mask = ct.where(mask, 0.0, -np.inf)
                qk += mask

            # Online softmax
            m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
            qk = qk * qk_scale - m_ij
            p = ct.exp2(qk, flush_to_zero=True)
            l_ij = ct.sum(p, axis=-1, keepdims=True)
            alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha

            # Load V block and accumulate
            v = ct.load(
                V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D),
                latency=4,
            ).reshape((TILE_N, TILE_D))
            p = p.astype(Q.dtype)
            acc = ct.mma(p, v, acc)
            m_i = m_ij

        # Final normalization and store
        acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
        acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
        ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)

    def _attn_cutile(Q, K, V, causal=False):
        """Launch CuTile FMHA kernel."""
        Batch, Heads, SeqLen_Q, D_k = Q.shape
        _, KV_Heads, SeqLen_KV, D_v = V.shape

        qk_scale = 1.0 / math.sqrt(D_k)
        tile_m, tile_n = 128, 128
        query_group_size = Heads // KV_Heads
        even_k = (SeqLen_KV % tile_n) == 0

        Out = torch.empty((Batch, Heads, SeqLen_Q, D_v), dtype=Q.dtype, device=Q.device)

        grid_x = math.ceil(SeqLen_Q / tile_m)
        grid_y = Batch * Heads
        grid = (grid_x, grid_y, 1)

        ct.launch(torch.cuda.current_stream(), grid, fmha_kernel, (
            Q, K, V, Out,
            qk_scale,
            0,     # input_pos
            D_k,
            Heads,
            tile_m,
            tile_n,
            query_group_size,
            causal,
            even_k
        ))
        return Out

    HAS_CUTILE = True

except ImportError:
    _attn_cutile = None


def attn_cutile(Q, K, V, causal=False):
    """
    Fused Multi-Head Attention via NVIDIA CuTile (cuda.tile).

    Expects input layout: [B, H, N, D] (standard PyTorch layout).
    Requires: Blackwell GPU (B200, B100, RTX 5090), CUDA 13.1+
    Install:  pip install cuda-tile cupy-cuda13x
    """
    if not HAS_CUTILE:
        raise RuntimeError(
            "cuTile not available. Install: pip install cuda-tile cupy-cuda13x\n"
            "cuTile requires Blackwell GPU (compute 10.x / 12.x), CUDA 13.1+"
        )
    return _attn_cutile(Q, K, V, causal=causal)
