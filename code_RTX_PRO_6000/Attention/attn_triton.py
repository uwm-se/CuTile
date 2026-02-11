"""
Triton Fused Attention - Flash Attention v2 algorithm in Triton.

Adapted from the official Triton tutorial (06-fused-attention.py).
Forward pass only, with autotuning over block sizes and warp counts.
On H100 (sm_90), tl.dot compiles to WGMMA instructions.
"""

import torch

HAS_TRITON = False
triton_attention_fn = None

try:
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        ],
        key=['N_CTX', 'HEAD_DIM'],
    )
    @triton.jit
    def _fwd_kernel(
        Q, K, V, Out,
        sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
        o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        # Load Q block - stays in SRAM
        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        # Online softmax state
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        qk_scale = sm_scale * 1.44269504  # sm_scale / ln(2)

        if CAUSAL:
            hi = min((start_m + 1) * BLOCK_M, N_CTX)
        else:
            hi = N_CTX

        offs_n = tl.arange(0, BLOCK_N)

        for start_n in range(0, hi, BLOCK_N):
            k_ptrs = K + k_offset + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)

            qk = tl.dot(q, tl.trans(k))

            if CAUSAL:
                causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = tl.where(causal_mask, qk * qk_scale, float("-inf"))
            else:
                qk = qk * qk_scale

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)

            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v_ptrs = V + v_offset + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
            p = p.to(v.dtype)
            acc = tl.dot(p, v, acc)

            m_i = m_ij

        acc = acc / l_i[:, None]

        o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)

    def triton_attention(Q, K, V, causal=False):
        B, H, N_CTX, HEAD_DIM = Q.shape
        assert K.shape == V.shape == Q.shape
        assert HEAD_DIM in {16, 32, 64, 128, 256}

        Out = torch.empty_like(Q)
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)

        grid = lambda META: (
            triton.cdiv(N_CTX, META['BLOCK_M']),
            B * H,
        )

        _fwd_kernel[grid](
            Q, K, V, Out,
            sm_scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            Z=B, H=H, N_CTX=N_CTX,
            HEAD_DIM=HEAD_DIM,
            CAUSAL=causal,
        )
        return Out

    triton_attention_fn = triton_attention
    HAS_TRITON = True
except ImportError:
    pass


def attn_triton(Q, K, V, causal=False):
    """Triton Fused Attention (Flash Attention v2 algorithm)."""
    if not HAS_TRITON:
        raise RuntimeError("Triton not available.")
    return triton_attention_fn(Q, K, V, causal=causal)
