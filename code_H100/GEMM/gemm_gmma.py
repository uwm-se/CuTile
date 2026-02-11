"""
GMMA GEMM - Hopper WGMMA (Tensor Core) implementation using Triton.
Triton compiles to PTX and uses wgmma instructions on H100 (sm_90).

Based on the official Triton matmul tutorial (03-matrix-multiplication.py)
with all key optimizations:
  1. @triton.autotune — searches 16 tile/warp/stage configurations
  2. L2 cache grouping (GROUP_SIZE_M) — grouped block scheduling for data reuse
  3. Pointer advancement — incremental instead of recomputing each K step
  4. tl.dot(a, b, acc) — fused accumulate form
  5. tl.assume() hints — helps compiler optimize address calculations
  6. Large tile configs — up to 128×256×64 for H100's large shared memory
"""

import torch

HAS_TRITON = False
triton_gemm = None

try:
    import triton
    import triton.language as tl

    # ---- Autotuning configs for CUDA (H100 / Hopper) ----
    # These cover a wide range of tile sizes, warp counts, and pipeline stages.
    # Triton benchmarks each config and picks the fastest for each (M, N, K).
    def get_autotune_configs():
        return [
            # Large tiles — best for big matrices on H100
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                num_stages=3, num_warps=8),
            triton.Config(
                {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                num_stages=3, num_warps=8),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                num_stages=4, num_warps=4),
            triton.Config(
                {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                num_stages=4, num_warps=4),
            # Medium tiles — good balance
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                num_stages=4, num_warps=4),
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                num_stages=4, num_warps=4),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                num_stages=4, num_warps=4),
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                num_stages=4, num_warps=4),
            # Smaller tiles — good for small matrices
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                num_stages=5, num_warps=2),
            triton.Config(
                {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                num_stages=5, num_warps=2),
            # Large K tiles — for high-K workloads
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                num_stages=3, num_warps=8),
            triton.Config(
                {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                num_stages=3, num_warps=8),
            triton.Config(
                {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                num_stages=4, num_warps=4),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                num_stages=4, num_warps=4),
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                num_stages=4, num_warps=4),
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                num_stages=4, num_warps=4),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                num_stages=4, num_warps=4),
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                num_stages=4, num_warps=4),
        ]

    @triton.autotune(
        configs=get_autotune_configs(),
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def _gemm_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters (filled by autotune)
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """
        Triton GEMM kernel with L2 cache grouping.
        On H100 (sm_90), tl.dot compiles to wgmma instructions (Tensor Cores).
        """
        # ---- L2 cache-friendly block scheduling ----
        # Group blocks so nearby blocks share A/B tiles in L2 cache.
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ---- Compiler hints for address optimization ----
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bn > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_cm > 0)
        tl.assume(stride_cn > 0)

        # ---- Initial pointer setup ----
        # Use modulo on M/N offsets to handle boundary (avoids per-element masking)
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # ---- K-loop: accumulate in FP32 ----
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Mask only needed for K dimension (M/N handled by modulo)
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

            # Fused dot-accumulate (more efficient than += tl.dot)
            accumulator = tl.dot(a, b, accumulator)

            # Advance pointers (cheaper than recomputing each iteration)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # ---- Store result as BF16 ----
        c = accumulator.to(tl.bfloat16)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    def triton_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Launch autotuned Triton GEMM kernel (uses WGMMA on Hopper)."""
        assert A.is_contiguous(), "Matrix A must be contiguous"
        assert B.is_contiguous(), "Matrix B must be contiguous"
        M, K = A.shape
        K_check, N = B.shape
        assert K == K_check, f"K mismatch: {K} vs {K_check}"

        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Grid is computed dynamically based on autotuned BLOCK_SIZE
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

        _gemm_kernel[grid](
            A, B, C,
            M=M, N=N, K=K,
            stride_am=A.stride(0), stride_ak=A.stride(1),
            stride_bk=B.stride(0), stride_bn=B.stride(1),
            stride_cm=C.stride(0), stride_cn=C.stride(1),
        )
        return C

    HAS_TRITON = True
except ImportError:
    pass


def gemm_gmma(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    GEMM via GMMA/WGMMA (Hopper Tensor Cores) using Triton.

    Triton's tl.dot compiles to wgmma instructions on sm_90 (H100).
    Uses @triton.autotune to search across 18 tile/warp/stage configs.

    Args:
        A: [M, K] matrix (BF16 or FP16)
        B: [K, N] matrix

    Returns:
        C: [M, N] matrix
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton not available. Install with: pip install triton\n"
            "Triton GEMM uses WGMMA (Tensor Cores) on H100."
        )
    return triton_gemm(A, B)
