"""
CuTile GEMM - NVIDIA cuTile Python (cuda.tile) implementation.
Based on the official NVIDIA CuTile MatMul.py sample:
  https://github.com/NVIDIA/cutile-python/blob/main/samples/MatMul.py

NOTE: cuTile requires Blackwell GPUs (compute 10.x / 12.x) and CUDA 13.1+.
cuTile does NOT run on H100 (sm_90). This implementation is provided for when
you have access to B200, B100, or RTX 5090. On H100, the benchmark will skip.
"""

import torch
from math import ceil

HAS_CUTILE = False
gemm_cutile_wrapper = None

try:
    import cuda.tile as ct
    import numpy as np

    ConstInt = ct.Constant[int]

    def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
        """L2 cache-friendly 2D block swizzle from 1D block ID."""
        bid = ct.bid(0)
        num_bid_m = ct.cdiv(M, tm)
        num_bid_n = ct.cdiv(N, tn)
        num_bid_in_group = GROUP_SIZE_M * num_bid_n
        group_id = bid // num_bid_in_group
        first_bid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
        bid_m = first_bid_m + (bid % group_size_m)
        bid_n = (bid % num_bid_in_group) // group_size_m
        return bid_m, bid_n

    @ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=2))
    def matmul_kernel(A, B, C,
                      tm: ConstInt,
                      tn: ConstInt,
                      tk: ConstInt):
        """
        CuTile GEMM kernel: C = A @ B
        Uses ct.mma for Tensor Core acceleration and L2-friendly swizzling.
        """
        GROUP_SIZE_M = 8
        M = A.shape[0]
        N = B.shape[1]
        bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

        num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

        # FP32 accumulator for precision
        accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
        zero_pad = ct.PaddingMode.ZERO

        # Convert fp32 to tf32 for tensor cores; keep bf16/fp16 as-is
        dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

        for k in range(num_tiles_k):
            a = ct.load(A, index=(bidx, k), shape=(tm, tk),
                        padding_mode=zero_pad, latency=4).astype(dtype)
            b = ct.load(B, index=(k, bidy), shape=(tk, tn),
                        padding_mode=zero_pad, latency=4).astype(dtype)
            accumulator = ct.mma(a, b, accumulator)

        accumulator = ct.astype(accumulator, C.dtype)
        ct.store(C, index=(bidx, bidy), tile=accumulator)

    def _gemm_cutile_wrapper(A_torch: torch.Tensor, B_torch: torch.Tensor) -> torch.Tensor:
        """Launch CuTile GEMM kernel and return result."""
        if A_torch.shape[1] != B_torch.shape[0]:
            raise ValueError(f"Incompatible K dims: A={A_torch.shape[1]}, B={B_torch.shape[0]}")
        if not A_torch.is_cuda or not B_torch.is_cuda:
            raise ValueError("Inputs must be on CUDA")

        # Tile sizes optimized for BF16/FP16 on Blackwell Tensor Cores
        if A_torch.dtype.itemsize == 2:  # fp16 or bf16
            tm, tn, tk = 128, 256, 64
        else:  # fp32
            tm, tn, tk = 32, 32, 32

        m, k = A_torch.shape
        _, n = B_torch.shape

        grid_x = ceil(m / tm)
        grid_y = ceil(n / tn)
        grid = (grid_x * grid_y, 1, 1)

        C = torch.empty((m, n), device=A_torch.device, dtype=A_torch.dtype)

        ct.launch(torch.cuda.current_stream(), grid, matmul_kernel,
                  (A_torch, B_torch, C, tm, tn, tk))

        return C

    gemm_cutile_wrapper = _gemm_cutile_wrapper
    HAS_CUTILE = True

except ImportError:
    pass


def gemm_cutile(A, B):
    """
    GEMM via NVIDIA CuTile (cuda.tile).

    Requires: Blackwell GPU (B200, B100, RTX 5090), CUDA 13.1+
    Install:  pip install cuda-tile cupy-cuda13x
    """
    if not HAS_CUTILE:
        raise RuntimeError(
            "cuTile not available. Install: pip install cuda-tile cupy-cuda13x\n"
            "cuTile requires Blackwell GPU (compute 10.x / 12.x), CUDA 13.1+"
        )
    return gemm_cutile_wrapper(A, B)
