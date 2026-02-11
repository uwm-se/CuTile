"""
cuBLAS GEMM - Uses PyTorch's matmul (cuBLAS backend, Tensor Cores on H100)
Baseline implementation for paper benchmarks.
"""

import torch


def gemm_cublas(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    GEMM via cuBLAS (torch.matmul).
    Uses Tensor Cores automatically on H100 with BF16/FP16.

    Args:
        A: [M, K] matrix
        B: [K, N] matrix

    Returns:
        C: [M, N] matrix
    """
    return torch.matmul(A, B)
