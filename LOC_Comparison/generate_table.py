#!/usr/bin/env python3
"""
Lines of Code (LOC) Comparison — Table 6

Counts kernel/implementation LOC for each abstraction level across GEMM and
Attention operations. Follows the standard LOC convention:
  - Count only non-blank, non-comment lines
  - Kernel code + minimal launcher (what the programmer writes)
  - Exclude test harness, benchmarking, argparse, validation boilerplate

For CuTile: uses official NVIDIA samples from github.com/NVIDIA/cutile-python.
For WMMA/Raw SIMT: counts from our .cu source files (or notes they were lost).
"""

import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "LOC_Comparison"

# ============================================================================
# LOC Data — manually counted from source files
#
# Methodology:
#   "Kernel LOC" = lines in the kernel function itself (the GPU code)
#   "Total LOC"  = kernel + launcher/wrapper + necessary helpers
#   Both exclude: blank lines, comments, docstrings, imports, test code
# ============================================================================

loc_data = {
    "GEMM": {
        "cuBLAS (torch.matmul)": {
            "kernel_loc": 1,
            "total_loc": 1,
            "source": "gemm_cublas.py",
            "notes": "Single call: torch.matmul(A, B). Zero kernel code — pure library call.",
            "abstraction": "Library API",
        },
        "Triton (WGMMA)": {
            "kernel_loc": 53,
            "total_loc": 76,
            "source": "gemm_gmma.py",
            "notes": (
                "Triton JIT kernel with L2 cache grouping, pointer advancement, "
                "fused dot-accumulate. 18 autotune configs (counted as data, not logic). "
                "Kernel: lines 94-162 (53 lines). Launcher: lines 164-186 (23 lines)."
            ),
            "abstraction": "DSL (Triton)",
        },
        "CuTile": {
            "kernel_loc": 22,
            "total_loc": 45,
            "source": "github.com/NVIDIA/cutile-python/samples/MatMul.py",
            "notes": (
                "Official CuTile sample. matmul_kernel: 22 lines of kernel code "
                "(ct.load, ct.mma, ct.store loop). cutile_matmul wrapper: ~23 lines "
                "(tile selection, grid calc, launch). Excludes persistent variant, "
                "swizzle helper, tests."
            ),
            "abstraction": "DSL (CuTile)",
        },
        "WMMA (Tensor Core CUDA)": {
            "kernel_loc": 123,
            "total_loc": 183,
            "source": "wmma_kernel.cu",
            "notes": (
                "Hand-written CUDA with nv_wmma API. Triple-buffered shared memory, "
                "cp.async pipelining, __launch_bounds__, shared memory padding, "
                "inline PTX for cp.async. Kernel 123 lines, defines+helpers 44 lines, "
                "host wrapper 16 lines. Total non-blank non-comment: 183 lines."
            ),
            "abstraction": "CUDA C++ (WMMA API)",
        },
        "Raw SIMT (CUDA)": {
            "kernel_loc": 32,
            "total_loc": 58,
            "source": "raw_simt_kernel.cu",
            "notes": (
                "Hand-written CUDA without Tensor Cores. Simple shared memory tiling, "
                "scalar FP32 FMA. Kernel 32 lines, host wrapper 16 lines, "
                "includes/pybind 10 lines. Total non-blank non-comment: 58 lines."
            ),
            "abstraction": "CUDA C++ (scalar FMA)",
        },
    },
    "Attention (Fused SDPA)": {
        "FlashAttention-2": {
            "kernel_loc": 1,
            "total_loc": 1,
            "source": "attn_flash_sdpa.py (wraps flash_attn 2.8.3)",
            "notes": (
                "Single call: flash_attn_func(Q, K, V, causal=True). "
                "Kernel is ~2000+ lines of optimized CUDA inside the flash_attn package."
            ),
            "abstraction": "Library API",
        },
        "PyTorch SDPA": {
            "kernel_loc": 1,
            "total_loc": 4,
            "source": "attn_cudnn_sdpa.py",
            "notes": (
                "F.scaled_dot_product_attention with backend selection. "
                "1 line kernel call inside a 4-line context manager."
            ),
            "abstraction": "Library API",
        },
        "Triton (Flash Attn v2)": {
            "kernel_loc": 62,
            "total_loc": 87,
            "source": "attn_triton.py",
            "notes": (
                "Triton JIT kernel implementing Flash Attention v2 with online softmax, "
                "tiled Q/K/V, exp2-based softmax, causal masking. "
                "Kernel: lines 30-104 (62 lines). Launcher: lines 106-130 (25 lines). "
                "6 autotune configs."
            ),
            "abstraction": "DSL (Triton)",
        },
        "CuTile (FMHA)": {
            "kernel_loc": 60,
            "total_loc": 95,
            "source": "github.com/NVIDIA/cutile-python/samples/AttentionFMHA.py",
            "notes": (
                "Official CuTile FMHA sample. fmha_kernel: ~60 lines (online softmax, "
                "ct.mma for QK and PV, causal masking, exp2). cutile_fmha wrapper: "
                "~35 lines (validation, grid calc, launch). Excludes autotuner variant, "
                "torch reference, tests."
            ),
            "abstraction": "DSL (CuTile)",
        },
        "Naive (Unfused)": {
            "kernel_loc": 10,
            "total_loc": 10,
            "source": "attn_naive.py",
            "notes": (
                "3 PyTorch ops: matmul + softmax + matmul, plus causal mask. "
                "Simple but O(N^2) memory."
            ),
            "abstraction": "PyTorch eager",
        },
    },
}


def print_table():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print("TABLE 6: LINES OF CODE COMPARISON")
    print("  Kernel LOC = GPU kernel code only")
    print("  Total LOC  = kernel + launcher/wrapper (excludes tests, benchmarks, imports)")
    print("=" * 95)

    for operation, impls in loc_data.items():
        print(f"\n  {operation}")
        print(f"  {'Implementation':<28} {'Abstraction':<22} {'Kernel LOC':>10} {'Total LOC':>10}  Source")
        print("  " + "-" * 92)

        for name, info in impls.items():
            print(f"  {name:<28} {info['abstraction']:<22} {info['kernel_loc']:>10} {info['total_loc']:>10}  {info['source']}")

    # Ratios
    print("\n" + "=" * 95)
    print("CODE REDUCTION RATIOS (vs hand-written CUDA)")
    print("=" * 95)

    # GEMM ratios
    wmma_total = loc_data["GEMM"]["WMMA (Tensor Core CUDA)"]["total_loc"]
    print(f"\n  GEMM (baseline: WMMA CUDA = {wmma_total} LOC)")
    for name, info in loc_data["GEMM"].items():
        ratio = wmma_total / info["total_loc"] if info["total_loc"] > 0 else float('inf')
        print(f"    {name:<28} {info['total_loc']:>5} LOC  ({ratio:>5.1f}x less code)")

    # Attention ratios — use Triton as reference (closest to manual)
    triton_total = loc_data["Attention (Fused SDPA)"]["Triton (Flash Attn v2)"]["total_loc"]
    print(f"\n  Attention (baseline: Triton = {triton_total} LOC)")
    for name, info in loc_data["Attention (Fused SDPA)"].items():
        ratio = triton_total / info["total_loc"] if info["total_loc"] > 0 else float('inf')
        print(f"    {name:<28} {info['total_loc']:>5} LOC  ({ratio:>5.1f}x less code)")

    # Save
    output = {
        "title": "Lines of Code Comparison (Table 6)",
        "methodology": (
            "Kernel LOC: non-blank non-comment lines in the GPU kernel function. "
            "Total LOC: kernel + launcher/wrapper code the programmer writes. "
            "Excludes: blank lines, comments, docstrings, imports, test harness, "
            "benchmark infrastructure, argparse."
        ),
        "loc_data": loc_data,
        "timestamp": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / "loc_comparison.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR}/loc_comparison.json")


if __name__ == "__main__":
    print_table()
