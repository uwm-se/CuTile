#!/usr/bin/env python3
"""
Additional GEMM Benchmarks for Research Paper
1. Rectangular GEMM shapes (LLaMA-7B FFN projections) — BF16
2. FP16 square GEMM benchmarks (same sizes as BF16)

Results saved separately from the main BF16 square GEMM results.
"""

import os
import sys
import json
import time
import platform
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

# Rectangular shapes: LLaMA-7B FFN projections (dim=4096, ffn=11008)
# Format: (M, K, N, label)
RECT_SHAPES = [
    (2048, 4096, 11008, "FFN_up_seq2048"),    # Up-projection, seq=2048
    (2048, 11008, 4096, "FFN_down_seq2048"),   # Down-projection, seq=2048
    (4096, 4096, 11008, "FFN_up_seq4096"),     # Up-projection, seq=4096
    (4096, 11008, 4096, "FFN_down_seq4096"),   # Down-projection, seq=4096
]

# Square sizes for FP16 benchmark
FP16_SQUARE_SIZES = [4096, 8192, 12288, 16384]

WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50
RAW_SIMT_WARMUP = 2
RAW_SIMT_ITERS = 3

RESULTS_DIR = Path(__file__).parent.parent.parent / "results_H100" / "GEMM"


def get_device_config() -> dict:
    """Collect hardware and software details."""
    props = torch.cuda.get_device_properties(0)
    hardware = {
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": f"sm_{props.major}{props.minor}",
        "num_sms": props.multi_processor_count,
        "global_memory_gb": round(props.total_memory / (1024**3), 2),
    }
    software = {
        "cuda_version": torch.version.cuda or "N/A",
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    try:
        import triton
        software["triton_version"] = triton.__version__
    except ImportError:
        software["triton_version"] = "N/A"
    # CuTile not available on H100 (requires Blackwell)
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            software["nvidia_driver"] = r.stdout.strip()
    except Exception:
        software["nvidia_driver"] = "N/A"
    return {"hardware": hardware, "software": software}


def calculate_tflops(M, N, K, time_ms):
    return (2 * M * N * K / (time_ms * 1e-3)) / 1e12


def benchmark_impl(name, gemm_fn, A, B, M, N, K):
    is_raw_simt = "Raw SIMT" in name
    warmup = RAW_SIMT_WARMUP if is_raw_simt else WARMUP_ITERATIONS
    iters = RAW_SIMT_ITERS if is_raw_simt else BENCHMARK_ITERATIONS

    try:
        for _ in range(warmup):
            gemm_fn(A, B)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            gemm_fn(A, B)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / iters
        tflops = calculate_tflops(M, N, K, elapsed_ms)
        return {
            "time_ms": round(elapsed_ms, 4),
            "tflops": round(tflops, 2),
            "status": "ok",
            "warmup_iterations": warmup,
            "benchmark_iterations": iters,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


def load_implementations():
    """Load all available GEMM implementations (no CuTile on H100)."""
    from gemm_cublas import gemm_cublas
    from gemm_gmma import gemm_gmma, HAS_TRITON

    HAS_RAW_SIMT = False
    raw_simt_fn = None
    try:
        import raw_simt_kernel
        raw_simt_fn = raw_simt_kernel.gemm
        HAS_RAW_SIMT = True
    except (ImportError, ModuleNotFoundError):
        pass

    HAS_WMMA = False
    wmma_fn = None
    try:
        import wmma_kernel
        wmma_fn = wmma_kernel.gemm
        HAS_WMMA = True
    except (ImportError, ModuleNotFoundError):
        pass

    return [
        ("cuBLAS",     gemm_cublas,  True),
        ("Triton",     gemm_gmma,    HAS_TRITON),
        ("WMMA",       wmma_fn,      HAS_WMMA),
        ("Raw SIMT",   raw_simt_fn,  HAS_RAW_SIMT),
    ]


# ============================================================================
# Part 1: Rectangular GEMM (BF16)
# ============================================================================

def run_rectangular_gemm():
    """Benchmark rectangular GEMM shapes (LLaMA-7B FFN projections)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = get_device_config()
    implementations = load_implementations()

    print("\n" + "=" * 80, flush=True)
    print("RECTANGULAR GEMM BENCHMARK — BF16 (LLaMA-7B FFN shapes)", flush=True)
    print("=" * 80, flush=True)
    print(f"Device: {config['hardware']['gpu_name']}", flush=True)

    results = {
        "title": f"Rectangular GEMM Benchmark — BF16 on {config['hardware']['gpu_name']}",
        "device_config": config,
        "benchmark_config": {
            "dtype": "torch.bfloat16",
            "matrix_type": "rectangular (LLaMA-7B FFN: dim=4096, ffn_dim=11008)",
            "shapes": [{"M": s[0], "K": s[1], "N": s[2], "label": s[3]} for s in RECT_SHAPES],
            "warmup_iterations": WARMUP_ITERATIONS,
            "benchmark_iterations": BENCHMARK_ITERATIONS,
        },
        "results": {},
        "timestamp": datetime.now().isoformat(),
    }

    for M, K, N, label in RECT_SHAPES:
        print(f"\n  {label}: {M}x{K} × {K}x{N}  (FLOPs: {2*M*N*K/1e9:.1f} GFLOPs)", flush=True)
        print("  " + "-" * 60, flush=True)

        torch.manual_seed(42)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

        for impl_name, gemm_fn, available in implementations:
            if gemm_fn is None or not available:
                results["results"].setdefault(impl_name, {})[label] = {"status": "unavailable"}
                print(f"    {impl_name:<18} N/A", flush=True)
                continue

            print(f"    {impl_name:<18} running...", end="", flush=True)
            r = benchmark_impl(impl_name, gemm_fn, A, B, M, N, K)
            results["results"].setdefault(impl_name, {})[label] = r

            if r["status"] == "ok":
                print(f"\r    {impl_name:<18} {r['time_ms']:>8.4f} ms  {r['tflops']:>7.2f} TFLOP/s", flush=True)
            else:
                print(f"\r    {impl_name:<18} ERROR: {r.get('error', '')[:50]}", flush=True)

        del A, B
        torch.cuda.empty_cache()
        time.sleep(1)

    out_path = RESULTS_DIR / "gemm_rectangular_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}", flush=True)

    # Print paper table
    print("\n" + "=" * 90)
    print("TABLE: Rectangular GEMM (BF16, LLaMA-7B FFN shapes)")
    print("=" * 90)
    impl_order = ["cuBLAS", "Triton", "WMMA", "Raw SIMT"]
    header = f"  {'Implementation':<18}"
    for _, _, _, label in RECT_SHAPES:
        short = label.replace("FFN_", "").replace("_", " ")
        header += f"  {short:>16}"
    print(header)
    print("  " + "-" * (18 + 18 * len(RECT_SHAPES)))
    for impl_name in impl_order:
        row = f"  {impl_name:<18}"
        for _, _, _, label in RECT_SHAPES:
            r = results["results"].get(impl_name, {}).get(label, {})
            if r.get("status") == "ok":
                row += f"  {r['tflops']:>12.1f} TF/s"
            else:
                row += f"  {'---':>16}"
        print(row)

    return results


# ============================================================================
# Part 2: FP16 Square GEMM
# ============================================================================

def run_fp16_gemm():
    """Benchmark square GEMM shapes with FP16 dtype."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = get_device_config()

    # For FP16, cuBLAS handles dtype correctly via torch.matmul.
    # Triton kernel may work via implicit cast.
    from gemm_cublas import gemm_cublas

    # Try Triton too (it might work via implicit cast)
    HAS_TRITON = False
    triton_fn = None
    try:
        from gemm_gmma import gemm_gmma, HAS_TRITON as _HT
        if _HT:
            triton_fn = gemm_gmma
            HAS_TRITON = True
    except ImportError:
        pass

    implementations = [
        ("cuBLAS",  gemm_cublas,  True),
        ("Triton",  triton_fn,    HAS_TRITON),
    ]

    print("\n" + "=" * 80, flush=True)
    print("FP16 SQUARE GEMM BENCHMARK", flush=True)
    print("=" * 80, flush=True)
    print(f"Device: {config['hardware']['gpu_name']}", flush=True)

    results = {
        "title": f"GEMM Benchmark — FP16 Square Matrices on {config['hardware']['gpu_name']}",
        "device_config": config,
        "benchmark_config": {
            "dtype": "torch.float16",
            "matrix_type": "square (M=N=K)",
            "matrix_sizes": FP16_SQUARE_SIZES,
            "warmup_iterations": WARMUP_ITERATIONS,
            "benchmark_iterations": BENCHMARK_ITERATIONS,
        },
        "results": {},
        "timestamp": datetime.now().isoformat(),
    }

    for size in FP16_SQUARE_SIZES:
        M = N = K = size
        print(f"\n  Matrix: {M}x{K} x {K}x{N} (FP16)  (FLOPs: {2*M*N*K/1e9:.1f} GFLOPs)", flush=True)
        print("  " + "-" * 56, flush=True)

        torch.manual_seed(42)
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)

        for impl_name, gemm_fn, available in implementations:
            if gemm_fn is None or not available:
                results["results"].setdefault(impl_name, {})[str(size)] = {"status": "unavailable"}
                print(f"    {impl_name:<18} N/A", flush=True)
                continue

            print(f"    {impl_name:<18} running...", end="", flush=True)
            r = benchmark_impl(impl_name, gemm_fn, A, B, M, N, K)
            results["results"].setdefault(impl_name, {})[str(size)] = r

            if r["status"] == "ok":
                print(f"\r    {impl_name:<18} {r['time_ms']:>8.4f} ms  {r['tflops']:>7.2f} TFLOP/s", flush=True)
            else:
                print(f"\r    {impl_name:<18} ERROR: {r.get('error', '')[:50]}", flush=True)

        del A, B
        torch.cuda.empty_cache()
        time.sleep(1)

    out_path = RESULTS_DIR / "gemm_fp16_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}", flush=True)

    # Print paper table
    print("\n" + "=" * 90)
    print("TABLE: FP16 Square GEMM")
    print("=" * 90)
    header = f"  {'Implementation':<18}"
    for s in FP16_SQUARE_SIZES:
        header += f" {'%dx%d' % (s, s):>12}"
    print(header)
    print("  " + "-" * (18 + 14 * len(FP16_SQUARE_SIZES)))
    for impl_name, _, _ in implementations:
        row = f"  {impl_name:<18}"
        for size in FP16_SQUARE_SIZES:
            r = results["results"].get(impl_name, {}).get(str(size), {})
            if r.get("status") == "ok":
                row += f" {r['tflops']:>8.1f} TF/s"
            else:
                row += f" {'---':>12}"
        print(row)

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ADDITIONAL GEMM BENCHMARKS")
    print("=" * 80)

    print("\n>>> Part 1: Rectangular GEMM (BF16, LLaMA-7B FFN shapes)")
    run_rectangular_gemm()

    print("\n\n>>> Part 2: FP16 Square GEMM")
    run_fp16_gemm()

    print("\n\nAll additional GEMM benchmarks complete!")
