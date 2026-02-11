#!/usr/bin/env python3
"""
GEMM Benchmark Runner for Research Paper
Runs cuBLAS, Triton, WMMA, and Raw SIMT.
Auto-detects GPU (tested on NVIDIA H100 NVL, Hopper architecture).
CuTile is NOT available on H100 (requires Blackwell sm_100/sm_120).
Saves comprehensive results with hardware config, software versions,
and implementation descriptions to results_H100/GEMM/.
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

# Matrix sizes from paper (Table 1) — square GEMMs
MATRIX_SIZES = [4096, 8192, 12288, 16384]

# Benchmark parameters
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50
RAW_SIMT_WARMUP = 2
RAW_SIMT_ITERS = 3

RESULTS_DIR = Path(__file__).parent.parent.parent / "results_H100" / "GEMM"


# ============================================================================
# Hardware & Software Configuration
# ============================================================================

def get_device_config() -> dict:
    """Collect comprehensive hardware and software details for reproducibility."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    props = torch.cuda.get_device_properties(0)

    # --- Architecture detection ---
    def detect_architecture(major, minor):
        if major >= 12:
            return "Blackwell"
        elif major >= 10:
            return "Blackwell"  # sm_100 = Blackwell consumer
        elif major == 9:
            return "Hopper"
        elif major == 8:
            return "Ampere"
        elif major == 7:
            return "Volta/Turing"
        return "Unknown"

    gpu_name = torch.cuda.get_device_name(0)
    arch = detect_architecture(props.major, props.minor)

    # --- Hardware ---
    hardware = {
        "gpu_name": gpu_name,
        "gpu_architecture": arch,
        "compute_capability": f"sm_{props.major}{props.minor}",
        "num_sms": props.multi_processor_count,
        "max_threads_per_sm": props.max_threads_per_multi_processor,
        "shared_memory_per_sm_kb": round(props.shared_memory_per_multiprocessor / 1024, 1),
        "global_memory_gb": round(props.total_memory / (1024**3), 2),
    }

    # GPU clock speed
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.sm,clocks.max.mem",
             "--format=csv,noheader"],
            capture_output=True, text=True)
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            hardware["max_sm_clock_mhz"] = parts[0].strip().replace(" MHz", "")
            hardware["max_mem_clock_mhz"] = parts[1].strip().replace(" MHz", "")
    except Exception:
        pass

    # --- Software ---
    software = {
        "cuda_version": torch.version.cuda or "N/A",
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
    }

    # Driver version
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True)
        if r.returncode == 0:
            software["nvidia_driver"] = r.stdout.strip()
    except Exception:
        software["nvidia_driver"] = "N/A"

    # Triton version
    try:
        import triton
        software["triton_version"] = triton.__version__
    except ImportError:
        software["triton_version"] = "N/A"

    # --- Detect cuda-tile version ---
    try:
        import cuda.tile as ct
        software["cuda_tile_version"] = ct.__version__
    except (ImportError, AttributeError):
        software["cuda_tile_version"] = "N/A"

    # --- Detect flash_attn version ---
    try:
        import flash_attn
        software["flash_attn_version"] = flash_attn.__version__
    except ImportError:
        software["flash_attn_version"] = "N/A"

    return {"hardware": hardware, "software": software}


# ============================================================================
# Implementation Descriptions (for paper & JSON metadata)
# ============================================================================

IMPL_DESCRIPTIONS = {
    "cuBLAS": {
        "full_name": "cuBLAS (torch.matmul)",
        "description": (
            "NVIDIA's vendor-optimized BLAS library via torch.matmul. cuBLAS "
            "auto-selects the best Tensor Core instruction set for the target GPU. "
            "Uses TMA, warp specialization, persistent kernels, and deep software "
            "pipelines. Represents the practical upper bound for library-based GEMM."
        ),
        "tensor_core_api": "Architecture-native Tensor Core instructions (auto-selected)",
        "memory_subsystem": "TMA + async pipeline + L2 swizzling",
        "optimization_level": "Vendor-optimized (closed-source)",
        "abstraction_level": "Library call",
    },
    "Triton": {
        "full_name": "Triton (autotuned)",
        "description": (
            "OpenAI Triton compiler with @triton.autotune over 18 tile/warp/stage "
            "configurations. Triton's tl.dot compiles to architecture-native Tensor "
            "Core instructions. Includes L2 cache grouping (GROUP_SIZE_M=8) for "
            "improved data reuse and incremental pointer advancement. Based on the "
            "official Triton matmul tutorial (03-matrix-multiplication.py). "
            "Does NOT use TMA or warp specialization."
        ),
        "tensor_core_api": "Architecture-native via tl.dot (Triton compiler lowers to PTX)",
        "memory_subsystem": "Standard global loads + L2 cache grouping",
        "optimization_level": "Autotuned (18 configs), no TMA/warp specialization",
        "abstraction_level": "High-level GPU compiler (Python DSL)",
    },
    "WMMA": {
        "full_name": "WMMA (hand-written CUDA, legacy Tensor Core API)",
        "description": (
            "Hand-written CUDA kernel using the WMMA (Warp Matrix Multiply-Accumulate) "
            "API — a Volta-era (sm_70+) Tensor Core interface. Uses mma.sync instructions "
            "(warp-level, 32 threads). Optimizations: shared memory tiling (128x128x32), "
            "double-buffered cp.async pipeline (NUM_STAGES=2), "
            "2x2 register tiling per warp, shared memory padding for bank conflict avoidance, "
            "and __launch_bounds__ for occupancy control."
        ),
        "tensor_core_api": "WMMA / mma.sync (legacy, 32-thread warp-level)",
        "memory_subsystem": "cp.async + double-buffered shared memory + padding",
        "optimization_level": "Hand-optimized CUDA with standard techniques",
        "abstraction_level": "Low-level CUDA C++ (hand-written kernel)",
    },
    "Raw SIMT": {
        "full_name": "Raw SIMT (hand-written CUDA, no Tensor Cores)",
        "description": (
            "Hand-written CUDA kernel using standard SIMT programming model — scalar "
            "FP32 FMA instructions only, no Tensor Cores. BF16 inputs are converted "
            "to FP32 for computation, then back to BF16 for output. Uses shared memory "
            "tiling for data reuse. Represents the baseline performance achievable "
            "without any Tensor Core acceleration."
        ),
        "tensor_core_api": "None (scalar FMA only)",
        "memory_subsystem": "Shared memory tiling",
        "optimization_level": "Basic hand-written CUDA (tiled shared memory)",
        "abstraction_level": "Low-level CUDA C++ (hand-written kernel)",
    },
    "CuTile": {
        "full_name": "NVIDIA CuTile (CUDA Tile, Blackwell)",
        "description": (
            "NVIDIA's official high-level tile abstraction for GPU kernels (cuda.tile). "
            "Abstracts warps, registers, and shared memory while preserving Tensor Core "
            "and TMA utilization. Requires Blackwell GPU with compute capability 10.x or "
            "12.x. Uses ct.mma for matmul, ct.load/ct.store for TMA-backed memory ops. "
            "Kernel decorator uses ByTarget for architecture-specific CTA clustering: "
            "num_ctas=ByTarget(sm_100=2, sm_120=2)."
        ),
        "tensor_core_api": "Blackwell Tensor Cores via ct.mma",
        "memory_subsystem": "TMA (automatic via CuTile abstraction)",
        "optimization_level": "High-level (compiler-managed tiling)",
        "abstraction_level": "Python DSL (NVIDIA cuda.tile)",
        "sm_120_fix": "Added sm_120=2 to ByTarget (originally only had sm_100=2, causing 4x slowdown)",
    },
}


def calculate_tflops(M, N, K, time_ms):
    """Compute TFLOP/s for C = A @ B (2*M*N*K FLOPs)."""
    return (2 * M * N * K / (time_ms * 1e-3)) / 1e12


def benchmark_impl(name, gemm_fn, A, B, M, N, K):
    """Run warmup + timed iterations, return timing and TFLOP/s."""
    is_raw_simt = "Raw SIMT" in name
    warmup = RAW_SIMT_WARMUP if is_raw_simt else WARMUP_ITERATIONS
    iters = RAW_SIMT_ITERS if is_raw_simt else BENCHMARK_ITERATIONS

    try:
        # Warmup
        for _ in range(warmup):
            gemm_fn(A, B)
        torch.cuda.synchronize()

        # Timed region using CUDA events (most accurate GPU timing)
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


def run_benchmarks():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Collect device config ----
    config = get_device_config()

    print(f"Device: {config['hardware']['gpu_name']} ({config['hardware']['compute_capability']})",
          flush=True)
    print(f"  SMs: {config['hardware']['num_sms']}  "
          f"Shared mem/SM: {config['hardware']['shared_memory_per_sm_kb']} KB  "
          f"VRAM: {config['hardware']['global_memory_gb']} GB", flush=True)
    print(f"  Architecture: {config['hardware']['gpu_architecture']}  "
          f"SMs Clock: {config['hardware'].get('max_sm_clock_mhz', 'N/A')} MHz", flush=True)
    print(f"  CUDA: {config['software']['cuda_version']}  "
          f"Driver: {config['software'].get('nvidia_driver', 'N/A')}  "
          f"PyTorch: {config['software']['pytorch_version']}  "
          f"Triton: {config['software']['triton_version']}", flush=True)

    # ---- Import implementations ----
    from gemm_cublas import gemm_cublas
    from gemm_gmma import gemm_gmma, HAS_TRITON
    # CuTile NOT available on H100 (requires Blackwell sm_100/sm_120)

    HAS_RAW_SIMT = False
    raw_simt_fn = None
    try:
        import raw_simt_kernel
        raw_simt_fn = raw_simt_kernel.gemm
        HAS_RAW_SIMT = True
        print("\n  Raw SIMT kernel: loaded", flush=True)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"\n  Raw SIMT kernel: unavailable ({e})", flush=True)

    HAS_WMMA = False
    wmma_fn = None
    try:
        import wmma_kernel
        wmma_fn = wmma_kernel.gemm
        HAS_WMMA = True
        print("  WMMA kernel: loaded", flush=True)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"  WMMA kernel: unavailable ({e})", flush=True)

    # Updated naming: "Triton" instead of "GMMA (Triton)"
    # CuTile excluded — requires Blackwell GPU (sm_100/sm_120)
    implementations = [
        ("cuBLAS",     gemm_cublas,  True),
        ("Triton",     gemm_gmma,    HAS_TRITON),
        ("WMMA",       wmma_fn,      HAS_WMMA),
        ("Raw SIMT",   raw_simt_fn,  HAS_RAW_SIMT),
    ]

    # ---- Benchmark config ----
    dtype = torch.bfloat16
    benchmark_config = {
        "dtype": str(dtype),
        "matrix_type": "square (M=N=K)",
        "matrix_sizes": MATRIX_SIZES,
        "operation": "C = A @ B  (A: [M,K] x B: [K,N] -> C: [M,N])",
        "flop_counting": "2 * M * N * K (multiply-add counted as 2 ops)",
        "timing_method": "CUDA events (torch.cuda.Event, enable_timing=True)",
        "warmup_iterations": WARMUP_ITERATIONS,
        "benchmark_iterations": BENCHMARK_ITERATIONS,
        "raw_simt_warmup_iterations": RAW_SIMT_WARMUP,
        "raw_simt_benchmark_iterations": RAW_SIMT_ITERS,
    }

    # ---- Build results structure ----
    results = {
        "title": f"GEMM Benchmark — BF16 Square Matrices on {config['hardware']['gpu_name']}",
        "device_config": config,
        "benchmark_config": benchmark_config,
        "implementation_details": {},
        "results": {},
        "timestamp": datetime.now().isoformat(),
    }

    # Add implementation descriptions
    for impl_name, _, _ in implementations:
        if impl_name in IMPL_DESCRIPTIONS:
            results["implementation_details"][impl_name] = IMPL_DESCRIPTIONS[impl_name]

    # ---- Run benchmarks ----
    print("\n" + "=" * 80, flush=True)
    print("GEMM BENCHMARK — BF16 Square Matrices", flush=True)
    print("=" * 80, flush=True)

    for size in MATRIX_SIZES:
        M = N = K = size
        print(f"\n  Matrix: {M}x{K} x {K}x{N}  (FLOPs: {2*M*N*K/1e9:.1f} GFLOPs)", flush=True)
        print("  " + "-" * 56, flush=True)

        torch.manual_seed(42)
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(K, N, device="cuda", dtype=dtype)

        for impl_name, gemm_fn, available in implementations:
            if gemm_fn is None or not available:
                results["results"].setdefault(impl_name, {})[str(size)] = {
                    "status": "unavailable",
                    "reason": "Requires Blackwell GPU" if impl_name == "CuTile" else "Module not loaded"
                }
                print(f"    {impl_name:<18} N/A", flush=True)
                continue

            print(f"    {impl_name:<18} running...", end="", flush=True)
            r = benchmark_impl(impl_name, gemm_fn, A, B, M, N, K)
            results["results"].setdefault(impl_name, {})[str(size)] = r

            if r["status"] == "ok":
                print(f"\r    {impl_name:<18} {r['time_ms']:>8.4f} ms  "
                      f"{r['tflops']:>7.2f} TFLOP/s", flush=True)
            else:
                print(f"\r    {impl_name:<18} ERROR: {r.get('error', '')[:50]}", flush=True)

        del A, B
        torch.cuda.empty_cache()
        time.sleep(1)

    # ---- Save detailed JSON ----
    with open(RESULTS_DIR / "gemm_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Save device config separately ----
    with open(RESULTS_DIR / "device_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Results saved to {RESULTS_DIR}/", flush=True)

    # ---- Print paper-ready table ----
    impl_order = ["cuBLAS", "Triton", "WMMA", "Raw SIMT"]
    print("\n" + "=" * 90)
    gpu_name = config['hardware']['gpu_name']
    print(f"TABLE: GEMM Performance on {gpu_name} (BF16, square matrices)")
    print("=" * 90)
    header = f"  {'Implementation':<18}"
    for s in MATRIX_SIZES:
        header += f" {'%dx%d' % (s, s):>12}"
    print(header)
    print("  " + "-" * 86)
    for impl_name in impl_order:
        # Use descriptive row label
        desc = IMPL_DESCRIPTIONS.get(impl_name, {})
        label = impl_name
        row = f"  {label:<18}"
        for size in MATRIX_SIZES:
            r = results["results"].get(impl_name, {}).get(str(size), {})
            if r.get("status") == "ok":
                row += f" {r['tflops']:>8.1f} TF/s"
            else:
                row += f" {'---':>12}"
        print(row)

    # ---- Print speedup table ----
    print("\n" + "=" * 90)
    print("TABLE: Speedup vs Raw SIMT (showing Tensor Core & optimization impact)")
    print("=" * 90)
    header = f"  {'Implementation':<18}"
    for s in MATRIX_SIZES:
        header += f" {'%dx%d' % (s, s):>12}"
    print(header)
    print("  " + "-" * 86)
    for impl_name in impl_order:
        row = f"  {impl_name:<18}"
        for size in MATRIX_SIZES:
            r = results["results"].get(impl_name, {}).get(str(size), {})
            base = results["results"].get("Raw SIMT", {}).get(str(size), {})
            if r.get("status") == "ok" and base.get("status") == "ok":
                speedup = r["tflops"] / base["tflops"]
                row += f" {speedup:>10.1f}x"
            else:
                row += f" {'---':>12}"
        print(row)


if __name__ == "__main__":
    run_benchmarks()
