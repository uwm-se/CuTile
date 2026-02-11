#!/usr/bin/env python3
"""
Fused Attention Benchmark Runner for Research Paper (Table 3)
Benchmarks FlashAttention-2, cuDNN SDPA, Triton Flash Attention, and Naive Unfused.
Saves comprehensive results with hardware config and implementation descriptions.
"""

import json
import time
import math
import platform
from datetime import datetime
from pathlib import Path

import torch

# ============================================================================
# Benchmark Configuration
# ============================================================================

# LLaMA-7B-like attention config
BATCH_SIZES = [8]
NUM_HEADS = 32
HEAD_DIM = 128
SEQ_LENGTHS = [512, 1024, 2048, 4096]
CAUSAL = True   # Causal (autoregressive) attention, standard for LLM inference

WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50
NAIVE_WARMUP = 3
NAIVE_ITERS = 10   # Naive is slower + uses O(N^2) memory

RESULTS_DIR = Path(__file__).parent.parent.parent / "results_H100" / "Attention"


def get_device_config() -> dict:
    """Collect hardware and software details."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    props = torch.cuda.get_device_properties(0)
    def detect_architecture(major, minor):
        if major >= 12:
            return "Blackwell"
        elif major >= 10:
            return "Blackwell"
        elif major == 9:
            return "Hopper"
        elif major == 8:
            return "Ampere"
        elif major == 7:
            return "Volta/Turing"
        return "Unknown"

    hardware = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_architecture": detect_architecture(props.major, props.minor),
        "compute_capability": f"sm_{props.major}{props.minor}",
        "num_sms": props.multi_processor_count,
        "shared_memory_per_sm_kb": round(props.shared_memory_per_multiprocessor / 1024, 1),
        "global_memory_gb": round(props.total_memory / (1024**3), 2),
    }
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.sm,clocks.max.mem,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True)
        if r.returncode == 0:
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            hardware["max_sm_clock_mhz"] = parts[0].replace(" MHz", "")
            hardware["max_mem_clock_mhz"] = parts[1].replace(" MHz", "")
    except Exception:
        pass
    software = {
        "cuda_version": torch.version.cuda or "N/A",
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
    }
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            software["nvidia_driver"] = r.stdout.strip()
    except Exception:
        software["nvidia_driver"] = "N/A"
    try:
        import triton
        software["triton_version"] = triton.__version__
    except ImportError:
        software["triton_version"] = "N/A"
    try:
        import flash_attn
        software["flash_attn_version"] = flash_attn.__version__
    except ImportError:
        software["flash_attn_version"] = "N/A"
    return {"hardware": hardware, "software": software}


# ============================================================================
# Implementation Descriptions
# ============================================================================

IMPL_DESCRIPTIONS = {
    "FlashAttention-2": {
        "full_name": "FlashAttention-2 (Tri Dao, flash-attn 2.8.3)",
        "description": (
            "Official FlashAttention-2 implementation from the flash-attn "
            "package (Tri Dao). Uses the IO-aware tiling algorithm with "
            "online softmax to avoid materializing the N x N attention matrix. "
            "Optimized CUDA kernels with Tensor Core acceleration. "
            "Memory complexity: O(N) instead of O(N^2). "
            "Reference: https://github.com/Dao-AILab/flash-attention"
        ),
        "algorithm": "FlashAttention-2 (online softmax, tiled IO)",
        "tensor_cores": "Yes (optimized CUDA kernels)",
        "memory_complexity": "O(N) - no N^2 attention matrix",
        "fusion": "Fully fused (QK^T + softmax + AV in single kernel)",
        "package": "flash-attn 2.8.3",
    },
    "cuDNN (SDPA)": {
        "full_name": "cuDNN Fused Attention via PyTorch SDPA",
        "description": (
            "NVIDIA cuDNN's fused multi-head attention, accessed through "
            "PyTorch's SDPA efficient_attention backend. Uses NVIDIA's "
            "vendor-optimized attention implementation with Tensor Cores. "
            "Different algorithm/implementation from FlashAttention-2."
        ),
        "algorithm": "cuDNN fused MHA (vendor-optimized)",
        "tensor_cores": "Yes (cuDNN Tensor Core kernels)",
        "memory_complexity": "O(N) - memory-efficient implementation",
        "fusion": "Fully fused (vendor implementation)",
    },
    "Triton": {
        "full_name": "Triton Fused Attention (Flash Attention v2)",
        "description": (
            "Flash Attention v2 algorithm implemented in Triton, adapted from "
            "the official tutorial (06-fused-attention.py). Uses online softmax "
            "with tiled Q/K/V processing. tl.dot compiles to architecture-native "
            "Tensor Core instructions. Autotuned across 6 BLOCK_M/BLOCK_N/warp configs."
        ),
        "algorithm": "Flash Attention v2 (online softmax, tiled)",
        "tensor_cores": "Yes (architecture-native Tensor Core instructions via tl.dot)",
        "memory_complexity": "O(N) - no N^2 attention matrix",
        "fusion": "Fully fused (single Triton kernel)",
    },
    "Naive (Unfused)": {
        "full_name": "Naive Unfused Attention (separate matmuls + softmax)",
        "description": (
            "Standard textbook attention: S = Q @ K^T * scale (cuBLAS), "
            "P = softmax(S) (PyTorch), O = P @ V (cuBLAS). Three separate "
            "kernel launches with full N x N attention matrix materialized "
            "in global memory. Represents the baseline without any fusion "
            "or memory optimization."
        ),
        "algorithm": "Standard matmul + softmax + matmul (3 separate ops)",
        "tensor_cores": "Yes (cuBLAS matmuls use Tensor Cores)",
        "memory_complexity": "O(N^2) - full attention matrix in memory",
        "fusion": "None (3 separate kernel launches)",
    },
    "CuTile": {
        "full_name": "NVIDIA CuTile FMHA (Blackwell-only)",
        "description": (
            "NVIDIA's official high-level tile abstraction for Fused Multi-Head "
            "Attention, based on the cuda.tile AttentionFMHA.py sample. Implements "
            "the FlashAttention-2 online softmax algorithm using ct.mma for Tensor "
            "Core acceleration and ct.load/ct.store for TMA-backed memory ops. "
            "Requires Blackwell GPU (B200, B100, RTX 5090) with compute 10.x/12.x."
        ),
        "algorithm": "Flash Attention v2 (online softmax, tiled) via CuTile",
        "tensor_cores": "Yes (Blackwell Tensor Cores via ct.mma)",
        "memory_complexity": "O(N) - no N^2 attention matrix",
        "fusion": "Fully fused (single CuTile kernel)",
    },
}


def calculate_attention_flops(batch, heads, seq_len, head_dim, causal=True):
    """
    FLOPs for forward attention:
      QK^T: 2 * B * H * N * N * D
      P @ V: 2 * B * H * N * N * D
      Total: 4 * B * H * N^2 * D
      Causal: divide by 2 (triangular)
    """
    flops = 4.0 * batch * heads * seq_len * seq_len * head_dim
    if causal:
        flops *= 0.5
    return flops


def prepare_inputs(impl_name, Q, K, V):
    """Prepare inputs in the native layout for each implementation.

    flash_attn expects [B, N, H, D]; everything else uses [B, H, N, D].
    The transpose is done ONCE, outside the timing loop, so we benchmark
    only the kernel compute, not data reshaping overhead.
    """
    if "FlashAttention" in impl_name:
        # [B, H, N, D] -> [B, N, H, D] for flash_attn native layout
        return (Q.transpose(1, 2).contiguous(),
                K.transpose(1, 2).contiguous(),
                V.transpose(1, 2).contiguous())
    return Q, K, V


def benchmark_attn(name, attn_fn, Q, K, V, causal, batch, heads, seq_len, head_dim):
    """Run warmup + timed iterations."""
    is_naive = "Naive" in name
    warmup = NAIVE_WARMUP if is_naive else WARMUP_ITERATIONS
    iters = NAIVE_ITERS if is_naive else BENCHMARK_ITERATIONS

    # Prepare data in native layout (outside timing)
    q, k, v = prepare_inputs(name, Q, K, V)

    try:
        # Warmup
        for _ in range(warmup):
            attn_fn(q, k, v, causal=causal)
        torch.cuda.synchronize()

        # Timed region
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            attn_fn(q, k, v, causal=causal)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / iters
        flops = calculate_attention_flops(batch, heads, seq_len, head_dim, causal)
        tflops = (flops / (elapsed_ms * 1e-3)) / 1e12

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
    config = get_device_config()

    print(f"Device: {config['hardware']['gpu_name']} ({config['hardware']['compute_capability']})", flush=True)
    print(f"  SMs: {config['hardware']['num_sms']}  VRAM: {config['hardware']['global_memory_gb']} GB", flush=True)
    print(f"  CUDA: {config['software']['cuda_version']}  PyTorch: {config['software']['pytorch_version']}  "
          f"Triton: {config['software']['triton_version']}", flush=True)

    # Import implementations
    from attn_flash_sdpa import attn_flash_sdpa
    from attn_cudnn_sdpa import attn_cudnn_sdpa
    from attn_naive import attn_naive

    HAS_TRITON_ATTN = False
    triton_fn = None
    try:
        from attn_triton import attn_triton, HAS_TRITON
        if HAS_TRITON:
            triton_fn = attn_triton
            HAS_TRITON_ATTN = True
            print("  Triton attention: loaded", flush=True)
    except Exception as e:
        print(f"  Triton attention: unavailable ({e})", flush=True)

    # CuTile NOT available on H100 (requires Blackwell sm_100/sm_120)

    implementations = [
        ("FlashAttention-2",  attn_flash_sdpa,  True),
        ("cuDNN (SDPA)",      attn_cudnn_sdpa,   True),
        ("Triton",            triton_fn,         HAS_TRITON_ATTN),
        ("Naive (Unfused)",   attn_naive,        True),
    ]

    dtype = torch.bfloat16
    benchmark_config = {
        "dtype": str(dtype),
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "batch_sizes": BATCH_SIZES,
        "seq_lengths": SEQ_LENGTHS,
        "causal": CAUSAL,
        "model_reference": "LLaMA-7B-like (32 heads, d=128)",
        "operation": "Scaled Dot-Product Attention: O = softmax(Q @ K^T / sqrt(d)) @ V",
        "flop_counting": "4 * B * H * N^2 * D (causal: /2)",
        "timing_method": "CUDA events",
        "warmup_iterations": WARMUP_ITERATIONS,
        "benchmark_iterations": BENCHMARK_ITERATIONS,
    }

    results = {
        "title": f"Fused Attention Benchmark - BF16 Causal Attention on {config['hardware']['gpu_name']}",
        "device_config": config,
        "benchmark_config": benchmark_config,
        "implementation_details": IMPL_DESCRIPTIONS,
        "results": {},
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n  Attention config: {NUM_HEADS} heads, d={HEAD_DIM}, causal={CAUSAL}", flush=True)
    print(f"  Seq lengths: {SEQ_LENGTHS}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("FUSED ATTENTION BENCHMARK - BF16, Causal, LLaMA-7B config", flush=True)
    print("=" * 90, flush=True)

    for batch in BATCH_SIZES:
        for seq_len in SEQ_LENGTHS:
            flops = calculate_attention_flops(batch, NUM_HEADS, seq_len, HEAD_DIM, CAUSAL)
            print(f"\n  batch={batch}, seq={seq_len}, heads={NUM_HEADS}, d={HEAD_DIM}  "
                  f"(FLOPs: {flops/1e9:.1f} GFLOPs)", flush=True)
            print("  " + "-" * 70, flush=True)

            torch.manual_seed(42)
            Q = torch.randn(batch, NUM_HEADS, seq_len, HEAD_DIM, device="cuda", dtype=dtype)
            K = torch.randn(batch, NUM_HEADS, seq_len, HEAD_DIM, device="cuda", dtype=dtype)
            V = torch.randn(batch, NUM_HEADS, seq_len, HEAD_DIM, device="cuda", dtype=dtype)

            size_key = f"b{batch}_s{seq_len}"

            for impl_name, attn_fn, available in implementations:
                if attn_fn is None or not available:
                    results["results"].setdefault(impl_name, {})[size_key] = {"status": "unavailable"}
                    print(f"    {impl_name:<26} N/A", flush=True)
                    continue

                # Skip naive for very long sequences (OOM risk)
                if "Naive" in impl_name and seq_len > 4096:
                    results["results"].setdefault(impl_name, {})[size_key] = {
                        "status": "skipped", "reason": "OOM risk for N^2 memory"
                    }
                    print(f"    {impl_name:<26} skipped (OOM risk)", flush=True)
                    continue

                print(f"    {impl_name:<26} running...", end="", flush=True)
                r = benchmark_attn(impl_name, attn_fn, Q, K, V, CAUSAL,
                                   batch, NUM_HEADS, seq_len, HEAD_DIM)
                results["results"].setdefault(impl_name, {})[size_key] = r

                if r["status"] == "ok":
                    print(f"\r    {impl_name:<26} {r['time_ms']:>8.4f} ms  "
                          f"{r['tflops']:>7.2f} TFLOP/s", flush=True)
                else:
                    print(f"\r    {impl_name:<26} ERROR: {r.get('error', '')[:50]}", flush=True)

            del Q, K, V
            torch.cuda.empty_cache()
            time.sleep(0.5)

    # Save
    with open(RESULTS_DIR / "attention_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(RESULTS_DIR / "device_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR}/", flush=True)

    # Paper table
    impl_order = ["FlashAttention-2", "cuDNN (SDPA)", "Triton", "Naive (Unfused)"]
    print("\n" + "=" * 90)
    gpu_name = config['hardware']['gpu_name']
    print(f"TABLE: Fused Attention on {gpu_name} (BF16, causal, batch={BATCH_SIZES[0]}, "
          f"{NUM_HEADS}h x d{HEAD_DIM})")
    print("=" * 90)

    header = f"  {'Implementation':<26}"
    for s in SEQ_LENGTHS:
        header += f"  {'seq=%d' % s:>12}"
    print(header)
    print("  " + "-" * 86)

    for impl_name in impl_order:
        row = f"  {impl_name:<26}"
        for seq_len in SEQ_LENGTHS:
            key = f"b{BATCH_SIZES[0]}_s{seq_len}"
            r = results["results"].get(impl_name, {}).get(key, {})
            if r.get("status") == "ok":
                row += f"  {r['tflops']:>8.1f} TF/s"
            else:
                row += f"  {'---':>12}"
        print(row)

    # Time table
    print()
    header = f"  {'Implementation':<26}"
    for s in SEQ_LENGTHS:
        header += f"  {'seq=%d' % s:>12}"
    print(header)
    print("  " + "-" * 86)

    for impl_name in impl_order:
        row = f"  {impl_name:<26}"
        for seq_len in SEQ_LENGTHS:
            key = f"b{BATCH_SIZES[0]}_s{seq_len}"
            r = results["results"].get(impl_name, {}).get(key, {})
            if r.get("status") == "ok":
                row += f"  {r['time_ms']:>9.3f} ms"
            else:
                row += f"  {'---':>12}"
        print(row)


if __name__ == "__main__":
    run_benchmarks()
