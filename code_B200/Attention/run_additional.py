#!/usr/bin/env python3
"""
Additional Attention Benchmarks for Research Paper
1. Longer sequence length (seq=8192) — BF16
2. FP16 attention benchmarks (all sequence lengths)

Results saved separately from the main BF16 attention results.
"""

import json
import time
import math
import platform
from datetime import datetime
from pathlib import Path

import torch

# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZE = 8
NUM_HEADS = 32
HEAD_DIM = 128
CAUSAL = True

# Part 1: BF16 with extended sequence lengths
EXTENDED_SEQ_LENGTHS = [8192]

# Part 2: FP16 for all sequence lengths
FP16_SEQ_LENGTHS = [512, 1024, 2048, 4096]

WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50
NAIVE_WARMUP = 3
NAIVE_ITERS = 10

RESULTS_DIR = Path(__file__).parent.parent.parent / "results_B200" / "Attention"


def get_device_config() -> dict:
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
    try:
        import flash_attn
        software["flash_attn_version"] = flash_attn.__version__
    except ImportError:
        software["flash_attn_version"] = "N/A"
    try:
        import cuda.tile as ct
        software["cuda_tile_version"] = ct.__version__
    except (ImportError, AttributeError):
        software["cuda_tile_version"] = "N/A"
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            software["nvidia_driver"] = r.stdout.strip()
    except Exception:
        software["nvidia_driver"] = "N/A"
    return {"hardware": hardware, "software": software}


def calculate_attention_flops(batch, heads, seq_len, head_dim, causal=True):
    flops = 4.0 * batch * heads * seq_len * seq_len * head_dim
    if causal:
        flops *= 0.5
    return flops


def prepare_inputs(impl_name, Q, K, V):
    if "FlashAttention" in impl_name:
        return (Q.transpose(1, 2).contiguous(),
                K.transpose(1, 2).contiguous(),
                V.transpose(1, 2).contiguous())
    return Q, K, V


def benchmark_attn(name, attn_fn, Q, K, V, causal, batch, heads, seq_len, head_dim):
    is_naive = "Naive" in name
    warmup = NAIVE_WARMUP if is_naive else WARMUP_ITERATIONS
    iters = NAIVE_ITERS if is_naive else BENCHMARK_ITERATIONS

    q, k, v = prepare_inputs(name, Q, K, V)

    try:
        for _ in range(warmup):
            attn_fn(q, k, v, causal=causal)
        torch.cuda.synchronize()

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


def load_implementations():
    from attn_flash_sdpa import attn_flash_sdpa
    from attn_cudnn_sdpa import attn_cudnn_sdpa
    from attn_naive import attn_naive

    triton_fn = None
    HAS_TRITON = False
    try:
        from attn_triton import attn_triton, HAS_TRITON as _HT
        if _HT:
            triton_fn = attn_triton
            HAS_TRITON = True
    except Exception:
        pass

    cutile_fn = None
    HAS_CUTILE = False
    try:
        from attn_cutile import attn_cutile, HAS_CUTILE as _HC
        if _HC:
            cutile_fn = attn_cutile
            HAS_CUTILE = True
    except Exception:
        pass

    return [
        ("FlashAttention-2",  attn_flash_sdpa,  True),
        ("cuDNN (SDPA)",      attn_cudnn_sdpa,   True),
        ("Triton",            triton_fn,         HAS_TRITON),
        ("CuTile",            cutile_fn,         HAS_CUTILE),
        ("Naive (Unfused)",   attn_naive,        True),
    ]


# ============================================================================
# Part 1: Extended Sequence Lengths (BF16)
# ============================================================================

def run_extended_seq():
    """Benchmark longer sequence lengths (seq=8192) with BF16."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = get_device_config()
    implementations = load_implementations()

    print("\n" + "=" * 90, flush=True)
    print("EXTENDED ATTENTION BENCHMARK — BF16, Long Sequences", flush=True)
    print("=" * 90, flush=True)

    results = {
        "title": f"Attention Benchmark — BF16 Extended Seq on {config['hardware']['gpu_name']}",
        "device_config": config,
        "benchmark_config": {
            "dtype": "torch.bfloat16",
            "batch": BATCH_SIZE,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "seq_lengths": EXTENDED_SEQ_LENGTHS,
            "causal": CAUSAL,
        },
        "results": {},
        "timestamp": datetime.now().isoformat(),
    }

    for seq_len in EXTENDED_SEQ_LENGTHS:
        flops = calculate_attention_flops(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, CAUSAL)
        print(f"\n  batch={BATCH_SIZE}, seq={seq_len}  (FLOPs: {flops/1e12:.2f} TFLOPs)", flush=True)
        print("  " + "-" * 70, flush=True)

        torch.manual_seed(42)
        Q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

        size_key = f"b{BATCH_SIZE}_s{seq_len}"

        for impl_name, attn_fn, available in implementations:
            if attn_fn is None or not available:
                results["results"].setdefault(impl_name, {})[size_key] = {"status": "unavailable"}
                print(f"    {impl_name:<26} N/A", flush=True)
                continue

            # Skip naive for seq>=8192 (OOM risk with O(N^2) memory)
            if "Naive" in impl_name and seq_len >= 8192:
                results["results"].setdefault(impl_name, {})[size_key] = {
                    "status": "skipped", "reason": "OOM risk for N^2 memory at seq=8192"
                }
                print(f"    {impl_name:<26} skipped (OOM risk)", flush=True)
                continue

            print(f"    {impl_name:<26} running...", end="", flush=True)
            r = benchmark_attn(impl_name, attn_fn, Q, K, V, CAUSAL,
                               BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM)
            results["results"].setdefault(impl_name, {})[size_key] = r

            if r["status"] == "ok":
                print(f"\r    {impl_name:<26} {r['time_ms']:>8.4f} ms  {r['tflops']:>7.2f} TFLOP/s", flush=True)
            else:
                print(f"\r    {impl_name:<26} ERROR: {r.get('error', '')[:50]}", flush=True)

        del Q, K, V
        torch.cuda.empty_cache()
        time.sleep(1)

    out_path = RESULTS_DIR / "attention_extended_seq_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}", flush=True)
    return results


# ============================================================================
# Part 2: FP16 Attention
# ============================================================================

def run_fp16_attention():
    """Benchmark all attention implementations with FP16 dtype."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = get_device_config()
    implementations = load_implementations()

    print("\n" + "=" * 90, flush=True)
    print("FP16 ATTENTION BENCHMARK", flush=True)
    print("=" * 90, flush=True)

    results = {
        "title": f"Attention Benchmark — FP16 Causal on {config['hardware']['gpu_name']}",
        "device_config": config,
        "benchmark_config": {
            "dtype": "torch.float16",
            "batch": BATCH_SIZE,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "seq_lengths": FP16_SEQ_LENGTHS,
            "causal": CAUSAL,
        },
        "results": {},
        "timestamp": datetime.now().isoformat(),
    }

    for seq_len in FP16_SEQ_LENGTHS:
        flops = calculate_attention_flops(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, CAUSAL)
        print(f"\n  batch={BATCH_SIZE}, seq={seq_len} (FP16)  (FLOPs: {flops/1e9:.1f} GFLOPs)", flush=True)
        print("  " + "-" * 70, flush=True)

        torch.manual_seed(42)
        Q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, device="cuda", dtype=torch.float16)
        K = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, device="cuda", dtype=torch.float16)
        V = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, device="cuda", dtype=torch.float16)

        size_key = f"b{BATCH_SIZE}_s{seq_len}"

        for impl_name, attn_fn, available in implementations:
            if attn_fn is None or not available:
                results["results"].setdefault(impl_name, {})[size_key] = {"status": "unavailable"}
                print(f"    {impl_name:<26} N/A", flush=True)
                continue

            print(f"    {impl_name:<26} running...", end="", flush=True)
            r = benchmark_attn(impl_name, attn_fn, Q, K, V, CAUSAL,
                               BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM)
            results["results"].setdefault(impl_name, {})[size_key] = r

            if r["status"] == "ok":
                print(f"\r    {impl_name:<26} {r['time_ms']:>8.4f} ms  {r['tflops']:>7.2f} TFLOP/s", flush=True)
            else:
                print(f"\r    {impl_name:<26} ERROR: {r.get('error', '')[:50]}", flush=True)

        del Q, K, V
        torch.cuda.empty_cache()
        time.sleep(0.5)

    out_path = RESULTS_DIR / "attention_fp16_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}", flush=True)
    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ADDITIONAL ATTENTION BENCHMARKS")
    print("=" * 80)

    print("\n>>> Part 1: Extended Sequence Lengths (BF16, seq=8192)")
    run_extended_seq()

    print("\n\n>>> Part 2: FP16 Attention (all seq lengths)")
    run_fp16_attention()

    print("\n\nAll additional attention benchmarks complete!")
