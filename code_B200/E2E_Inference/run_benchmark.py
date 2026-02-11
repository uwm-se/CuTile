#!/usr/bin/env python3
"""
End-to-End Inference Benchmark Runner (Tables 4 & 5)
Measures prefill throughput (tokens/sec) and decode latency (ms/token)
for a LLaMA-7B-like transformer stack across different implementations.

Implementations:
  1. Eager (Naive)    - unfused matmul attention, no compilation
  2. Eager (SDPA)     - PyTorch SDPA (auto-dispatches to flash/efficient)
  3. Eager (FA2)      - official flash_attn package
  4. torch.compile    - compiled graph with SDPA backend
"""

import json
import time
import platform
import gc
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import LLaMAModel, DIM, NUM_HEADS, HEAD_DIM, FFN_DIM, NUM_LAYERS

RESULTS_DIR = Path(__file__).parent.parent.parent / "results_B200" / "E2E_Inference"

# ============================================================================
# Benchmark Configuration
# ============================================================================

PREFILL_SEQ_LEN = 2048
DECODE_CONTEXT_LEN = 2048
BATCH_SIZE = 1
DTYPE = torch.bfloat16
NUM_LAYERS_BENCH = NUM_LAYERS  # 4 layers

WARMUP_ITERS = 5
PREFILL_ITERS = 20
DECODE_ITERS = 50
COMPILE_WARMUP = 3  # Extra warmup for torch.compile (first calls trigger compilation)


def get_device_config():
    """Collect hardware and software details."""
    props = torch.cuda.get_device_properties(0)
    config = {
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": f"sm_{props.major}{props.minor}",
            "num_sms": props.multi_processor_count,
            "global_memory_gb": round(props.total_memory / (1024**3), 2),
        },
        "software": {
            "cuda_version": torch.version.cuda or "N/A",
            "pytorch_version": torch.__version__,
            "python_version": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}",
        }
    }
    try:
        import triton
        config["software"]["triton_version"] = triton.__version__
    except ImportError:
        config["software"]["triton_version"] = "N/A"
    try:
        import flash_attn
        config["software"]["flash_attn_version"] = flash_attn.__version__
    except ImportError:
        config["software"]["flash_attn_version"] = "N/A"
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            config["software"]["nvidia_driver"] = r.stdout.strip()
    except Exception:
        pass
    return config


def calculate_prefill_flops(batch, seq_len, dim, num_heads, head_dim, ffn_dim, num_layers):
    """FLOPs per prefill pass (forward only)."""
    flops_per_layer = 0
    # QKV projections: 3 * 2*B*S*D*D
    flops_per_layer += 3 * 2 * batch * seq_len * dim * dim
    # Attention: QK^T + P@V = 2 * 2*B*H*S*S*d (causal /2)
    flops_per_layer += 2 * 2 * batch * num_heads * seq_len * seq_len * head_dim // 2
    # Output projection: 2*B*S*D*D
    flops_per_layer += 2 * batch * seq_len * dim * dim
    # FFN: gate + up + down = 3 * 2*B*S*D*FFN
    flops_per_layer += 3 * 2 * batch * seq_len * dim * ffn_dim
    return flops_per_layer * num_layers


def calculate_decode_flops(batch, context_len, dim, num_heads, head_dim, ffn_dim, num_layers):
    """FLOPs per single decode step (S_q=1)."""
    flops_per_layer = 0
    # QKV projections: 3 * 2*B*1*D*D
    flops_per_layer += 3 * 2 * batch * 1 * dim * dim
    # Attention: Q(1)@K^T(ctx) + P@V = 2 * 2*B*H*1*ctx*d
    flops_per_layer += 2 * 2 * batch * num_heads * 1 * context_len * head_dim
    # Output projection: 2*B*1*D*D
    flops_per_layer += 2 * batch * 1 * dim * dim
    # FFN: 3 * 2*B*1*D*FFN
    flops_per_layer += 3 * 2 * batch * 1 * dim * ffn_dim
    return flops_per_layer * num_layers


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_prefill(model, batch_size, seq_len, warmup, iters, label=""):
    """Benchmark prefill (full sequence forward pass)."""
    x = torch.randn(batch_size, seq_len, DIM, device="cuda", dtype=DTYPE)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.prefill(x)
    torch.cuda.synchronize()

    # Timed
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        with torch.no_grad():
            model.prefill(x)
    end_ev.record()
    torch.cuda.synchronize()

    avg_ms = start_ev.elapsed_time(end_ev) / iters
    tokens_per_sec = (batch_size * seq_len) / (avg_ms * 1e-3)
    flops = calculate_prefill_flops(batch_size, seq_len, DIM, NUM_HEADS, HEAD_DIM, FFN_DIM, NUM_LAYERS_BENCH)
    tflops = (flops / (avg_ms * 1e-3)) / 1e12

    return {
        "time_ms": round(avg_ms, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "tflops": round(tflops, 2),
        "status": "ok",
    }


def benchmark_decode(model, batch_size, context_len, warmup, iters, label=""):
    """Benchmark single-token decode with pre-populated KV cache."""
    # Generate KV cache via prefill
    prefill_x = torch.randn(batch_size, context_len, DIM, device="cuda", dtype=DTYPE)
    with torch.no_grad():
        _, kv_caches = model.prefill(prefill_x)
    del prefill_x
    torch.cuda.synchronize()

    # Decode input: single new token
    decode_x = torch.randn(batch_size, 1, DIM, device="cuda", dtype=DTYPE)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.decode_step(decode_x, kv_caches)
    torch.cuda.synchronize()

    # Timed
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        with torch.no_grad():
            model.decode_step(decode_x, kv_caches)
    end_ev.record()
    torch.cuda.synchronize()

    avg_ms = start_ev.elapsed_time(end_ev) / iters
    tokens_per_sec = batch_size / (avg_ms * 1e-3)
    flops = calculate_decode_flops(batch_size, context_len, DIM, NUM_HEADS, HEAD_DIM, FFN_DIM, NUM_LAYERS_BENCH)
    tflops = (flops / (avg_ms * 1e-3)) / 1e12

    del kv_caches
    return {
        "time_ms": round(avg_ms, 4),
        "ms_per_token": round(avg_ms, 4),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "tflops": round(tflops, 2),
        "status": "ok",
    }


def create_model(attn_mode):
    """Create a fresh model with given attention backend."""
    model = LLaMAModel(
        num_layers=NUM_LAYERS_BENCH, dim=DIM, num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM, attn_mode=attn_mode,
    ).to(device="cuda", dtype=DTYPE).eval()
    return model


def run_single_benchmark(name, attn_mode, use_compile=False):
    """Run prefill + decode benchmarks for one implementation."""
    print(f"\n  [{name}]", flush=True)

    model = create_model(attn_mode)

    if use_compile:
        print(f"    Compiling model (this may take a minute)...", flush=True)
        model = torch.compile(model, mode="reduce-overhead")
        # Extra warmup to trigger compilation
        dummy = torch.randn(BATCH_SIZE, PREFILL_SEQ_LEN, DIM, device="cuda", dtype=DTYPE)
        for _ in range(COMPILE_WARMUP):
            with torch.no_grad():
                model.prefill(dummy)
        torch.cuda.synchronize()
        del dummy

        # Also compile decode path
        dummy_prefill = torch.randn(BATCH_SIZE, DECODE_CONTEXT_LEN, DIM, device="cuda", dtype=DTYPE)
        with torch.no_grad():
            _, kv = model.prefill(dummy_prefill)
        dummy_decode = torch.randn(BATCH_SIZE, 1, DIM, device="cuda", dtype=DTYPE)
        for _ in range(COMPILE_WARMUP):
            with torch.no_grad():
                model.decode_step(dummy_decode, kv)
        torch.cuda.synchronize()
        del dummy_prefill, dummy_decode, kv
        torch.cuda.empty_cache()

    result = {"name": name, "attn_mode": attn_mode, "compiled": use_compile}

    # Prefill benchmark
    try:
        print(f"    Prefill (B={BATCH_SIZE}, S={PREFILL_SEQ_LEN})...", end="", flush=True)
        prefill_result = benchmark_prefill(model, BATCH_SIZE, PREFILL_SEQ_LEN,
                                           warmup=WARMUP_ITERS, iters=PREFILL_ITERS)
        result["prefill"] = prefill_result
        if prefill_result["status"] == "ok":
            print(f" {prefill_result['time_ms']:.3f} ms, "
                  f"{prefill_result['tokens_per_sec']:.0f} tok/s, "
                  f"{prefill_result['tflops']:.1f} TF/s", flush=True)
    except Exception as e:
        result["prefill"] = {"status": "error", "error": str(e)[:200]}
        print(f" ERROR: {str(e)[:80]}", flush=True)

    torch.cuda.empty_cache()

    # Decode benchmark
    try:
        print(f"    Decode  (B={BATCH_SIZE}, ctx={DECODE_CONTEXT_LEN})...", end="", flush=True)
        decode_result = benchmark_decode(model, BATCH_SIZE, DECODE_CONTEXT_LEN,
                                         warmup=WARMUP_ITERS, iters=DECODE_ITERS)
        result["decode"] = decode_result
        if decode_result["status"] == "ok":
            print(f" {decode_result['ms_per_token']:.4f} ms/tok, "
                  f"{decode_result['tokens_per_sec']:.0f} tok/s, "
                  f"{decode_result['tflops']:.2f} TF/s", flush=True)
    except Exception as e:
        result["decode"] = {"status": "error", "error": str(e)[:200]}
        print(f" ERROR: {str(e)[:80]}", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)
    return result


# ============================================================================
# Implementation Descriptions
# ============================================================================

IMPL_DESCRIPTIONS = {
    "Eager (Naive)": {
        "description": (
            "PyTorch eager mode with manual unfused attention (separate matmul, "
            "softmax, matmul). No kernel fusion, no compilation. Baseline showing "
            "the cost of unfused operations."
        ),
        "attention": "Manual matmul + softmax + matmul (3 kernel launches)",
        "ffn": "PyTorch eager nn.Linear + F.silu",
        "compilation": "None",
    },
    "Eager (SDPA)": {
        "description": (
            "PyTorch eager mode with scaled_dot_product_attention, which "
            "auto-dispatches to FlashAttention or memory-efficient backend. "
            "Attention is fused, but FFN/norm/residual remain unfused."
        ),
        "attention": "F.scaled_dot_product_attention (auto backend)",
        "ffn": "PyTorch eager nn.Linear + F.silu",
        "compilation": "None",
    },
    "Eager (FA2)": {
        "description": (
            "PyTorch eager mode with official flash_attn package (Tri Dao). "
            "Uses FlashAttention-2 optimized CUDA kernels for attention. "
            "FFN/norm/residual remain unfused."
        ),
        "attention": "flash_attn_func (FlashAttention-2, flash-attn 2.8.3)",
        "ffn": "PyTorch eager nn.Linear + F.silu",
        "compilation": "None",
    },
    "torch.compile (SDPA)": {
        "description": (
            "PyTorch 2.x torch.compile with inductor backend. Compiles the "
            "entire model graph, fusing elementwise ops (norm, residual, "
            "activation) and selecting optimal attention backend. "
            "Mode: reduce-overhead (CUDA graph capture)."
        ),
        "attention": "Compiler-selected (typically FlashAttention via SDPA)",
        "ffn": "Compiled (fused elementwise + linear)",
        "compilation": "torch.compile(mode='reduce-overhead')",
    },
}


def run_benchmarks():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = get_device_config()

    print("=" * 80, flush=True)
    print("END-TO-END INFERENCE BENCHMARK", flush=True)
    print("=" * 80, flush=True)
    print(f"Device: {config['hardware']['gpu_name']} ({config['hardware']['compute_capability']})", flush=True)
    print(f"  CUDA: {config['software']['cuda_version']}  PyTorch: {config['software']['pytorch_version']}  "
          f"Triton: {config['software'].get('triton_version', 'N/A')}", flush=True)
    print(f"\nModel: LLaMA-7B config ({NUM_LAYERS_BENCH} layers, dim={DIM}, {NUM_HEADS} heads, "
          f"d={HEAD_DIM}, FFN={FFN_DIM})", flush=True)
    print(f"Dtype: {DTYPE}", flush=True)
    print(f"Prefill: batch={BATCH_SIZE}, seq_len={PREFILL_SEQ_LEN}", flush=True)
    print(f"Decode:  batch={BATCH_SIZE}, context={DECODE_CONTEXT_LEN}", flush=True)

    # Calculate FLOPs
    prefill_flops = calculate_prefill_flops(BATCH_SIZE, PREFILL_SEQ_LEN, DIM, NUM_HEADS, HEAD_DIM, FFN_DIM, NUM_LAYERS_BENCH)
    decode_flops = calculate_decode_flops(BATCH_SIZE, DECODE_CONTEXT_LEN, DIM, NUM_HEADS, HEAD_DIM, FFN_DIM, NUM_LAYERS_BENCH)
    print(f"Prefill FLOPs: {prefill_flops/1e12:.3f} TFLOPs", flush=True)
    print(f"Decode FLOPs:  {decode_flops/1e9:.3f} GFLOPs", flush=True)

    implementations = [
        ("Eager (Naive)",           "naive", False),
        ("Eager (SDPA)",            "sdpa",  False),
        ("Eager (FA2)",             "flash", False),
        ("torch.compile (SDPA)",    "sdpa",  True),
    ]

    all_results = []
    for name, attn_mode, use_compile in implementations:
        try:
            result = run_single_benchmark(name, attn_mode, use_compile)
            all_results.append(result)
        except Exception as e:
            print(f"\n  [{name}] FAILED: {e}", flush=True)
            all_results.append({"name": name, "status": "failed", "error": str(e)[:200]})

    # Summary tables
    print("\n" + "=" * 80)
    print("TABLE 4: PREFILL THROUGHPUT")
    print(f"  (LLaMA-7B config, {NUM_LAYERS_BENCH} layers, batch={BATCH_SIZE}, seq={PREFILL_SEQ_LEN})")
    print("=" * 80)
    print(f"  {'Implementation':<26} {'Time (ms)':>10} {'Tokens/s':>12} {'TFLOP/s':>10}")
    print("  " + "-" * 62)
    for r in all_results:
        p = r.get("prefill", {})
        if p.get("status") == "ok":
            print(f"  {r['name']:<26} {p['time_ms']:>10.3f} {p['tokens_per_sec']:>12.0f} {p['tflops']:>10.1f}")
        else:
            print(f"  {r['name']:<26} {'ERROR':>10} {'---':>12} {'---':>10}")

    print(f"\nTABLE 5: DECODE LATENCY")
    print(f"  (LLaMA-7B config, {NUM_LAYERS_BENCH} layers, batch={BATCH_SIZE}, context={DECODE_CONTEXT_LEN})")
    print("=" * 80)
    print(f"  {'Implementation':<26} {'ms/token':>10} {'Tokens/s':>12} {'TFLOP/s':>10}")
    print("  " + "-" * 62)
    for r in all_results:
        d = r.get("decode", {})
        if d.get("status") == "ok":
            print(f"  {r['name']:<26} {d['ms_per_token']:>10.4f} {d['tokens_per_sec']:>12.0f} {d['tflops']:>10.2f}")
        else:
            print(f"  {r['name']:<26} {'ERROR':>10} {'---':>12} {'---':>10}")

    # Speedup table
    baseline_prefill = None
    baseline_decode = None
    for r in all_results:
        if r.get("name") == "Eager (Naive)":
            if r.get("prefill", {}).get("status") == "ok":
                baseline_prefill = r["prefill"]["time_ms"]
            if r.get("decode", {}).get("status") == "ok":
                baseline_decode = r["decode"]["ms_per_token"]

    if baseline_prefill and baseline_decode:
        print(f"\nSPEEDUP vs Eager (Naive):")
        print("  " + "-" * 50)
        for r in all_results:
            p = r.get("prefill", {})
            d = r.get("decode", {})
            p_speedup = baseline_prefill / p["time_ms"] if p.get("status") == "ok" else 0
            d_speedup = baseline_decode / d["ms_per_token"] if d.get("status") == "ok" else 0
            print(f"  {r['name']:<26} Prefill: {p_speedup:>5.2f}x  Decode: {d_speedup:>5.2f}x")

    # Save results
    output = {
        "title": f"End-to-End Inference Benchmark - LLaMA-7B config on {config['hardware']['gpu_name']}",
        "device_config": config,
        "benchmark_config": {
            "model": "LLaMA-7B architecture",
            "num_layers": NUM_LAYERS_BENCH,
            "dim": DIM,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "ffn_dim": FFN_DIM,
            "dtype": str(DTYPE),
            "prefill_seq_len": PREFILL_SEQ_LEN,
            "decode_context_len": DECODE_CONTEXT_LEN,
            "batch_size": BATCH_SIZE,
            "prefill_flops": prefill_flops,
            "decode_flops": decode_flops,
            "warmup_iterations": WARMUP_ITERS,
            "prefill_iterations": PREFILL_ITERS,
            "decode_iterations": DECODE_ITERS,
        },
        "implementation_details": IMPL_DESCRIPTIONS,
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    with open(RESULTS_DIR / "e2e_inference_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/e2e_inference_results.json", flush=True)


if __name__ == "__main__":
    run_benchmarks()
