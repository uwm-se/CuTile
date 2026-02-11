#!/usr/bin/env python3
"""
Additional E2E Inference Benchmarks for Research Paper
Runs batch=8 and batch=32 in addition to the existing batch=1 results.

Results saved separately from the main batch=1 results.
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

RESULTS_DIR = Path(__file__).parent.parent.parent / "results_H100" / "E2E_Inference"

# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZES = [8, 32]  # Additional batch sizes (batch=1 already done)
PREFILL_SEQ_LEN = 2048
DECODE_CONTEXT_LEN = 2048
DTYPE = torch.bfloat16
NUM_LAYERS_BENCH = NUM_LAYERS  # 4 layers

WARMUP_ITERS = 5
PREFILL_ITERS = 20
DECODE_ITERS = 50
COMPILE_WARMUP = 3


def get_device_config():
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
    flops_per_layer = 0
    flops_per_layer += 3 * 2 * batch * seq_len * dim * dim
    flops_per_layer += 2 * 2 * batch * num_heads * seq_len * seq_len * head_dim // 2
    flops_per_layer += 2 * batch * seq_len * dim * dim
    flops_per_layer += 3 * 2 * batch * seq_len * dim * ffn_dim
    return flops_per_layer * num_layers


def calculate_decode_flops(batch, context_len, dim, num_heads, head_dim, ffn_dim, num_layers):
    flops_per_layer = 0
    flops_per_layer += 3 * 2 * batch * 1 * dim * dim
    flops_per_layer += 2 * 2 * batch * num_heads * 1 * context_len * head_dim
    flops_per_layer += 2 * batch * 1 * dim * dim
    flops_per_layer += 3 * 2 * batch * 1 * dim * ffn_dim
    return flops_per_layer * num_layers


def benchmark_prefill(model, batch_size, seq_len, warmup, iters):
    x = torch.randn(batch_size, seq_len, DIM, device="cuda", dtype=DTYPE)

    for _ in range(warmup):
        with torch.no_grad():
            model.prefill(x)
    torch.cuda.synchronize()

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


def benchmark_decode(model, batch_size, context_len, warmup, iters):
    prefill_x = torch.randn(batch_size, context_len, DIM, device="cuda", dtype=DTYPE)
    with torch.no_grad():
        _, kv_caches = model.prefill(prefill_x)
    del prefill_x
    torch.cuda.synchronize()

    decode_x = torch.randn(batch_size, 1, DIM, device="cuda", dtype=DTYPE)

    for _ in range(warmup):
        with torch.no_grad():
            model.decode_step(decode_x, kv_caches)
    torch.cuda.synchronize()

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
        "ms_per_token": round(avg_ms / batch_size, 4),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "tflops": round(tflops, 2),
        "status": "ok",
    }


def create_model(attn_mode):
    model = LLaMAModel(
        num_layers=NUM_LAYERS_BENCH, dim=DIM, num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM, attn_mode=attn_mode,
    ).to(device="cuda", dtype=DTYPE).eval()
    return model


def run_single(name, attn_mode, batch_size, use_compile=False):
    """Run prefill + decode for one implementation at given batch size."""
    print(f"\n  [{name}] batch={batch_size}", flush=True)
    model = create_model(attn_mode)

    if use_compile:
        print(f"    Compiling model...", flush=True)
        model = torch.compile(model, mode="reduce-overhead")
        dummy = torch.randn(batch_size, PREFILL_SEQ_LEN, DIM, device="cuda", dtype=DTYPE)
        for _ in range(COMPILE_WARMUP):
            with torch.no_grad():
                model.prefill(dummy)
        torch.cuda.synchronize()
        del dummy

        dummy_pf = torch.randn(batch_size, DECODE_CONTEXT_LEN, DIM, device="cuda", dtype=DTYPE)
        with torch.no_grad():
            _, kv = model.prefill(dummy_pf)
        dummy_dc = torch.randn(batch_size, 1, DIM, device="cuda", dtype=DTYPE)
        for _ in range(COMPILE_WARMUP):
            with torch.no_grad():
                model.decode_step(dummy_dc, kv)
        torch.cuda.synchronize()
        del dummy_pf, dummy_dc, kv
        torch.cuda.empty_cache()

    result = {"name": name, "attn_mode": attn_mode, "batch_size": batch_size, "compiled": use_compile}

    # Prefill
    try:
        print(f"    Prefill (B={batch_size}, S={PREFILL_SEQ_LEN})...", end="", flush=True)
        pf = benchmark_prefill(model, batch_size, PREFILL_SEQ_LEN, WARMUP_ITERS, PREFILL_ITERS)
        result["prefill"] = pf
        if pf["status"] == "ok":
            print(f" {pf['time_ms']:.3f} ms, {pf['tokens_per_sec']:.0f} tok/s, {pf['tflops']:.1f} TF/s", flush=True)
    except Exception as e:
        result["prefill"] = {"status": "error", "error": str(e)[:200]}
        print(f" ERROR: {str(e)[:80]}", flush=True)

    torch.cuda.empty_cache()

    # Decode
    try:
        print(f"    Decode  (B={batch_size}, ctx={DECODE_CONTEXT_LEN})...", end="", flush=True)
        dc = benchmark_decode(model, batch_size, DECODE_CONTEXT_LEN, WARMUP_ITERS, DECODE_ITERS)
        result["decode"] = dc
        if dc["status"] == "ok":
            print(f" {dc['ms_per_token']:.4f} ms/tok, {dc['tokens_per_sec']:.0f} tok/s, {dc['tflops']:.2f} TF/s", flush=True)
    except Exception as e:
        result["decode"] = {"status": "error", "error": str(e)[:200]}
        print(f" ERROR: {str(e)[:80]}", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)
    return result


def run_benchmarks():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = get_device_config()

    print("=" * 80, flush=True)
    print("ADDITIONAL E2E INFERENCE BENCHMARKS (batch=8, batch=32)", flush=True)
    print("=" * 80, flush=True)
    print(f"Device: {config['hardware']['gpu_name']}", flush=True)
    print(f"Model: LLaMA-7B config ({NUM_LAYERS_BENCH} layers, dim={DIM})", flush=True)

    implementations = [
        ("Eager (Naive)",        "naive", False),
        ("Eager (SDPA)",         "sdpa",  False),
        ("Eager (FA2)",          "flash", False),
        ("torch.compile (SDPA)", "sdpa",  True),
    ]

    all_results = {}

    for batch_size in BATCH_SIZES:
        print(f"\n\n{'='*80}")
        print(f"BATCH SIZE = {batch_size}")
        print(f"{'='*80}")

        batch_results = []
        for name, attn_mode, use_compile in implementations:
            try:
                result = run_single(name, attn_mode, batch_size, use_compile)
                batch_results.append(result)
            except Exception as e:
                print(f"\n  [{name}] FAILED: {e}", flush=True)
                batch_results.append({
                    "name": name, "batch_size": batch_size,
                    "status": "failed", "error": str(e)[:200]
                })

        all_results[f"batch_{batch_size}"] = batch_results

        # Print summary table
        print(f"\n  PREFILL SUMMARY (batch={batch_size}, seq={PREFILL_SEQ_LEN}):")
        print(f"  {'Implementation':<26} {'Time (ms)':>10} {'Tokens/s':>12} {'TFLOP/s':>10}")
        print("  " + "-" * 62)
        for r in batch_results:
            p = r.get("prefill", {})
            if p.get("status") == "ok":
                print(f"  {r['name']:<26} {p['time_ms']:>10.3f} {p['tokens_per_sec']:>12.0f} {p['tflops']:>10.1f}")
            else:
                print(f"  {r['name']:<26} {'ERROR':>10}")

        print(f"\n  DECODE SUMMARY (batch={batch_size}, ctx={DECODE_CONTEXT_LEN}):")
        print(f"  {'Implementation':<26} {'ms/tok':>10} {'Tokens/s':>12} {'TFLOP/s':>10}")
        print("  " + "-" * 62)
        for r in batch_results:
            d = r.get("decode", {})
            if d.get("status") == "ok":
                print(f"  {r['name']:<26} {d['ms_per_token']:>10.4f} {d['tokens_per_sec']:>12.0f} {d['tflops']:>10.2f}")
            else:
                print(f"  {r['name']:<26} {'ERROR':>10}")

    # Save all results
    output = {
        "title": f"E2E Inference Benchmark (Multi-Batch) on {config['hardware']['gpu_name']}",
        "device_config": config,
        "benchmark_config": {
            "model": "LLaMA-7B architecture",
            "num_layers": NUM_LAYERS_BENCH,
            "dim": DIM,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "ffn_dim": FFN_DIM,
            "dtype": str(DTYPE),
            "batch_sizes": BATCH_SIZES,
            "prefill_seq_len": PREFILL_SEQ_LEN,
            "decode_context_len": DECODE_CONTEXT_LEN,
        },
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = RESULTS_DIR / "e2e_inference_batch_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    run_benchmarks()
