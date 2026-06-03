# CuTile Benchmark Suite

**Evaluating NVIDIA CuTile: Performance and Productivity Tradeoffs for GPU Kernel Programming on Blackwell Architecture**

This repository contains the complete benchmark suite for evaluating GPU kernel programming abstractions — including NVIDIA's new CuTile (`cuda.tile`) programming model — across GEMM, Fused Multi-Head Attention, End-to-End LLM Inference, and Lines-of-Code productivity metrics.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Benchmarks Overview](#benchmarks-overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Quick Start (One Command)](#quick-start-one-command)
- [Step-by-Step Replication Guide](#step-by-step-replication-guide)
- [Running Individual Benchmarks](#running-individual-benchmarks)
- [Understanding the Results](#understanding-the-results)
- [Known Issues and Fixes](#known-issues-and-fixes)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

---

## Repository Structure

```
CuTile/
├── code_RTX_PRO_6000/              # Code for RTX PRO 6000 Blackwell Server Edition (sm_120)
│   ├── GEMM/
│   │   ├── run_benchmark.py        # GEMM benchmark runner (square BF16)
│   │   ├── run_additional.py       # Additional GEMM (rectangular + FP16)
│   │   ├── gemm_cublas.py          # cuBLAS via torch.matmul
│   │   ├── gemm_cutile.py          # CuTile GEMM kernel
│   │   ├── gemm_gmma.py            # Triton GEMM kernel
│   │   ├── wmma_kernel.cu          # WMMA Tensor Core CUDA kernel (NUM_STAGES=2)
│   │   ├── raw_simt_kernel.cu      # Raw SIMT CUDA kernel
│   │   └── setup.py                # Build script for CUDA C++ extensions
│   ├── Attention/
│   │   ├── run_benchmark.py        # Attention benchmark runner (BF16, seq 512-4096)
│   │   ├── run_additional.py       # Additional Attention (seq=8192 + FP16)
│   │   ├── attn_cutile.py          # CuTile fused attention kernel
│   │   ├── attn_flash_sdpa.py      # FlashAttention-2 wrapper
│   │   ├── attn_cudnn_sdpa.py      # cuDNN SDPA wrapper
│   │   ├── attn_triton.py          # Triton attention kernel
│   │   └── attn_naive.py           # Naive unfused attention
│   ├── E2E_Inference/
│   │   ├── run_benchmark.py        # End-to-end inference benchmark (batch=1)
│   │   ├── run_additional.py       # Additional E2E (batch=8, batch=32)
│   │   └── model.py                # LLaMA-7B model definition
│   ├── LOC_Comparison/
│   │   └── generate_table.py       # Lines-of-code comparison generator
│   ├── collect_hw_spec.py          # Hardware & software specification collector
│   ├── setup_and_run.sh            # Automated setup + run ALL benchmarks
│   └── run_all_additional.sh       # Run additional benchmarks only
│
├── code_B200/                      # Code for NVIDIA B200 (sm_100)
│   ├── (same structure as above)
│   └── setup_and_run.sh            # Automated setup + run ALL benchmarks (B200)
│
├── results_RTX_PRO_6000/           # Benchmark results — RTX PRO 6000
│   ├── GEMM/
│   │   ├── gemm_results.json       # Square BF16 GEMM (4096-16384)
│   │   ├── gemm_rectangular_results.json   # Rectangular GEMM (FFN shapes)
│   │   ├── gemm_fp16_results.json          # FP16 GEMM
│   │   └── device_config.json
│   ├── Attention/
│   │   ├── attention_results.json  # BF16 Attention (seq 512-4096)
│   │   ├── attention_extended_seq_results.json  # seq=8192
│   │   ├── attention_fp16_results.json          # FP16 Attention
│   │   └── device_config.json
│   ├── E2E_Inference/
│   │   ├── e2e_inference_results.json       # Batch=1
│   │   └── e2e_inference_batch_results.json # Batch=8, 32
│   ├── LOC_Comparison/
│   │   └── loc_comparison.json
│   ├── hardware_specification.json   # Full GPU/CUDA/pip freeze snapshot
│   ├── reproducibility_config.json   # Detailed reproducibility config
│
├── results_B200/                   # Benchmark results — B200
│   ├── (same structure as results_RTX_PRO_6000/)
│   └── hardware_specification.json
│
├── requirements.txt                # Python package requirements
└── README.md                       # This file
```

> **Note:** Code directories are kept **separate per hardware** because different GPU architectures require different tuning parameters (e.g., `NUM_STAGES`, `ByTarget`, shared memory limits).

---

## Benchmarks Overview

| Benchmark | Implementations | Key Metric |
|-----------|----------------|------------|
| **GEMM** | cuBLAS, Triton, WMMA, Raw SIMT, CuTile | TFLOP/s |
| **Fused Attention** | FlashAttention-2, cuDNN SDPA, Triton, CuTile, Naive | TFLOP/s |
| **E2E Inference** | Eager (Naive), Eager (SDPA), Eager (FA2), torch.compile (SDPA) | tokens/sec, ms/token |
| **LOC Comparison** | All of the above | Lines of Code |

### GEMM
- **Square sizes:** 4096, 8192, 12288, 16384 (M=N=K)
- **Rectangular shapes:** 2048×4096×11008, 2048×11008×4096, 4096×4096×11008, 4096×11008×4096 (FFN)
- **Dtypes:** BF16, FP16
- **Iterations:** 10 warmup + 50 timed (CUDA events)

### Fused Attention
- **Config:** Batch=8, Heads=32, Head dim=128 (LLaMA-7B-like)
- **Sequence Lengths:** 512, 1024, 2048, 4096, 8192
- **Causal masking:** Yes
- **Dtypes:** BF16, FP16

### End-to-End Inference
- **Model:** LLaMA-7B architecture (4 transformer layers)
- **Dim:** 4096, FFN dim: 11008, Heads: 32
- **Prefill:** 2048 tokens, Decode: 2048-token context
- **Batch sizes:** 1, 8, 32
- **Dtype:** BF16

### LOC Comparison
- Non-blank, non-comment lines for kernel + launcher code
- Hardware-independent (code analysis only)

---

## Hardware Requirements

### Tested Hardware

| | RTX PRO 6000 Blackwell Server Edition | NVIDIA B200 |
|---|---|---|
| **Architecture** | Blackwell (sm_120) | Blackwell (sm_100) |
| **SMs** | 188 | 148 |
| **VRAM** | 96 GB | 180 GB HBM3e |
| **Max SM Clock** | 2430 MHz | 1965 MHz |
| **Max Mem Clock** | 12481 MHz | 3996 MHz |
| **TDP** | 600 W | 1000 W |
| **Driver** | 570.195.03 | 580.126.09 |
| **CPU** | AMD EPYC 9655 96-Core | AMD EPYC 9555 64-Core |
| **Platform** | RunPod | RunPod |

### CuTile GPU Compatibility

CuTile requires **Blackwell architecture** GPUs:
- ✅ RTX PRO 6000 Blackwell Server Edition (sm_120)
- ✅ NVIDIA B100 / B200 (sm_100)
- ✅ RTX 5090 (sm_100)
- ❌ H100 (sm_90 — Hopper, not supported)
- ❌ A100 (sm_80 — Ampere, not supported)
- ❌ RTX 6000 Ada (sm_89 — Ada Lovelace, not supported)

> The non-CuTile benchmarks (cuBLAS, Triton, WMMA, FlashAttention-2, etc.) run on any CUDA-capable GPU, but results will differ.

---

## Software Requirements

### Exact Tested Versions

| Package | RTX PRO 6000 | B200 |
|---------|-------------|------|
| **OS** | Ubuntu 24.04.3 LTS | Ubuntu 24.04.3 LTS |
| **Kernel** | 6.8.0-86-generic | 6.8.0-100-generic |
| **NVIDIA Driver** | 570.195.03 | 580.126.09 |
| **CUDA (PyTorch)** | 12.8 | 12.8 |
| **CUDA (tileiras)** | 13.1 | 13.1 |
| **Python** | 3.12.3 | 3.12.3 |
| **PyTorch** | 2.8.0+cu128 | 2.8.0+cu128 |
| **Triton** | 3.4.0 | 3.4.0 |
| **FlashAttention** | 2.8.3 | 2.8.3 |
| **cuda-tile** | 1.1.0 | 1.1.0 |
| **tileiras** | V13.1.115 | V13.1.80 |
| **cuDNN** | 9.10.02 | 9.10.02 |
| **NumPy** | 2.1.2 | 2.1.2 |

### System-Level Packages Required (apt)

```bash
# CuTile requires the tileiras compiler from CUDA Toolkit 13.1
# This is the MOST CRITICAL setup step — without it CuTile won't work
apt-get update
apt-get install -y cuda-toolkit-13-1

# This installs:
#   /usr/local/cuda-13.1/bin/tileiras  — the CuTile tile IR compiler
#   /usr/local/cuda-13.1/compat/       — CUDA 13.1 compat libraries
#   /usr/local/cuda-13.1/lib64/        — CUDA 13.1 runtime libraries
```

### Critical Environment Variables

```bash
# REQUIRED: CuTile runtime needs CUDA 13.1 compat libs to override the host driver
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-13.1/bin:$PATH

# Without this, CuTile will fail with:
#   ERROR: Minimum driver version required is 13.0, got 12.8
```

### Python Package Installation

```bash
# Install from requirements.txt
pip install --break-system-packages -r requirements.txt

# OR install individually:
pip install --break-system-packages cuda-tile==1.1.0
pip install --break-system-packages --no-build-isolation flash-attn==2.8.3
pip install --break-system-packages triton==3.4.0 numpy==2.1.2

# NOTE: flash-attn requires --no-build-isolation because it needs torch at build time
# NOTE: --break-system-packages is needed on Ubuntu 24.04 (PEP 668)
```

### Building CUDA C++ Extensions

```bash
cd code_<GPU>/GEMM

# CRITICAL: Must use CUDA 12.8 for building (matches PyTorch's compiled CUDA version)
# If you use CUDA 13.1, you'll get:
#   RuntimeError: The detected CUDA version (13.1) mismatches PyTorch (12.8)
CUDA_HOME=/usr/local/cuda-12.8 python3 setup.py install --user

# Then restore CUDA 13.1 for CuTile runtime:
export PATH=/usr/local/cuda-13.1/bin:$PATH
```

---

## Quick Start (One Command)

### RTX PRO 6000

```bash
cd code_RTX_PRO_6000
chmod +x setup_and_run.sh
./setup_and_run.sh
```

### B200

```bash
cd code_B200
chmod +x setup_and_run.sh
./setup_and_run.sh
```

Each `setup_and_run.sh` script will:
1. Install CUDA Toolkit 13.1 (provides `tileiras` compiler for CuTile)
2. Set `LD_LIBRARY_PATH` and `PATH` for CUDA 13.1
3. Install Python dependencies (`cuda-tile`, `flash-attn`, `triton`)
4. Build CUDA C++ extensions with CUDA 12.8 (WMMA + Raw SIMT kernels)
5. Run all main benchmarks (GEMM, Attention, E2E, LOC)
6. Run additional benchmarks (rectangular, FP16, batch scaling)
7. Collect hardware specification
8. Save results to `results_<GPU>/`

**Estimated runtime:** ~15-20 minutes total per GPU.

---

## Step-by-Step Replication Guide

This is the exact sequence of commands we ran to produce the results in this paper.

### 1. Provision a Blackwell GPU on RunPod

1. Go to [runpod.io](https://runpod.io) and create an account
2. Add your SSH public key under Account → SSH Keys
3. Deploy a **GPU Pod** with:
   - GPU: **RTX PRO 6000 Blackwell Server Edition** or **B200**
   - Template: **RunPod Pytorch 2.8.0** (includes PyTorch 2.8.0+cu128, Python 3.12, Ubuntu 24.04)
   - Disk: At least 50 GB
4. Note the SSH connection details from the pod page

### 2. Connect via SSH

```bash
# Direct TCP (recommended — supports SCP/SFTP):
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519

# Or via RunPod proxy:
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### 3. Upload the Repository

```bash
# From your LOCAL machine:
scp -P <PORT> -i ~/.ssh/id_ed25519 -r CuTile/ root@<IP>:~/
```

### 4. Install CUDA Toolkit 13.1 (for CuTile)

```bash
# ON THE POD:
apt-get update
apt-get install -y cuda-toolkit-13-1

# Verify tileiras is available:
/usr/local/cuda-13.1/bin/tileiras --version
# Expected: Cuda compilation tools, release 13.1, V13.1.115
```

**Why this is needed:** CuTile v1.1.0 compiles tile IR kernels at runtime using the `tileiras` binary. This binary is part of CUDA Toolkit 13.1+ and is **not** included in the `cuda-tile` pip package.

### 5. Set Environment Variables

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-13.1/bin:$PATH
```

**Why:** The RunPod base image ships with CUDA 12.8 driver. CuTile requires driver ≥13.0. The `cuda-compat` libraries in `/usr/local/cuda-13.1/compat/` provide forward compatibility, allowing CuTile to work on a 12.8 driver.

### 6. Install Python Packages

```bash
pip install --break-system-packages cuda-tile==1.1.0
pip install --break-system-packages --no-build-isolation flash-attn==2.8.3
pip install --break-system-packages triton==3.4.0 numpy==2.1.2

# Verify:
python3 -c "
import torch, triton, flash_attn, cuda.tile as ct
print(f'PyTorch:    {torch.__version__}')
print(f'CUDA:       {torch.version.cuda}')
print(f'Triton:     {triton.__version__}')
print(f'flash-attn: {flash_attn.__version__}')
print(f'cuda-tile:  {ct.__version__}')
print(f'GPU:        {torch.cuda.get_device_name(0)}')
"
```

### 7. Build CUDA C++ Extensions

```bash
cd ~/CuTile/code_<GPU>/GEMM

# MUST use CUDA 12.8 to match PyTorch's build:
CUDA_HOME=/usr/local/cuda-12.8 python3 setup.py install --user

# Restore CUDA 13.1 in PATH for CuTile:
export PATH=/usr/local/cuda-13.1/bin:$PATH

# Verify:
python3 -c "import wmma_kernel, raw_simt_kernel; print('Extensions loaded OK')"
```

### 8. Run All Benchmarks

```bash
cd ~/CuTile/code_<GPU>

# Main benchmarks:
python3 GEMM/run_benchmark.py
python3 Attention/run_benchmark.py
python3 E2E_Inference/run_benchmark.py
python3 LOC_Comparison/generate_table.py

# Additional benchmarks:
python3 GEMM/run_additional.py
python3 Attention/run_additional.py
python3 E2E_Inference/run_additional.py

# Hardware specification:
python3 collect_hw_spec.py
```

### 9. Download Results

```bash
# From your LOCAL machine:
scp -P <PORT> -i ~/.ssh/id_ed25519 -r root@<IP>:~/CuTile/results_<GPU>/ ./results_<GPU>/
```

---

## Running Individual Benchmarks

### GEMM Benchmark

```bash
cd ~/CuTile/code_<GPU>/GEMM
python3 run_benchmark.py          # Square BF16 (4096-16384)
python3 run_additional.py         # Rectangular + FP16
```

**Output:** `results_<GPU>/GEMM/gemm_results.json`, `gemm_rectangular_results.json`, `gemm_fp16_results.json`

### Fused Attention Benchmark

```bash
cd ~/CuTile/code_<GPU>/Attention
python3 run_benchmark.py          # BF16 (seq 512-4096)
python3 run_additional.py         # seq=8192 + FP16
```

**Output:** `results_<GPU>/Attention/attention_results.json`, `attention_extended_seq_results.json`, `attention_fp16_results.json`

### End-to-End Inference Benchmark

```bash
cd ~/CuTile/code_<GPU>/E2E_Inference
python3 run_benchmark.py          # Batch=1
python3 run_additional.py         # Batch=8, 32
```

**Output:** `results_<GPU>/E2E_Inference/e2e_inference_results.json`, `e2e_inference_batch_results.json`

### Lines of Code Comparison

```bash
cd ~/CuTile/code_<GPU>/LOC_Comparison
python3 generate_table.py
```

**Output:** `results_<GPU>/LOC_Comparison/loc_comparison.json`

---

## Understanding the Results

### Result JSON Structure

Each result file contains:
- **`device_config`** — Full hardware specs and software versions for reproducibility
- **`benchmark_config`** — Parameters used (matrix sizes, iterations, dtype, etc.)
- **`implementation_details`** — Description of each implementation
- **`results`** — Performance data per configuration

### Key Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `tflops` | TFLOP/s | Tera floating-point operations per second |
| `time_ms` | ms | Kernel execution time |
| `tokens_per_sec` | tokens/s | End-to-end inference throughput |
| `ms_per_token` | ms | Per-token decode latency |
| `kernel_loc` | lines | Lines of kernel code |
| `total_loc` | lines | Total implementation lines (kernel + launcher) |

### Sample Results

**GEMM Peak TFLOP/s (BF16, 16384×16384):**

| Implementation | RTX PRO 6000 | B200 |
|---------------|-------------|------|
| cuBLAS | 393 TFLOP/s | 698 TFLOP/s |
| Triton | 389 TFLOP/s | 616 TFLOP/s |
| CuTile | 302 TFLOP/s | 533 TFLOP/s |
| WMMA | 194 TFLOP/s | 322 TFLOP/s |
| Raw SIMT | 5.9 TFLOP/s | 7.2 TFLOP/s |

**Fused Attention Peak TFLOP/s (BF16, seq=4096):**

| Implementation | RTX PRO 6000 | B200 |
|---------------|-------------|------|
| FlashAttention-2 | 336 TFLOP/s | 401 TFLOP/s |
| CuTile | 179 TFLOP/s | 1007 TFLOP/s |
| Triton | 288 TFLOP/s | 296 TFLOP/s |
| cuDNN SDPA | 133 TFLOP/s | 286 TFLOP/s |

---

## Known Issues and Fixes

### 1. WMMA Shared Memory Overflow on sm_120

**Problem:** `ptxas error: uses too much shared data (0xde00 bytes, 0xc000 max)`

**Cause:** Original WMMA kernel uses `NUM_STAGES=3` (triple buffering), requiring 56 KB shared memory. RTX PRO 6000 (sm_120) only allows 48 KB per thread block.

**Fix:** Reduced `NUM_STAGES` from 3 to 2 in `wmma_kernel.cu`. Already applied in `code_RTX_PRO_6000/` and `code_B200/`.

### 2. CuTile Driver Version Error

**Problem:** `ERROR: Minimum driver version required is 13.0, got 12.8`

**Cause:** RunPod host driver provides CUDA 12.8, but CuTile requires CUDA ≥ 13.0.

**Fix:**
```bash
apt-get install -y cuda-toolkit-13-1
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH
```

### 3. CuTile GEMM Slow on sm_120 (Missing ByTarget)

**Problem:** CuTile GEMM only achieved 64–75 TFLOP/s (~17% of cuBLAS).

**Cause:** `@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))` did not include `sm_120`, so it fell back to default (1 CTA).

**Fix:** Updated to `@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=2))`. Already applied.

### 4. CUDA Version Mismatch When Building Extensions

**Problem:** `RuntimeError: CUDA version (13.1) mismatches PyTorch (12.8)`

**Fix:** Build with:
```bash
CUDA_HOME=/usr/local/cuda-12.8 python3 setup.py install --user
```

### 5. tileiras Not Found on B200

**Problem:** `ERROR: 'tileiras' compiler not found` (B200 pod had CUDA 12.8 only)

**Fix:**
```bash
apt-get install -y cuda-toolkit-13-1
export PATH=/usr/local/cuda-13.1/bin:$PATH
which tileiras  # Should show /usr/local/cuda-13.1/bin/tileiras
```

---

## Reproducibility

### Configuration Files

Every results directory contains:

| File | Purpose |
|------|---------|
| `hardware_specification.json` | Full GPU specs, CUDA versions, `pip freeze`, `lscpu`, `nvidia-smi` output |
| `reproducibility_config.json` | Environment setup, benchmark parameters, applied fixes, CuTile-specific config |
| `*/device_config.json` | Per-benchmark device snapshot |

### Reproducing on the Same Hardware

1. Provision the **exact GPU model** on RunPod
2. Use the matching `setup_and_run.sh` script
3. Results should match within **~5% variance** (normal GPU run-to-run variation)

### Porting to New Hardware

When porting to a new GPU:

1. **Create a new code directory:** `code_<GPU_NAME>/`
2. **Copy** all files from an existing code directory
3. **Check and modify:**
   - `wmma_kernel.cu` — `NUM_STAGES` based on shared memory limits
   - `gemm_cutile.py` — `ByTarget` decorator needs the new `sm_*` target
   - `attn_cutile.py` — `occupancy` and tuning parameters
   - All `run_benchmark.py` — `RESULTS_DIR` path
4. **Document** the exact hardware/software config

---

## Timing Methodology

All GEMM and Attention benchmarks use **CUDA events** for accurate GPU timing:

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
# ... kernel execution ...
end_event.record()
torch.cuda.synchronize()

elapsed_ms = start_event.elapsed_time(end_event)
```

- **Warmup:** 10 iterations (discarded)
- **Timed:** 50 iterations (median reported)
- **FLOP counting:** `2 × M × N × K` for GEMM; `4 × B × H × N² × D / 2` for causal attention

---

## Citation

If you use this benchmark suite in your research, please cite:

```bibtex
@inproceedings{YadavCuTile2026,
  author  = {Yadav, Divakar Kumar and Zhao, Tian and Kumar, Deepak},
  title   = {Evaluating NVIDIA CuTile: Performance and Productivity Tradeoffs for GPU Kernel Programming on Blackwell Architecture},
  booktitle = {Proceedings of the 50th IEEE Annual Computers, Software, and Applications Conference (CompSAC'26)}.
  location = {Madrid, Spain},
  month = {July},
  year    = {2026},
  url     = {https://github.com/uwm-se/CuTile}
}
```

---

## License

This project is for academic research purposes. Individual kernel implementations may reference NVIDIA sample code subject to NVIDIA's license terms.
