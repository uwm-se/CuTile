#!/bin/bash
# =============================================================================
# NVIDIA H100 NVL — Full Benchmark Setup & Run Script
# =============================================================================
# This script reproduces ALL benchmarks on an NVIDIA H100 NVL (Hopper, sm_90).
#
# NOTE: CuTile is NOT available on H100 (requires Blackwell sm_100/sm_120).
# This script runs: cuBLAS, Triton, WMMA, Raw SIMT, FlashAttention-2,
# cuDNN SDPA, Triton Attention, Naive Attention, E2E Inference, and LOC.
#
# Tested Environment:
#   - Azure VM with NVIDIA H100 NVL
#   - Driver: 580.105.08
#   - CUDA Toolkit: 12.6 (used by PyTorch)
#   - Python: 3.12.3
#   - OS: Ubuntu 24.04.2 LTS
#
# What this script does:
#   1. Verifies GPU and environment
#   2. Installs Python packages (if missing)
#   3. Builds CUDA C++ extensions (WMMA + Raw SIMT kernels)
#   4. Runs all benchmarks (GEMM, Attention, E2E, LOC)
#   5. Runs additional benchmarks (rectangular GEMM, FP16, batch scaling)
#   6. Collects hardware specification
#
# Usage:
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh
#
# Time estimate: ~15-20 minutes total
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo "NVIDIA H100 NVL Benchmark — Full Setup & Run"
echo "Started: $(date)"
echo "================================================================"

# ==========================================================================
# STEP 1: Verify GPU and environment
# ==========================================================================
echo ""
echo "[1/6] Verifying GPU and environment..."

python3 -c "
import torch
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f'  GPU:         {name}')
print(f'  SM:          sm_{cap[0]}{cap[1]}')
print(f'  PyTorch:     {torch.__version__}')
print(f'  CUDA:        {torch.version.cuda}')
assert cap[0] == 9 and cap[1] == 0, f'Expected sm_90 (H100), got sm_{cap[0]}{cap[1]}'
print('  ✅ H100 NVL confirmed (sm_90, Hopper)')
print('  ⚠️  CuTile NOT available (requires Blackwell sm_100/sm_120)')
"

# ==========================================================================
# STEP 2: Install Python packages (if missing)
# ==========================================================================
echo ""
echo "[2/6] Checking Python packages..."

# Check if key packages exist
python3 -c "import triton; print(f'  ✅ triton {triton.__version__}')" 2>/dev/null || {
    echo "  Installing triton..."
    pip install --user -q triton
}

python3 -c "import flash_attn; print(f'  ✅ flash-attn {flash_attn.__version__}')" 2>/dev/null || {
    echo "  Installing flash-attn (this may take several minutes)..."
    pip install --user --no-build-isolation -q flash-attn
}

python3 -c "import numpy; print(f'  ✅ numpy {numpy.__version__}')" 2>/dev/null || {
    echo "  Installing numpy..."
    pip install --user -q numpy
}

# Verify all packages
python3 -c "
import torch, triton, flash_attn
print(f'  Package versions:')
print(f'    PyTorch:     {torch.__version__}')
print(f'    CUDA:        {torch.version.cuda}')
print(f'    Triton:      {triton.__version__}')
print(f'    flash-attn:  {flash_attn.__version__}')
print(f'    GPU:         {torch.cuda.get_device_name(0)}')
"

# ==========================================================================
# STEP 3: Build CUDA C++ extensions (WMMA + Raw SIMT kernels)
# ==========================================================================
echo ""
echo "[3/6] Building CUDA C++ extensions..."

# Find CUDA home that matches PyTorch's CUDA version
PYTORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)")
echo "  PyTorch CUDA version: $PYTORCH_CUDA"

# Try to find matching CUDA toolkit
CUDA_HOME_PATH=""
for candidate in "/usr/local/cuda-${PYTORCH_CUDA}" "/usr/local/cuda"; do
    if [ -f "$candidate/bin/nvcc" ]; then
        CUDA_HOME_PATH="$candidate"
        break
    fi
done

if [ -z "$CUDA_HOME_PATH" ]; then
    echo "  ❌ ERROR: Could not find CUDA toolkit matching PyTorch CUDA $PYTORCH_CUDA"
    exit 1
fi

echo "  Using CUDA_HOME=$CUDA_HOME_PATH"

cd "$SCRIPT_DIR/GEMM"
CUDA_HOME="$CUDA_HOME_PATH" python3 setup.py install --user 2>&1 | tail -5

# Verify extensions loaded
python3 -c "
import wmma_kernel, raw_simt_kernel
print('  ✅ WMMA kernel:     loaded')
print('  ✅ Raw SIMT kernel: loaded')
"

# ==========================================================================
# STEP 4: Run main benchmarks
# ==========================================================================
echo ""
echo "[4/6] Running main benchmarks..."

echo ""
echo "  --- GEMM (BF16 square: 4096-16384) ---"
cd "$SCRIPT_DIR"
python3 GEMM/run_benchmark.py

echo ""
echo "  --- Attention (BF16, seq 512-4096, batch=8, causal) ---"
python3 Attention/run_benchmark.py

echo ""
echo "  --- E2E Inference (LLaMA-7B, 4 layers, batch=1) ---"
python3 E2E_Inference/run_benchmark.py

echo ""
echo "  --- Lines of Code Comparison ---"
python3 LOC_Comparison/generate_table.py

# ==========================================================================
# STEP 5: Run additional benchmarks
# ==========================================================================
echo ""
echo "[5/6] Running additional benchmarks..."

echo ""
echo "  --- GEMM: Rectangular (FFN shapes) + FP16 ---"
python3 GEMM/run_additional.py

echo ""
echo "  --- Attention: seq=8192 + FP16 ---"
python3 Attention/run_additional.py

echo ""
echo "  --- E2E Inference: batch=8 and batch=32 ---"
python3 E2E_Inference/run_additional.py

# ==========================================================================
# STEP 6: Collect hardware specification
# ==========================================================================
echo ""
echo "[6/6] Collecting hardware specification..."
python3 collect_hw_spec.py

# ==========================================================================
# SUMMARY
# ==========================================================================
echo ""
echo "================================================================"
echo "ALL BENCHMARKS COMPLETE!"
echo "Finished: $(date)"
echo "================================================================"
echo ""
echo "Hardware: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Driver:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo "CUDA:     $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "PyTorch:  $(python3 -c 'import torch; print(torch.__version__)')"
echo "CuTile:   NOT AVAILABLE (requires Blackwell)"
echo ""
RESULTS_DIR="$SCRIPT_DIR/../results_H100"
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Result files:"
find "$RESULTS_DIR" -name "*.json" -type f | sort | while read f; do
    echo "  ✅ $f"
done
