#!/bin/bash
# =============================================================================
# NVIDIA RTX PRO 6000 Blackwell Server Edition — Full Benchmark Setup & Run
# =============================================================================
# This script reproduces ALL benchmarks on an NVIDIA RTX PRO 6000 Blackwell
# Server Edition (sm_120, 188 SMs, 96 GB VRAM).
#
# Tested Environment:
#   - RunPod pod with RTX PRO 6000 Blackwell Server Edition
#   - Driver: 570.195.03 (CUDA 12.8)
#   - Base CUDA Toolkit: 12.8 (pre-installed, used by PyTorch)
#   - Python: 3.12.3
#   - OS: Ubuntu 24.04 LTS
#
# IMPORTANT NOTES for RTX PRO 6000 (sm_120):
#   - CuTile requires CUDA Toolkit 13.1 for the tileiras compiler
#   - WMMA kernel uses NUM_STAGES=2 (48KB shared mem limit on sm_120)
#   - CuTile GEMM uses num_ctas=ByTarget(sm_100=2, sm_120=4) — 
#     sm_120 was manually tuned (see paper for details)
#   - CUDA extensions must be built with CUDA_HOME=/usr/local/cuda-12.8
#     to match PyTorch's CUDA version
#
# What this script does:
#   1. Installs CUDA Toolkit 13.1 (provides tileiras compiler for CuTile)
#   2. Installs Python packages (cuda-tile, flash-attn, triton, etc.)
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
echo "RTX PRO 6000 Benchmark — Full Setup & Run"
echo "Started: $(date)"
echo "================================================================"

# ==========================================================================
# STEP 1: Install CUDA Toolkit 13.1 (required for CuTile's tileiras compiler)
# ==========================================================================
echo ""
echo "[1/7] Installing CUDA Toolkit 13.1 (for tileiras compiler)..."
echo "  CuTile v1.1.0 requires tileiras from CUDA Toolkit 13.1+"
echo "  See: https://docs.nvidia.com/cuda/cutile-python/quickstart.html"

# Add NVIDIA CUDA repo if not present
if [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]; then
    echo "  Adding NVIDIA CUDA repository..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb 2>/dev/null || true
fi

apt-get update -qq 2>/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-13-1 2>&1 | tail -3
echo "  ✅ CUDA Toolkit 13.1 installed"

# Verify tileiras
if [ -f /usr/local/cuda-13.1/bin/tileiras ]; then
    echo "  ✅ tileiras found: $(/usr/local/cuda-13.1/bin/tileiras --version 2>&1 | head -1)"
else
    echo "  ❌ ERROR: tileiras not found at /usr/local/cuda-13.1/bin/tileiras"
    exit 1
fi

# ==========================================================================
# STEP 2: Set environment variables
# ==========================================================================
echo ""
echo "[2/7] Setting environment variables..."

# CUDA 13.1 for tileiras (CuTile runtime)
export PATH=/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}

echo "  PATH includes:       /usr/local/cuda-13.1/bin"
echo "  LD_LIBRARY_PATH:     /usr/local/cuda-13.1/compat:..."
echo "  tileiras:            $(which tileiras)"

# ==========================================================================
# STEP 3: Install Python packages
# ==========================================================================
echo ""
echo "[3/7] Installing Python packages..."

# cuda-tile: NVIDIA's tile-based GPU programming DSL
pip install --break-system-packages -q cuda-tile 2>&1 | tail -2
echo "  ✅ cuda-tile $(python3 -c 'import cuda.tile; print(cuda.tile.__version__)')"

# flash-attn: FlashAttention-2 (needs --no-build-isolation because torch must be present)
pip install --break-system-packages --no-build-isolation -q flash-attn 2>&1 | tail -2
echo "  ✅ flash-attn $(python3 -c 'import flash_attn; print(flash_attn.__version__)')"

# Verify all packages
echo ""
echo "  Package versions:"
python3 -c "
import torch, triton, flash_attn, cuda.tile as ct
print(f'    PyTorch:     {torch.__version__}')
print(f'    CUDA:        {torch.version.cuda}')
print(f'    Triton:      {triton.__version__}')
print(f'    flash-attn:  {flash_attn.__version__}')
print(f'    cuda-tile:   {ct.__version__}')
print(f'    GPU:         {torch.cuda.get_device_name(0)}')
print(f'    SM:          {torch.cuda.get_device_capability(0)}')
"

# ==========================================================================
# STEP 4: Build CUDA C++ extensions (WMMA + Raw SIMT kernels)
# ==========================================================================
echo ""
echo "[4/7] Building CUDA C++ extensions..."
echo "  Using CUDA_HOME=/usr/local/cuda-12.8 (must match PyTorch's CUDA 12.8)"
echo "  NOTE: WMMA kernel uses NUM_STAGES=2 for sm_120 (48KB shared mem limit)"

cd "$SCRIPT_DIR/GEMM"
CUDA_HOME=/usr/local/cuda-12.8 python3 setup.py install --user 2>&1 | tail -5

# Restore CUDA 13.1 in PATH for CuTile runtime
export PATH=/usr/local/cuda-13.1/bin:$PATH

# Verify extensions loaded
python3 -c "
import wmma_kernel, raw_simt_kernel
print('  ✅ WMMA kernel:     loaded')
print('  ✅ Raw SIMT kernel: loaded')
"

# ==========================================================================
# STEP 5: Run main benchmarks
# ==========================================================================
echo ""
echo "[5/7] Running main benchmarks..."

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
# STEP 6: Run additional benchmarks
# ==========================================================================
echo ""
echo "[6/7] Running additional benchmarks..."

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
# STEP 7: Collect hardware specification
# ==========================================================================
echo ""
echo "[7/7] Collecting hardware specification..."
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
echo "CuTile:   $(python3 -c 'import cuda.tile; print(cuda.tile.__version__)' 2>/dev/null || echo 'N/A')"
echo "tileiras: $(tileiras --version 2>&1 | head -1)"
echo ""
echo "Results saved to: $SCRIPT_DIR/../results_RTX_PRO_6000/"
echo ""
RESULTS_DIR="$SCRIPT_DIR/../results_RTX_PRO_6000"
echo "Result files:"
find "$RESULTS_DIR" -name "*.json" -type f | sort | while read f; do
    echo "  ✅ $f"
done
