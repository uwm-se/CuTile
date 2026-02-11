#!/bin/bash
# ============================================================================
# Run All Additional Benchmarks for B200
# ============================================================================
# This script runs:
#   1. Hardware specification collection
#   2. Rectangular GEMM (BF16) + FP16 GEMM
#   3. Extended Attention (seq=8192) + FP16 Attention
#   4. E2E Inference with batch=8 and batch=32
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Environment setup for CuTile
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/compat:/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH}"
export PATH="/usr/local/cuda-13.1/bin:${PATH}"

echo "============================================================"
echo "ADDITIONAL BENCHMARKS — B200"
echo "Started: $(date)"
echo "============================================================"

# 1. Collect hardware specification
echo ""
echo ">>> Step 1/4: Collecting hardware specification..."
cd "$SCRIPT_DIR"
python3 collect_hw_spec.py

# 2. Additional GEMM benchmarks
echo ""
echo ">>> Step 2/4: Running additional GEMM benchmarks..."
cd "$SCRIPT_DIR/GEMM"
# Build CUDA extensions if needed
if [ ! -f "wmma_kernel*.so" ] 2>/dev/null && [ -f "setup.py" ]; then
    echo "  Building CUDA extensions..."
    CUDA_HOME=/usr/local/cuda-12.8 python3 setup.py install 2>/dev/null || echo "  (CUDA ext build skipped or already done)"
fi
python3 run_additional.py

# 3. Additional Attention benchmarks
echo ""
echo ">>> Step 3/4: Running additional Attention benchmarks..."
cd "$SCRIPT_DIR/Attention"
python3 run_additional.py

# 4. Additional E2E Inference benchmarks
echo ""
echo ">>> Step 4/4: Running additional E2E Inference benchmarks..."
cd "$SCRIPT_DIR/E2E_Inference"
python3 run_additional.py

echo ""
echo "============================================================"
echo "ALL ADDITIONAL BENCHMARKS COMPLETE"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "New result files:"
RESULTS_DIR="$SCRIPT_DIR/../results_B200"
ls -la "$RESULTS_DIR"/hardware_specification.json 2>/dev/null
ls -la "$RESULTS_DIR"/GEMM/gemm_rectangular_results.json 2>/dev/null
ls -la "$RESULTS_DIR"/GEMM/gemm_fp16_results.json 2>/dev/null
ls -la "$RESULTS_DIR"/Attention/attention_extended_seq_results.json 2>/dev/null
ls -la "$RESULTS_DIR"/Attention/attention_fp16_results.json 2>/dev/null
ls -la "$RESULTS_DIR"/E2E_Inference/e2e_inference_batch_results.json 2>/dev/null
