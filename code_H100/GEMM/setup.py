"""Build Raw SIMT and WMMA CUDA extensions with auto-detected GPU architecture."""
from setuptools import setup
from torch.utils import cpp_extension
import subprocess
import sys


def get_cuda_arch():
    """Auto-detect GPU compute capability for nvcc -arch flag."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            sm = f"sm_{props.major}{props.minor}"
            print(f"[setup.py] Detected GPU: {torch.cuda.get_device_name(0)} -> {sm}")
            return sm
    except Exception:
        pass

    # Fallback: try nvidia-smi
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            cc = r.stdout.strip().split("\n")[0].strip()
            major, minor = cc.split(".")
            sm = f"sm_{major}{minor}"
            print(f"[setup.py] Detected via nvidia-smi: {sm}")
            return sm
    except Exception:
        pass

    # Default fallback
    print("[setup.py] WARNING: Could not detect GPU arch, defaulting to sm_90")
    return "sm_90"


CUDA_ARCH = get_cuda_arch()

# Common nvcc flags
NVCC_FLAGS = ["-O3", "--use_fast_math", f"-arch={CUDA_ARCH}", "-std=c++17"]
CXX_FLAGS = ["-O3", "-std=c++17"]

ext_modules = [
    cpp_extension.CUDAExtension(
        name="raw_simt_kernel",
        sources=["raw_simt_kernel.cu"],
        extra_compile_args={
            "nvcc": NVCC_FLAGS,
            "cxx": CXX_FLAGS,
        },
    ),
    cpp_extension.CUDAExtension(
        name="wmma_kernel",
        sources=["wmma_kernel.cu"],
        extra_compile_args={
            "nvcc": NVCC_FLAGS,
            "cxx": CXX_FLAGS,
        },
    ),
]

setup(
    name="gemm_benchmark_ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
