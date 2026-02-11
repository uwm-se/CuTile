#!/usr/bin/env python3
"""
Comprehensive Hardware Specification & Library Version Collector.
Saves full system details for reproducibility in the research paper.
"""

import json
import subprocess
import platform
import os
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results_H100"


def run_cmd(cmd, shell=False):
    """Run a command and return stdout or 'N/A'."""
    try:
        r = subprocess.run(cmd if isinstance(cmd, list) else cmd,
                           capture_output=True, text=True, timeout=30,
                           shell=shell)
        return r.stdout.strip() if r.returncode == 0 else f"error: {r.stderr.strip()[:100]}"
    except Exception as e:
        return f"N/A ({str(e)[:50]})"


def collect_gpu_details():
    """Collect detailed GPU specifications."""
    gpu = {}

    # Basic info via nvidia-smi
    fields = [
        "name", "driver_version", "pci.bus_id", "compute_cap",
        "memory.total", "memory.free", "memory.used",
        "clocks.max.sm", "clocks.max.mem", "clocks.current.sm", "clocks.current.mem",
        "power.limit", "power.max_limit", "power.draw",
        "temperature.gpu", "fan.speed",
        "compute_mode", "persistence_mode",
        "ecc.mode.current", "pstate",
        "gpu_serial", "uuid",
    ]
    query = ",".join(fields)
    raw = run_cmd(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if raw and not raw.startswith("N/A"):
        values = [v.strip() for v in raw.split(",")]
        for field, val in zip(fields, values):
            gpu[field.replace(".", "_")] = val

    # Detailed nvidia-smi -q sections
    full_smi = run_cmd(["nvidia-smi", "-q"])
    if full_smi and not full_smi.startswith("N/A"):
        # Parse specific sections
        for line in full_smi.split("\n"):
            line = line.strip()
            if "CUDA Version" in line and ":" in line:
                gpu["nvidia_smi_cuda_version"] = line.split(":")[-1].strip()
            elif "Product Architecture" in line and ":" in line:
                gpu["product_architecture"] = line.split(":")[-1].strip()
            elif "CUDA Capability" in line and ":" in line:
                gpu["cuda_capability_from_smi"] = line.split(":")[-1].strip()

    # nvidia-smi -q -d MEMORY
    mem_info = run_cmd(["nvidia-smi", "-q", "-d", "MEMORY"])
    gpu["memory_detail"] = mem_info[:500] if mem_info else "N/A"

    return gpu


def collect_cuda_details():
    """Collect CUDA toolkit and driver details."""
    cuda = {}

    # nvcc version
    cuda["nvcc_version"] = run_cmd(["nvcc", "--version"])

    # CUDA directories
    cuda_dirs = []
    for path in sorted(Path("/usr/local").glob("cuda-*")):
        if path.is_dir():
            cuda_dirs.append(str(path))
    cuda["installed_cuda_dirs"] = cuda_dirs

    # Current CUDA_HOME
    cuda["CUDA_HOME"] = os.environ.get("CUDA_HOME", "not set")
    cuda["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "not set")
    cuda["PATH_cuda_bins"] = [p for p in os.environ.get("PATH", "").split(":") if "cuda" in p]

    # cuda-compat packages
    compat = run_cmd("dpkg -l | grep cuda-compat", shell=True)
    cuda["cuda_compat_packages"] = compat if compat else "none found"

    return cuda


def collect_python_packages():
    """Collect Python package versions relevant to the benchmarks."""
    packages = {}

    # Core packages
    import torch
    packages["torch"] = torch.__version__
    packages["torch_cuda_version"] = torch.version.cuda or "N/A"
    packages["torch_cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    packages["torch_cuda_available"] = torch.cuda.is_available()

    # Compute capability from PyTorch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        packages["torch_compute_cap"] = f"{props.major}.{props.minor}"
        packages["torch_device_name"] = torch.cuda.get_device_name(0)
        packages["torch_num_sms"] = props.multi_processor_count
        packages["torch_global_mem_bytes"] = props.total_memory
        packages["torch_shared_mem_per_sm"] = props.shared_memory_per_multiprocessor
        packages["torch_max_threads_per_sm"] = props.max_threads_per_multi_processor

    try:
        import triton
        packages["triton"] = triton.__version__
    except ImportError:
        packages["triton"] = "not installed"

    try:
        import flash_attn
        packages["flash_attn"] = flash_attn.__version__
    except ImportError:
        packages["flash_attn"] = "not installed"

    try:
        import cuda.tile as ct
        packages["cuda_tile"] = ct.__version__
        packages["cuda_tile_path"] = str(Path(ct.__file__).parent)
    except (ImportError, AttributeError):
        packages["cuda_tile"] = "not installed"

    try:
        import numpy
        packages["numpy"] = numpy.__version__
    except ImportError:
        packages["numpy"] = "not installed"

    try:
        import cupy
        packages["cupy"] = cupy.__version__
    except ImportError:
        packages["cupy"] = "not installed"

    # pip list for all packages
    pip_list = run_cmd(["/usr/bin/env", "python3", "-m", "pip", "list", "--format=json"])
    try:
        packages["all_pip_packages"] = json.loads(pip_list)
    except (json.JSONDecodeError, TypeError):
        packages["all_pip_packages"] = "parse error"

    return packages


def collect_system_info():
    """Collect OS and system info."""
    system = {
        "hostname": platform.node(),
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }

    # CPU info
    cpu_info = run_cmd(["lscpu"])
    system["lscpu"] = cpu_info[:1000] if cpu_info else "N/A"

    # Memory info
    mem_info = run_cmd(["free", "-h"])
    system["memory"] = mem_info if mem_info else "N/A"

    # OS release
    os_release = run_cmd(["cat", "/etc/os-release"])
    system["os_release_file"] = os_release[:500] if os_release else "N/A"

    # Kernel
    system["kernel"] = run_cmd(["uname", "-a"])

    return system


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COLLECTING HARDWARE SPECIFICATION & LIBRARY VERSIONS")
    print("=" * 80)

    print("\n  Collecting GPU details...", flush=True)
    gpu = collect_gpu_details()

    print("  Collecting CUDA details...", flush=True)
    cuda = collect_cuda_details()

    print("  Collecting Python package versions...", flush=True)
    packages = collect_python_packages()

    print("  Collecting system info...", flush=True)
    system = collect_system_info()

    spec = {
        "collection_date": datetime.now().isoformat(),
        "purpose": "Hardware specification and software versions for benchmark reproducibility",
        "gpu": gpu,
        "cuda": cuda,
        "python_packages": packages,
        "system": system,
    }

    out_path = RESULTS_DIR / "hardware_specification.json"
    with open(out_path, "w") as f:
        json.dump(spec, f, indent=2, default=str)

    print(f"\n  Saved: {out_path}", flush=True)

    # Print summary
    print("\n" + "=" * 80)
    print("HARDWARE SPECIFICATION SUMMARY")
    print("=" * 80)
    print(f"  GPU: {gpu.get('name', 'N/A')}")
    print(f"  Architecture: {gpu.get('product_architecture', 'N/A')}")
    print(f"  Compute Cap: {gpu.get('compute_cap', 'N/A')}")
    print(f"  VRAM: {gpu.get('memory_total', 'N/A')}")
    print(f"  SM Clock: {gpu.get('clocks_max_sm', 'N/A')} MHz")
    print(f"  Mem Clock: {gpu.get('clocks_max_mem', 'N/A')} MHz")
    print(f"  Power Limit: {gpu.get('power_limit', 'N/A')} W")
    print(f"  Driver: {gpu.get('driver_version', 'N/A')}")
    print(f"  PyTorch: {packages.get('torch', 'N/A')}")
    print(f"  CUDA: {packages.get('torch_cuda_version', 'N/A')}")
    print(f"  cuDNN: {packages.get('torch_cudnn_version', 'N/A')}")
    print(f"  Triton: {packages.get('triton', 'N/A')}")
    print(f"  FlashAttention: {packages.get('flash_attn', 'N/A')}")
    print(f"  CuTile: {packages.get('cuda_tile', 'N/A')}")
    print(f"  NumPy: {packages.get('numpy', 'N/A')}")
    print(f"  OS: {system.get('os', 'N/A')} {system.get('os_release', 'N/A')}")
    print(f"  Python: {system.get('python_version', 'N/A')}")


if __name__ == "__main__":
    main()
