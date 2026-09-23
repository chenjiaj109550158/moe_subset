import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
out = parser.parse_args().run_dir
out.mkdir(parents=True, exist_ok=True)


def command(args):
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {"command": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as e:
        return {"command": args, "error": str(e)}


mem = {
    line.split(":")[0]: int(line.split()[1]) * 1024
    for line in Path("/proc/meminfo").read_text().splitlines()
}
info = {
    "os": platform.platform(),
    "arch": platform.machine(),
    "cpu": command(["lscpu"]),
    "ram_total_bytes": mem["MemTotal"],
    "ram_available_bytes": mem["MemAvailable"],
    "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
    "cgroup_memory_current": Path("/sys/fs/cgroup/memory.current").read_text().strip(),
    "disk_free_bytes": shutil.disk_usage(".").free,
    "gpu": command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,compute_cap,memory.total,memory.free,driver_version,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ),
    "gpu_activity": command(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv"]
    ),
    "topology": command(["nvidia-smi", "topo", "-m"]),
    "nvcc": command(["nvcc", "--version"]),
    "torch": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "visible_gpu_count": torch.cuda.device_count(),
    "capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
    "pinned_allocation": torch.empty(1024, pin_memory=True).is_pinned(),
    "memlock_limits": resource.getrlimit(resource.RLIMIT_MEMLOCK),
    "environment_allowlist": {
        k: os.environ.get(k)
        for k in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "OMP_NUM_THREADS", "HF_HOME")
    },
}
(out / "hardware.json").write_text(json.dumps(info, indent=2))
# Conservative, implementable streaming envelope (no second complete checkpoint copy).
expert = 48 * 128 * 3 * 2048 * 768 * 2
slotbytes = expert * 32 // 128
dense = 61083502592 - expert
r = {
    "slots_per_layer": 32,
    "window_tokens": 8,
    "resource_only_fallback_used": False,
    "expert_host_bytes": expert,
    "host_peak_estimate_bytes": expert + 4 * 2**30 + 8 * 2**30 + 4 * 2**30,
    "gpu_envelope_bytes": {
        "dense_estimate": dense,
        "expert_slots": slotbytes,
        "kv_cap": 2 * 2**30,
        "workspaces_pseudo_allocator": 8 * 2**30,
        "headroom": 10 * 2**30,
    },
    "host_available_bytes": mem["MemAvailable"],
    "model_download_ready": True,
    "cuda_smoke_ready": True,
    "real_model_ready": False,
    "reason": "awaiting checkpoint SHA-256 verification and loader/backend correctness",
    "host_mode_planned": "full_pinned_cpu",
}
(out / "resource_resolution.json").write_text(json.dumps(r, indent=2))
(out / "correctness_protocol.json").write_text(
    json.dumps(
        {
            "protocol": "restart_v1_correctness_revision2",
            "revision": 2,
            "full_prefix_cross_implementation": {
                "max_nrmse": 0.03,
                "min_cosine": 0.999,
                "finite": True,
            },
            "same_backend_sync_async_requires_bitwise": True,
            "calibration_source": "docs/restart/numerical_protocol_v2.md",
            "frozen_before_candidate_outputs": True,
            "fp32": {"atol": 1e-5, "rtol": 1e-4, "tf32": False},
            "bf16": {"finite": True, "max_nrmse": 0.01, "min_cosine": 0.999},
            "router": "native float32 softmax then topk normalize then cast BF16",
            "accumulation": "BF16 expert projection/output; FP32 weighted combine and streaming group accumulation",  # noqa: E501
            "attention": "sdpa",
            "zero_reference_requires_exact_zero": True,
        },
        indent=2,
    )
)
print(
    json.dumps(
        {
            "gpu": info["gpu"],
            "host_estimate_gib": r["host_peak_estimate_bytes"] / 2**30,
            "gpu_estimate_gib": sum(r["gpu_envelope_bytes"].values()) / 2**30,
        },
        indent=2,
    )
)
