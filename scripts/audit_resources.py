import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil


def container_limits(root=Path("/sys/fs/cgroup")):
    limits = {}
    for name in ("memory.max", "memory.current"):
        path = root / name
        if path.exists():
            value = path.read_text().strip()
            limits[name] = None if value == "max" else int(value)
    quota_path = root / "cpu.max"
    if quota_path.exists():
        quota, period = quota_path.read_text().split()
        limits["cpu_quota_cores"] = None if quota == "max" else int(quota) / int(period)
    return limits


def main():
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(Path.cwd())
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": os.cpu_count(),
        "ram_total_gib": round(memory.total / 2**30, 2),
        "ram_available_gib": round(memory.available / 2**30, 2),
        "project_disk_total_gib": round(disk.total / 2**30, 2),
        "project_disk_free_gib": round(disk.free / 2**30, 2),
        "cgroup_v2": container_limits(),
    }
    print(json.dumps(report, indent=2))
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
        check=True,
    )


if __name__ == "__main__":
    main()
