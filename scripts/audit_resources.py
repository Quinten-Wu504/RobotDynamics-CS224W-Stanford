import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil


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
    }
    print(json.dumps(report, indent=2))
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
        check=True,
    )


if __name__ == "__main__":
    main()
