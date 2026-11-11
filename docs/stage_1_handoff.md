# Environment setup

The full stack was verified on AutoDL with Ubuntu 22.04 and Python 3.10.8. The [check record](../reports/stage1/autodl/README.md) includes the source commit and outputs. The older `stage-1-environment` tag records the initial Windows handoff, before Linux validation.

## Python packages

Use Python 3.10 and an isolated environment. Git, CMake 3.x, Make, and a C/C++ compiler are required for the source dependencies. The tested bootstrap sequence was:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements/build.txt -r requirements/base.txt
.venv/bin/python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu121/
.venv/bin/python -m pip install -r requirements/simulation.txt
.venv/bin/python -m pip check
```

The official CUDA 12.1 index remains in `requirements/torch.txt`; NJU was used because the direct route was slow. [linux-cu121.lock.txt](../requirements/linux-cu121.lock.txt) records all installed versions and source commits. To reproduce those exact versions, install the matching torch wheels first, then use `python -m pip install -r requirements/linux-cu121.lock.txt`. This lock was checked against the working installation; a second clean installation has not been run.

On AutoDL, `source /etc/network_turbo` was needed for GitHub. Keep the package mirror hosts out of that proxy:

```bash
source /etc/network_turbo >/dev/null
export no_proxy=mirrors.aliyun.com,pypi.org,files.pythonhosted.org,mirrors.nju.edu.cn
export NO_PROXY="$no_proxy"
```

The project used these cache locations:

```bash
mkdir -p /root/autodl-fs/RobotDynamics-CS224W-Stanford-cache/tmp
export PIP_CACHE_DIR=/root/autodl-fs/RobotDynamics-CS224W-Stanford-cache/pip
export TMPDIR=/root/autodl-fs/RobotDynamics-CS224W-Stanford-cache/tmp
```

## EGL runtime on the tested Ubuntu image

The image had the NVIDIA vendor library and GL dispatch runtime, but not `libEGL.so.1`. The following adds the missing loader inside the project environment. It does not upgrade system graphics packages.

```bash
apt-get update
mkdir -p outputs/stage1_autodl/native-package
(cd outputs/stage1_autodl/native-package && apt-get download libegl1=1.4.0-1)
dpkg-deb -x outputs/stage1_autodl/native-package/libegl1_1.4.0-1_amd64.deb .venv/native
export LD_LIBRARY_PATH="$PWD/.venv/native/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

These commands were tested on Ubuntu 22.04 amd64. Other hosts need their own compatible EGL runtime. Repeat the `LD_LIBRARY_PATH` export in each new shell when using this project-local loader.

## Checks

From the project root:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check robot_dynamics scripts tests
.venv/bin/python -m robot_dynamics.config
.venv/bin/python scripts/audit_resources.py
.venv/bin/python scripts/smoke_environment.py --device cpu --output outputs/cpu_check.json
MUJOCO_GL=egl .venv/bin/python scripts/smoke_environment.py --device cuda --simulation --output outputs/cuda_check.json
```

The smoke check imports packages and Square classes, steps a small MuJoCo model, checks gradients, and round-trips an in-memory HDF5 array. It does not render images or replay dataset trajectories. At stage 1, `--require-audited` intentionally failed. The stage 2 data audit has since verified those fields, and the current configuration passes that check.

Existing result files are not overwritten; choose a new output filename for repeat runs. There is no model checkpoint to resume. Dataset download and extraction are now described in [data processing](data_processing.md); training and prediction evaluation are not implemented yet.

## Export

```bash
mkdir -p outputs/stage1
.venv/bin/python -m pip freeze --all > outputs/stage1/linux.lock.txt
git rev-parse HEAD > outputs/stage1/commit.txt
tar -czf stage1-results.tar.gz -C outputs stage1
```

Include the smoke JSON files and relevant logs when sharing results. Preserve failures as well as successful runs. The existing AutoDL outputs are under `outputs/stage1_autodl`; selected results are committed under `reports/stage1/autodl`.
