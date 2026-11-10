# Stage 1: Linux environment check

The Windows base environment is usable, but `egl-probe` fails to build. Stage 1 remains `BLOCKED_COMPUTE` until the full stack can be checked. See [the hardware and error record](environment.md).

## Host

Use an existing Linux x86-64 host with Python 3.10, Git, a C/C++ compiler, Make, CMake 3.x, and EGL/OpenGL runtime libraries. A reasonable starting allocation for this environment check is 16 GiB RAM and 20 GiB free disk, plus an NVIDIA GPU and driver compatible with CUDA 12.1. These are setup allowances, not measured training requirements. The smoke tensors are tiny and do not establish a batch-size or VRAM budget for future models.

No paid instance is needed or requested by these instructions. WSL is not enabled on the audited Windows machine; installing it would be a separate system change.

## Install

These commands target Linux and have **not** been run locally. Use the stage 1 commit linked in the delivery message; the tag below identifies the code/configuration snapshot used for the smoke-check handoff.

```bash
git clone https://github.com/Quinten-Wu504/RobotDynamics-CS224W-Stanford.git
cd RobotDynamics-CS224W-Stanford
git checkout stage-1-environment
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/build.txt
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/torch.txt
python -m pip install -r requirements/simulation.txt
python -m pip check
```

Install PyTorch first so the simulation dependencies retain the intended CUDA wheel. Stop if any install or dependency check fails; return its log rather than changing package versions. On headless Linux, set `MUJOCO_GL=egl` for simulation imports.

## Check and export

```bash
mkdir -p outputs/stage1
python -m robot_dynamics.config > outputs/stage1/config.json
python -m pytest -q > outputs/stage1/tests.txt 2>&1
python -m ruff check robot_dynamics scripts tests
python scripts/audit_resources.py > outputs/stage1/resources.txt
python scripts/smoke_environment.py --device cpu --output outputs/stage1/cpu.json
MUJOCO_GL=egl python scripts/smoke_environment.py --device cuda --simulation --output outputs/stage1/cuda_simulation.json > outputs/stage1/smoke.log 2>&1
python -m pip freeze --all > outputs/stage1/linux.lock.txt
git rev-parse HEAD > outputs/stage1/commit.txt
git status --short > outputs/stage1/worktree.txt
tar -czf stage1-results.tar.gz -C outputs stage1
```

The full smoke check imports the selected packages and Square D0/D1 classes, steps a small MuJoCo model, runs a PyTorch forward/backward operation, and checks an in-memory HDF5 round trip. It does not replay Square trajectories, test rendering, or train a model. Those checks remain separate.

`--device numpy` is the base-only check available locally. It explicitly skips PyTorch and does not claim CPU autograd or CUDA success. The full CPU/CUDA paths have only been syntax-checked on Windows; their execution is part of this handoff.

Output files are protected against overwrite. After an interruption, rerun installation commands in the same environment; for a repeated smoke check use a fresh output filename, such as `outputs/stage1/cuda_simulation_retry.json`. There is no training state or checkpoint to resume.

## Return files

Return `stage1-results.tar.gz`, including configuration, hardware, package snapshot, smoke results, logs, and commit information. If installation fails, include the failing command and its complete log as well. Avoid adding credentials or unrelated machine details.

No dataset input is required. Dataset download, extraction, training, checkpoint recovery, and prediction evaluation are not implemented in this stage, so there are no commands for them yet. After the returned checks pass, record the Linux lock and results, close stage 1, and only then schedule the data audit.
