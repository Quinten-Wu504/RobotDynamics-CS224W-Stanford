# RobotDynamics-CS224W-Stanford

CS224W course project at Stanford. We study graph-based prediction of robot and object motion using MimicGen Square.

## Project

The main question is whether graph structure helps predict manipulation dynamics compared with MLP and GRU baselines. We will also compare geometric equivariance and JEPA pretraining, keeping input information and training budgets comparable.

Models will take state histories and planned actions as input and predict position changes for the end effector and manipulated object. The verified control frequency is 20 Hz: the initial horizons are 1, 5, and 10 steps (0.05, 0.25, and 0.5 s). D0 will be used for training and development; held-out D0 and D1 trajectories will test generalization.

## Current status

The environment and Square data audit are complete on AutoDL: Ubuntu 22.04, Python 3.10.8, and an RTX 4090 D. The [experiment protocol](docs/experiment_protocol.md) defines the task; the [data report](reports/stage2/README.md) contains statistics, checksums, and recovery plots.

- **Code:** verified download, full-file audit, numeric state extraction, simulator recovery, and trajectory plots. Seventeen tests pass. Splits, prediction windows, and models are not implemented yet.
- **Data:** 1,000 D0 and 1,000 D1 episodes scanned, with 153,477 and 152,400 recorded states respectively. No nonfinite values or inconsistent field layouts were found. Six complete trajectories (905 frames) passed state recovery checks.
- **Experiments:** no prediction experiments or results. Recovery checks and short action replays validate data handling; they are not model evaluation.
- **Environment:** the full Linux dependency snapshot is locked. Windows still supports the base checks only.

## Evaluation plan

We will report end-effector and object position errors separately in cm, averaging by episode and excluding static fixtures from the main metric. Splits will be made by trajectory before generating windows. Formal comparisons will use at least three seeds and nested training subsets, with model selection based on validation data. Results will include the data split, sample counts, prediction horizon, configuration, and code commit.

## Setup

Clone the repository and follow the [environment setup](docs/stage_1_handoff.md) for installation, the project-local EGL library, and mirror options:

```sh
git clone https://github.com/Quinten-Wu504/RobotDynamics-CS224W-Stanford.git
cd RobotDynamics-CS224W-Stanford
```

The existing AutoDL installation is at `/root/autodl-tmp/RobotDynamics-CS224W-Stanford`. From that directory:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/native/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv/bin/python -m pytest -q
.venv/bin/python -m robot_dynamics.config
.venv/bin/python scripts/smoke_environment.py --device cpu --output outputs/cpu_check.json
MUJOCO_GL=egl .venv/bin/python scripts/smoke_environment.py --device cuda --simulation --output outputs/cuda_check.json
```

Use a new output filename for repeat checks; existing results are protected against overwrite. The configuration now passes `--require-audited` for the pinned Square release.

For data acquisition and processing, follow the [tested commands](docs/data_processing.md). The two raw HDF5 files and numeric caches are stored under `/root/autodl-fs/RobotDynamics-CS224W-Stanford-data`, outside Git. The [field table](docs/state_fields.md) explains the different D0/D1 observation layouts, world-coordinate fixtures, and end-effector orientation convention.

The container has a 16-core CPU quota, 80 GiB RAM, and 24 GiB VRAM. Its host-visible CPU and memory totals are larger than these limits. At the end of setup, the data disk had about 13 GiB free and file storage about 197 GiB free. Formal training requirements have not been measured.

For Windows base-only checks, use Python 3.10 with `requirements/windows-base.lock.txt` and `--device numpy`. Full simulator support was verified on Linux. [Environment notes](docs/environment.md) describe both machines and the resolved setup issues.

## Next steps

Stage 3 will create episode splits, causal windows, and a common evaluation path for position-hold and constant-velocity baselines. Normalization will use training episodes only. D1 remains reserved for final testing; its inspection so far concerns decoding and integrity. There are no training, prediction evaluation, or checkpoint-resume commands yet.

## Data and references

Download data from the official source and follow its usage terms. Raw datasets, checkpoints, and private course materials are kept out of this repository.

Starting references:

- [MimicGen datasets](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html)
- [MimicGen installation](https://mimicgen.github.io/docs/introduction/installation.html)
- [robomimic data format](https://robomimic.github.io/docs/datasets/overview.html)
- [robosuite controllers](https://robosuite.ai/docs/modules/controllers.html)
- [Graph Networks as Learnable Physics Engines](https://proceedings.mlr.press/v80/sanchez-gonzalez18a.html)
- [E(n) Equivariant Graph Neural Networks](https://proceedings.mlr.press/v139/satorras21a.html)
- [I-JEPA](https://arxiv.org/abs/2301.08243)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
