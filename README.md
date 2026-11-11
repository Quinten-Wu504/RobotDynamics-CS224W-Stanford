# RobotDynamics-CS224W-Stanford

CS224W course project at Stanford. We study graph-based prediction of robot and object motion using MimicGen Square.

## Project

The main question is whether graph structure helps predict manipulation dynamics compared with MLP and GRU baselines. We will also compare geometric equivariance and JEPA pretraining, keeping input information and training budgets comparable.

Models will take state histories and planned actions as input and predict position changes for the end effector and manipulated object. The verified control frequency is 20 Hz: the initial horizons are 1, 5, and 10 steps (0.05, 0.25, and 0.5 s). D0 will be used for training and development; held-out D0 and D1 trajectories will test generalization.

## Current status

The environment and Square data audit are complete on AutoDL: Ubuntu 22.04, Python 3.10.8, and an RTX 4090 D. The [experiment protocol](docs/experiment_protocol.md) defines the task; the [data report](reports/stage2/README.md) contains statistics, checksums, and recovery plots.

- **Code:** data auditing and recovery, fixed episode splits, training-only normalization, causal windows, and a shared position-baseline evaluator. Twenty-five tests pass. Learned models are not implemented yet.
- **Data:** 1,000 D0 and 1,000 D1 episodes scanned, with 153,477 and 152,400 recorded states respectively. No nonfinite values or inconsistent field layouts were found. Six complete trajectories (905 frames) passed state recovery checks.
- **Experiments:** position-hold and constant-velocity baselines have run on 100 D0 validation episodes. No learned model has been trained. D0 test and D1 remain reserved for formal evaluation.
- **Environment:** the full Linux dependency snapshot is locked. Windows still supports the base checks only.

## Evaluation plan

We report end-effector and object position errors separately in cm, averaging by episode and excluding static fixtures from the main metric. D0 is split into 800/100/100 train/validation/test episodes, with nested 80/200/800 training subsets and split seed 224. Learned-model formal comparisons will use at least three training seeds, with model selection based on validation data. Results include data hashes, sample counts, prediction horizon, configuration, and code commit.

## Baseline results

On D0 validation, mean end-effector errors at 0.05 / 0.25 / 0.5 s are **0.715 / 3.344 / 6.242 cm** for position hold and **0.247 / 1.973 / 5.138 cm** for constant velocity. Nut errors are **0.490 / 2.313 / 4.309 cm** and **0.207 / 1.449 / 3.494 cm**, respectively. Both methods use the same four-frame-history windows and ignore planned actions.

Constant velocity is a strong short-horizon baseline, but its advantage is smaller over 0.5 s and disappears in some low-motion groups. The [baseline report](reports/stage3/README.md) contains episode-bootstrap intervals, motion breakdowns, example predictions, and limitations. These are validation results, not final-test claims.

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

The [baseline commands](docs/baseline_protocol.md) reproduce splitting, normalization, evaluation, and plots. Use the committed [manifest](reports/stage3/manifest.json) for comparisons. On AutoDL CPU with a local cache copy, fitting all three normalizers took 1.8 s and evaluation took 1.3 s, with peak RSS below 145 MiB; these timings exclude cache copying, tests, and plotting. No GPU computation is needed for these baselines.

The container has a 16-core CPU quota, 80 GiB RAM, and 24 GiB VRAM. Its host-visible CPU and memory totals are larger than these limits. At the end of setup, the data disk had about 13 GiB free and file storage about 197 GiB free. Formal training requirements have not been measured.

For Windows base-only checks, use Python 3.10 with `requirements/windows-base.lock.txt` and `--device numpy`. Full simulator support was verified on Linux. [Environment notes](docs/environment.md) describe both machines and the resolved setup issues.

## Next steps

Stage 4 will add current-state/history MLP and GRU baselines, validation-based checkpoint selection, and training recovery. Small fitting checks will precede broader runs. There are no learned-model training or checkpoint-resume commands yet.

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
