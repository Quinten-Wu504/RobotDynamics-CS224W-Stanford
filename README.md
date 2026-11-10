# RobotDynamics-CS224W-Stanford

CS224W course project at Stanford. We study graph-based prediction of robot and object motion using MimicGen Square.

## Project

The main question is whether graph structure helps predict manipulation dynamics compared with MLP and GRU baselines. We will also compare geometric equivariance and JEPA pretraining, keeping input information and training budgets comparable.

Models will take state histories and planned actions as input and predict position changes for the end effector and manipulated object. We will start with one-step prediction, then use 0.25 s and 0.5 s horizons once the data's control frequency is confirmed. D0 will be used for training and development; held-out D0 and D1 trajectories will test generalization.

## Current status

The [experiment protocol](docs/experiment_protocol.md) and initial configuration are in place. The base environment works on Windows, but the simulation stack is blocked by a dependency build failure. See [project progress](docs/progress.md).

- **Code:** configuration checks, a resource-audit script, and an environment smoke-test script. No data pipeline or models yet.
- **Experiments:** no prediction experiments or results. Environment checks are separate from model evaluation.
- **Environment:** Python 3.10 base dependencies are installed and locked. Full simulation imports and PyTorch CPU/CUDA checks still need a compatible host.

The [stage 1 check record](reports/stage1/validation.md) includes nine passing tests and the base smoke-test output with its source commit.

## Evaluation plan

We will report end-effector and object position errors separately in cm, averaging by episode and excluding static fixtures from the main metric. Splits will be made by trajectory before generating windows. Formal comparisons will use at least three seeds and nested training subsets, with model selection based on validation data. Results will include the data split, sample counts, prediction horizon, configuration, and code commit.

## Setup

Clone the repository:

```sh
git clone https://github.com/Quinten-Wu504/RobotDynamics-CS224W-Stanford.git
cd RobotDynamics-CS224W-Stanford
```

For the tested Windows base environment, use Python 3.10:

```powershell
D:\ProgramData\anaconda3\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements/windows-base.lock.txt
.venv\Scripts\python.exe -m robot_dynamics.config
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/smoke_environment.py --device numpy --output outputs/base_smoke.json
```

The smoke check tests imports and an in-memory HDF5 round trip. It does not train a model. Output files are not overwritten; use a new output path for a repeat run. `--require-audited` on the config command intentionally fails until the stage 2 data checks are complete.

The full environment uses pinned MimicGen and robomimic commits, robosuite 1.4.1, and MuJoCo 2.3.2. Its `egl-probe` dependency fails to build on this Windows machine. [Environment notes](docs/environment.md) contain the error and version choices; [Linux handoff](docs/stage_1_handoff.md) contains installation, CPU/CUDA smoke checks, and export commands. The full stack has not yet been verified.

Local hardware: i7-10750H, 15.76 GiB RAM (2.46 GiB available during the audit), Quadro T1000 Max-Q with 4 GiB VRAM, and 77.8 GiB free on the project drive. Formal training requirements have not been measured. There are no dataset preparation, training, evaluation, or checkpoint-resume commands yet.

## Next steps

Finish the full environment checks on a compatible Linux host. After that, the next stage will audit Square D0/D1 data, confirm actions and timing, restore world poses, and inspect sample trajectories.

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
