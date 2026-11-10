# RobotDynamics-CS224W-Stanford

CS224W course project at Stanford. We study graph-based prediction of robot and object motion using MimicGen Square.

## Project

The main question is whether graph structure helps predict manipulation dynamics compared with MLP and GRU baselines. We will also compare geometric equivariance and JEPA pretraining, keeping input information and training budgets comparable.

Models will take state histories and planned actions as input and predict position changes for the end effector and manipulated object. We will start with one-step prediction, then use 0.25 s and 0.5 s horizons once the data's control frequency is confirmed. D0 will be used for training and development; held-out D0 and D1 trajectories will test generalization.

## Current status

The repository is initialized. Implementation and experiments have not started. See the [project progress](docs/progress.md) for the remaining work.

- **Code:** documentation and ignore rules are in place and checked. No data pipeline or models yet.
- **Experiments:** none run; no results yet.
- **Environment:** dependency versions and hardware requirements still need checking.

## Evaluation plan

We will report end-effector and object position errors separately in cm, averaging by episode and excluding static fixtures from the main metric. Splits will be made by trajectory before generating windows. Formal comparisons will use at least three seeds and nested training subsets, with model selection based on validation data. Results will include the data split, sample counts, prediction horizon, configuration, and code commit.

## Setup

Clone the repository:

```sh
git clone https://github.com/Quinten-Wu504/RobotDynamics-CS224W-Stanford.git
cd RobotDynamics-CS224W-Stanford
```

There is no training entry point yet. Installation and run commands will be added with the implementation, after checking the MimicGen, robosuite, and MuJoCo versions on the available hardware.

## Next steps

Check local resources and platform compatibility, write the experiment protocol, and set up an isolated environment. The first checks will cover dependency imports and device access; data processing comes next.

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
