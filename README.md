# RobotDynamics-CS224W-Stanford

Graph learning for robot dynamics prediction, developed as a CS224W course project at Stanford.

## Research scope

The project will test whether graph structure, geometric equivariance, and joint-embedding predictive learning (JEPA) improve robot and object dynamics prediction under comparable information and training budgets. Positive results are not assumed.

The initial task is MimicGen Square: D0 for development and training, followed by held-out D0 and D1 distribution-shift evaluation. Planned inputs are causal state histories and planned action sequences; initial targets are end-effector and manipulated-object position changes. Static fixtures will not be included in the primary average error. One-step alignment will be checked before predicting 0.25 s and 0.5 s ahead, using the verified control frequency.

## Current status

Stage 0 contains repository documentation only. See [progress](docs/progress.md) and the [initialization record](docs/stage_0.md).

| Area | Status |
| --- | --- |
| Code implemented | Documentation and Git ignore rules only |
| Code verified | Initialization content reviewed; no model or data code exists |
| Experiments run | None |
| Experiments pending | All data audits, baselines, models, and evaluations |
| Local compute | Not audited; no compute blocker has been established |

## Results and reproducibility

There are no experimental results or empirical conclusions yet. Dataset counts, split manifests, seeds, fitted normalization, configurations, and result artifacts have not been produced. Planned primary metrics are separate end-effector and object Euclidean position errors in cm, aggregated by episode. Formal comparisons will use at least three seeds, nested training subsets, and validation-only model selection. Each future result will identify its source commit and configuration.

## Running the project

There is no training, preprocessing, evaluation, visualization, or checkpoint-resume entry point yet. No runtime dependencies or environment have been installed for this project. The repository can be retrieved with Git:

```sh
git clone https://github.com/Quinten-Wu504/RobotDynamics-CS224W-Stanford.git
cd RobotDynamics-CS224W-Stanford
```

Hardware requirements and local/remote installation instructions will be documented after the resource and compatibility audit. There are currently no generated outputs or checkpoints.

## Next stage

Stage 1 will audit CPU, RAM, GPU, disk, Python, and platform compatibility; verify compatible dependency versions; define the experiment protocol; and prepare an isolated minimal environment with import and device smoke tests. Dataset extraction and training belong to later stages.

## Data and references

Data must be obtained from its official source under the applicable terms. Raw datasets, credentials, environments, large checkpoints, and private course materials are not distributed here. Repository visibility is public at the owner's request; data and third-party materials retain their respective usage conditions.

These are planned references; dependency versions and data semantics have not yet been audited:

- [MimicGen datasets](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html)
- [MimicGen installation](https://mimicgen.github.io/docs/introduction/installation.html)
- [robomimic data format](https://robomimic.github.io/docs/datasets/overview.html)
- [robosuite controllers](https://robosuite.ai/docs/modules/controllers.html)
- [Graph Networks as Learnable Physics Engines](https://proceedings.mlr.press/v80/sanchez-gonzalez18a.html)
- [E(n) Equivariant Graph Neural Networks](https://proceedings.mlr.press/v139/satorras21a.html)
- [I-JEPA](https://arxiv.org/abs/2301.08243)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
