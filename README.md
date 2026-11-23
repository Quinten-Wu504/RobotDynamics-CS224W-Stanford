# RobotDynamics-CS224W-Stanford

CS224W course project at Stanford. We study graph-based prediction of robot and object motion using MimicGen Square and Stack, with a separate history-only GRAIL pickup demonstration.

## Project

The main question is whether graph structure helps predict manipulation dynamics compared with MLP and GRU baselines. Square also tests geometric equivariance and JEPA pretraining; Stack replicates the supervised graph comparison with a message-ablation control.

Models take state histories and planned actions as input and predict position changes for the end effector and manipulated objects. The verified control frequency is 20 Hz: horizons are 1, 5, and 10 steps (0.05, 0.25, and 0.5 s). D0 supplies training and development data; held-out D0 and D1 trajectories test generalization.

## Reproduce and read the results

- [Final report](reports/stage10/README.md): conclusions, negative results, costs and limitations across Square, Stack and GRAIL.
- [Colab notebook](https://colab.research.google.com/github/Quinten-Wu504/RobotDynamics-CS224W-Stanford/blob/main/notebooks/robot_dynamics_demo.ipynb): a real eight-training/two-validation Square example with graph construction, training, checkpoint reload and prediction plots.
- [Reproduction guide](docs/reproduction.md): pinned data, environment, command-line example, checkpoint locations and full-matrix requirements.

For figures and tables only, no GPU, model weights or raw data are needed:

```bash
python -m pip install -r requirements/base.txt
python -m robot_dynamics.final_results --output outputs/final_results
```

The exporter includes all 62 aggregate results and 84 learned-run costs. It preserves each task's metric rather than pooling scores. The notebook is a short pipeline demonstration, separate from the formal evidence below.

## Current status

The formal experiments are complete: 60 Square configurations, 12 Stack configurations and 12 GRAIL configurations, with all three seeds retained. The code covers audited data processing, causal windows, MLP/GRU and graph models, canonical coordinates, JEPA/state-pretraining controls, checkpoint sealing, evaluation and plotting.

The final reproduction notebook has been executed in a fresh Windows CPU environment and a separate clean clone. It reproduces its checkpoint predictions exactly; a two-epoch interruption/resume check also matches uninterrupted training. Both Windows CPU and Linux pass 63 tests. A formal GRAIL checkpoint was independently replayed during final verification. The hosted Colab interface itself was not run; [validation records](reports/stage10/README.md#reproduction-and-delivery-checks) distinguish these checks. Formal weights remain in AutoDL storage, while the notebook trains and exports its own small checkpoint.

GRAIL uses a separate history-only task on G1 robot-root and object anchors: 240 motions split by asset, with horizons of 0.04/0.24/0.48 s at 25 Hz. Models are trained from scratch; see the [GRAIL protocol](docs/grail_protocol.md) for inputs and limitations.

## Evaluation protocol

We report end-effector and object position errors separately in cm, averaging by episode and excluding static fixtures from the main metric. D0 is split into 800/100/100 train/validation/test episodes, with nested 80/200/800 training subsets and split seed 224. Learned-model formal comparisons use three training seeds, with model selection based on validation data. Results include data hashes, sample counts, prediction horizon, configuration, and code commit.

## Formal Square results

Directly supervised Temporal GNN outperforms the tested history MLP and GRU at every data fraction on both D0 test and D1, with the same direction across all three seeds. At 800 training episodes, the six-error means ± seed SD are:

| Model | D0 test, cm | D1, cm |
| --- | ---: | ---: |
| History MLP | 0.1972 ± 0.0057 | 2.4142 ± 0.0968 |
| GRU | 0.1792 ± 0.0015 | 1.4505 ± 0.1171 |
| Temporal GNN | **0.1665 ± 0.0014** | 1.2010 ± 0.0227 |
| Canonical Temporal GNN | 0.1696 ± 0.0007 | 1.1754 ± 0.0541 |
| Temporal GNN + JEPA | 0.1865 ± 0.0031 | 1.1740 ± 0.0838 |
| Canonical Temporal GNN + JEPA | 0.1861 ± 0.0044 | 1.2721 ± 0.2158 |

Ordinary supervised TGN reduces error relative to GRU by **7.1% on D0 and 17.2% on D1** at this fraction. This supports these graph-model configurations on Square, but not universal or compute-normalized superiority: TGN takes roughly twice GRU's measured update time. D1 errors remain much higher than D0 errors, and constant velocity still wins at D1's shortest horizon.

JEPA worsens D0 results across all fractions. Its apparent D1 gains, and canonicalization's small full-data D1 gain, vary across seeds and have intervals that include zero. All 18 JEPA collapse screens and 21 canonical coordinate checks pass, so implementation consistency does not by itself establish prediction benefit. The [full report](reports/stage8/README.md) contains paired intervals, motion/proximity groups, costs, all seed values, compressed episode errors, and reproduction commands. No test-driven tuning was performed; future test-inspired extensions must be labeled exploratory.

## Stack replication results

The second task uses 800 training episodes, three seeds, and the same 40-epoch budget. Its primary metric averages the end effector and both cubes across three horizons; it should not be pooled with Square's six-cell score.

| Model | D0 test, cm | D1, cm |
| --- | ---: | ---: |
| History MLP | 0.1622 ± 0.0026 | 0.2655 ± 0.0125 |
| GRU | 0.1419 ± 0.0030 | 0.2875 ± 0.0084 |
| Temporal GNN | **0.1175 ± 0.0030** | **0.1929 ± 0.0063** |
| Temporal GNN, messages closed | 0.1362 ± 0.0034 | 0.2254 ± 0.0084 |

GNN reduces error relative to GRU by 17.2%/32.9% and relative to the message-ablation control by 13.7%/14.4% on D0/D1. Every seed favors GNN in these aggregate comparisons. The message variants have equal nominal parameter counts and nearly equal update time; GNN and GRU have almost equal parameters but GNN takes about twice as long to train. Position hold still beats learned models on cube B at every horizon. The [complete report](reports/stage9/README.md) retains all seeds, paired intervals, subgroup exceptions, and reproducible episode errors. This supports supervised graph prediction in two tasks, not universal superiority, extra JEPA benefit, or closed-loop control performance.

## GRAIL demonstration results

The history-only GRAIL task uses 168 training motions and 36 held-out test motions from 18 unseen assets. Its six-cell mean errors are in cm; lower is better.

| Model | Test mean ± seed SD, cm |
| --- | ---: |
| History MLP | 1.0001 ± 0.0209 |
| GRU | 1.0825 ± 0.0242 |
| Temporal GNN | 1.0109 ± 0.0189 |
| Temporal GNN, messages disabled | 0.9823 ± 0.0269 |
| Position hold | 1.7052 |
| Constant velocity | 1.2488 |

GNN improves over GRU by 6.6%, but its mean is 1.1% worse than MLP and 2.9% worse than messages disabled. The joint seed/asset intervals for those last two differences include zero. This does not reproduce Stack's message-passing benefit. All six graph runs selected the earliest allowed checkpoint (epoch 11), after an earlier validation minimum; the conclusion applies to this fixed recipe, not optimally tuned architectures. The [GRAIL report](reports/stage9/grail/README.md) retains all seeds, subgroups, costs, learning curves and trajectory examples. No checkpoints were reselected after seeing test results.

## Implementation and earlier experiments

The [data audit](reports/stage2/README.md), [baseline study](reports/stage3/README.md), [learning pilots](reports/stage4/README.md), [graph pilots](reports/stage5/README.md), [equivariance checks](reports/stage6/README.md), and [pretraining pilots](reports/stage7/README.md) preserve the development history. Their validation scores are separate from the held-out results above. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) is an archived pilot summary, not the current result report.

The [reproduction guide](docs/reproduction.md) is the current starting point for installation, data processing, training, recovery, checkpoint evaluation and plotting. The [Linux environment record](docs/environment.md) covers the original CUDA/simulator setup; the numeric demo needs only the CPU dependencies. Data and checkpoints are excluded from git.

## Remaining scope

The planned experiments and reproduction materials are delivered; Stage 10 is in final review (`CHECKING`). No training or evaluation job remains running. Hosted Colab execution, public distribution of the formal checkpoints, and closed-loop control have not been demonstrated. All formal test sets have been used, so any new test-inspired experiment is exploratory. No blog has been published.

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
