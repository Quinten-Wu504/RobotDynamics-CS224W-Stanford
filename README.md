# RobotDynamics-CS224W-Stanford

CS224W course project at Stanford. We study graph-based prediction of robot and object motion using MimicGen Square and Stack.

## Project

The main question is whether graph structure helps predict manipulation dynamics compared with MLP and GRU baselines. Square also tests geometric equivariance and JEPA pretraining; Stack replicates the supervised graph comparison with a message-ablation control.

Models take state histories and planned actions as input and predict position changes for the end effector and manipulated objects. The verified control frequency is 20 Hz: horizons are 1, 5, and 10 steps (0.05, 0.25, and 0.5 s). D0 supplies training and development data; held-out D0 and D1 trajectories test generalization.

## Current status

The environment and Square data audit are complete on AutoDL: Ubuntu 22.04, Python 3.10.8, and an RTX 4090 D. The [experiment protocol](docs/experiment_protocol.md) defines the task; the [data report](reports/stage2/README.md) contains statistics, checksums, and recovery plots.

- **Code:** data processing, fixed splits, causal windows, shared evaluation, MLP/GRU and graph baselines, canonical-frame equivariant models, and JEPA/direct-state pretraining. The resumable formal pipeline includes checkpoint sealing, test evaluation, diagnostics, and automatic tables/plots. The Linux suite has 56 passing tests.
- **Data (Square):** 1,000 D0 and 1,000 D1 episodes scanned, with 153,477 and 152,400 recorded states respectively. No nonfinite values or inconsistent field layouts were found. Six complete trajectories (905 frames) passed state recovery checks.
- **Square experiments:** all 60 formal configurations and 124 evaluation sets are complete, covering three seeds and 80/200/800 training episodes, with the state-pretraining control limited to 80 episodes. Held-out D0 and D1 were evaluated only after all checkpoints were selected and sealed. All diagnostics passed; the [formal report](reports/stage8/README.md) retains every seed and comparison.
- **Stack replication:** 12 trained configurations and 28 evaluation sets are complete, covering four models and three seeds. Both releases contain 1,000 episodes; the Stack audit and six complete restorations passed. The [Stack report](reports/stage9/README.md) contains the full comparison and ablation. The separate [GRAIL demonstration](reports/stage9/grail/README.md) now has a frozen 12-run history-only forecasting protocol; results are pending.
- **Environment:** the Linux snapshot now includes PyG 2.6.1. Windows supports the base checks and plotting only.

GRAIL uses a separate history-only task on G1 robot-root and object anchors: 240 motions split by asset, with horizons of 0.04/0.24/0.48 s at 25 Hz. Models are trained from scratch; this is an abstraction demonstration, not zero-shot or pretrained weight transfer. See the [GRAIL protocol](docs/grail_protocol.md) for its inputs and limitations.

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

## Development results before formal testing

The following stages used D0 validation only. Their pilot observations motivated the frozen formal comparison above.

On D0 validation, mean end-effector errors at 0.05 / 0.25 / 0.5 s are **0.715 / 3.344 / 6.242 cm** for position hold and **0.247 / 1.973 / 5.138 cm** for constant velocity. Nut errors are **0.490 / 2.313 / 4.309 cm** and **0.207 / 1.449 / 3.494 cm**, respectively. Both methods use the same four-frame-history windows and ignore planned actions.

Constant velocity is a strong short-horizon baseline, but its advantage is smaller over 0.5 s and disappears in some low-motion groups. The [baseline report](reports/stage3/README.md) contains episode-bootstrap intervals, motion breakdowns, example predictions, and limitations. These are validation results, not final-test claims.

The seed-0 learning pilots reach 0.5-second end-effector / nut errors of **0.550 / 0.455 cm** (current-state MLP), **0.518 / 0.446 cm** (history MLP), and **0.496 / 0.437 cm** (GRU). Among these three non-graph baselines, GRU has the lowest overall validation error, but the differences are small. They use planned actions, which the non-learning baselines ignore. The [learning report](reports/stage4/README.md) includes all horizons, learning curves, confidence intervals, resource measurements, and reproducibility checks. These single-seed pilots do not establish graph or temporal-model superiority.

The [graph pilots](reports/stage5/README.md) give a mixed result. Temporal GNN reaches **0.537 / 0.439 cm** at 0.5 seconds and a six-error mean of **0.329 cm**, close to GRU's **0.333 cm**. Closing messages raises that mean to 0.346 cm; adding the static fixture edges gives 0.331 cm. Temporal GNN improves low-motion nut prediction but trails GRU on moving windows, and much of the low-motion gain remains without messages. The predefined 5 cm proximity group has only four episodes, so it does not support a contact-stage claim. No model is a clear overall winner across accuracy and cost.

The [equivariance pilots](reports/stage6/README.md) use the known peg frame, without adding parameters. Canonical Temporal GNN reaches a six-error mean of **0.325 cm**, a small improvement over ordinary Temporal GNN; canonical GRU worsens to **0.347 cm**. Both pass full-chain yaw/translation consistency checks below 3.5e-8 m maximum discrepancy. This verifies a coordinate property, not physical OOD performance. The real D0 placement-coverage diagnostic has only five outside-hull episodes and shows no consistent benefit; D1 had not yet been evaluated at that stage. The construction uses canonicalization, not an EGNN message layer.

The [pretraining pilots](reports/stage7/README.md) do not show an extra JEPA benefit at this budget. With 10 pretraining plus 30 fine-tuning epochs, ordinary/canonical JEPA reaches **0.325 / 0.320 cm**. Direct-state pretraining gives **0.333 / 0.340 cm**, while 40 epochs of direct supervision gives **0.305 / 0.310 cm**. JEPA beats the state-pretraining control but trails longer supervision in both architectures, despite using slightly more update time. Both JEPA variants pass the declared collapse screen. These are fixed-budget, seed-0 validation results; the earlier GRU used only 30 epochs, so this comparison does not establish graph superiority.

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

The [learning commands](docs/learning_baselines.md) cover fitting, pilot training, checkpoint evaluation, recovery, and report generation. The three 30-epoch pilots took about 159 seconds of measured run time in total, plus 26 seconds of input preparation. Checkpoints remain in AutoDL `outputs/stage4`; metrics and plots are committed under `reports/stage4`.

For graph models, follow the [graph protocol and commands](docs/graph_baselines.md), including installation of `torch-geometric==2.6.1`. The full environment is recorded in [linux-stage5.lock.txt](requirements/linux-stage5.lock.txt). Four graph pilots took about 12.6 minutes of measured run time plus 34 seconds of input preparation. Graph checkpoints remain in AutoDL `outputs/stage5`, and small results are in `reports/stage5`. Figures can be regenerated on CPU without data or checkpoints:

```bash
python -m robot_dynamics.plot_graphs reports/stage5/analysis/analysis.json --output outputs/stage5_figures
```

The [equivariance protocol](docs/equivariance_protocol.md) provides stage 6 training and validation commands using the same environment. Its two pilots took about six minutes of measured run time plus 18 seconds of input preparation; concurrent GPU work prevents a controlled speed comparison with earlier runs. Checkpoints remain in AutoDL `outputs/stage6`. To redraw its figures without data or checkpoints:

```bash
python -m robot_dynamics.plot_equivariance reports/stage6/analysis/analysis.json --output outputs/stage6_figures
```

The [pretraining protocol](docs/pretraining_protocol.md) provides the six stage 7 training commands, exact-resume checks, independent evaluation, and paired analysis. It uses the same Linux environment, with no new dependencies. Checkpoints remain in AutoDL `outputs/stage7`; small results are in `reports/stage7`. Figures can be redrawn on CPU:

```bash
python -m robot_dynamics.plot_pretraining reports/stage7/analysis/analysis.json --output outputs/stage7_figures
```

The [formal protocol](docs/formal_protocol.md) gives the end-to-end stage 8 command. The complete pipeline took 14 h 53 min on the existing RTX 4090 D; peak recorded CUDA allocation was 1.15 GiB and sampled RSS 2.59 GiB. Its checkpoint and raw-data files stay on AutoDL; all selected configurations, metrics, logs, and compressed episode errors are committed under `reports/stage8/formal`. To redraw the formal figures on CPU:

```bash
python -m robot_dynamics.plot_formal reports/stage8/formal/analysis/analysis.json --output outputs/stage8_figures
```

The container has a 16-core CPU quota, 80 GiB RAM, and 24 GiB VRAM. Its host-visible CPU and memory totals are larger than these limits.

For Windows base-only checks, use Python 3.10 with `requirements/windows-base.lock.txt` and `--device numpy`. Full simulator support was verified on Linux. [Environment notes](docs/environment.md) describe both machines and the resolved setup issues.

## Next steps

The Square formal study and Stack replication are complete. Stack training took 3 h 50 min; its 28 test evaluation sets and independent checks are archived in the [stage 9 report](reports/stage9/README.md). Both tasks' test sets have now been evaluated, so new test-inspired experiments must be marked exploratory. GRAIL preparation and a 12-run training queue are documented in the [launch record](reports/stage9/grail/README.md); Stage 9 remains open until training and held-out evaluation finish. Stage 10 reproduction/Colab work has not started.

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
