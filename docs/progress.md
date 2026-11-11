# Progress

## Roadmap

| Stage | Scope | Status |
| --- | --- | --- |
| 0 | Repository setup | DONE |
| 1 | Protocol, resource audit, minimal environment | DONE |
| 2 | Data audit, state recovery, visualization | DONE |
| 3 | Splits, windows, evaluation, non-learning baselines | DONE |
| 4 | Non-graph learning baselines | IN_PROGRESS |
| 5 | Scene graphs and temporal GNNs | NOT_STARTED |
| 6 | Geometric equivariance | NOT_STARTED |
| 7 | JEPA and pretraining controls | NOT_STARTED |
| 8 | Frozen protocol and formal experiments | NOT_STARTED |
| 9 | Evidence-driven robustness or extension | NOT_STARTED |
| 10 | Reproduction and final deliverables | NOT_STARTED |

## Current notes

Stage 1 checks are complete on AutoDL. Eleven tests, dependency checks, CPU/CUDA gradients, HDF5 round trips, simulation imports, and a small MuJoCo step passed. The [results](../reports/stage1/autodl/README.md) identify source commit `6344eeb`.

The Windows `egl-probe` build failure was resolved by using Linux. A missing EGL loader on AutoDL was added inside the project environment; existing system graphics packages were not upgraded. The Linux package snapshot is saved in `requirements/linux-cu121.lock.txt`.

Stage 2 scanned all 2,000 Square D0/D1 episodes, produced numeric caches, and verified six complete trajectories against restored simulator states. The [data report](../reports/stage2/README.md) records counts, checksums, coordinate conventions, plots, and replay limitations.

Stage 3 fixed the split and nested training subsets, fitted training-only normalizers, and evaluated position hold and constant velocity on D0 validation. Twenty-five tests and 600 real boundary-window checks passed. Repeated evaluation produced identical episode results and summaries. See the [baseline report](../reports/stage3/README.md). No learned model has been trained, and the test sets remain untouched by prediction evaluation.

Next: stage 4 MLP/GRU fitting, training logs, and checkpoint recovery. `CHECKING` marks a stage's final review; `DONE` marks a closed stage.
