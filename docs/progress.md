# Progress

## Roadmap

| Stage | Scope | Status |
| --- | --- | --- |
| 0 | Repository setup | DONE |
| 1 | Protocol, resource audit, minimal environment | CHECKING |
| 2 | Data audit, state recovery, visualization | NOT_STARTED |
| 3 | Splits, windows, evaluation, non-learning baselines | NOT_STARTED |
| 4 | Non-graph learning baselines | NOT_STARTED |
| 5 | Scene graphs and temporal GNNs | NOT_STARTED |
| 6 | Geometric equivariance | NOT_STARTED |
| 7 | JEPA and pretraining controls | NOT_STARTED |
| 8 | Frozen protocol and formal experiments | NOT_STARTED |
| 9 | Evidence-driven robustness or extension | NOT_STARTED |
| 10 | Reproduction and final deliverables | NOT_STARTED |

## Current notes

Stage 1 checks are complete on AutoDL. Eleven tests, dependency checks, CPU/CUDA gradients, HDF5 round trips, simulation imports, and a small MuJoCo step passed. The [results](../reports/stage1/autodl/README.md) identify source commit `6344eeb`.

The Windows `egl-probe` build failure was resolved by using Linux. A missing EGL loader on AutoDL was added inside the project environment; existing system graphics packages were not upgraded. The Linux package snapshot is saved in `requirements/linux-cu121.lock.txt`.

The protocol and configuration are ready for the data audit. Data fields, controller semantics, control frequency, timing, and fixture poses still need verification against the dataset. No data or model experiment has run.

Next: stage 2 data inspection and state recovery. `CHECKING` marks a stage's final review; `DONE` marks a closed stage.
