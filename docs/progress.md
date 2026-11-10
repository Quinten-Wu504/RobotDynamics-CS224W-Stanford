# Progress

## Roadmap

| Stage | Scope | Status |
| --- | --- | --- |
| 0 | Repository setup | DONE |
| 1 | Protocol, resource audit, minimal environment | BLOCKED_COMPUTE |
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

The task protocol and configuration are written. The Windows base environment is installed and the configuration checks pass. The full simulation environment is blocked by an `egl-probe` build failure. No data has been downloaded and no model has been trained.

Stage 1 remains open. Next: run the full environment check on Linux, return its logs and dependency snapshot, and check the results before starting data work. See [environment notes](environment.md) and [handoff commands](stage_1_handoff.md).

Local verification: nine configuration tests pass; the base import/HDF5 smoke check passed from commit `74cb4d3`. [Check record](../reports/stage1/validation.md).

`CHECKING` means a final pass is in progress; `DONE` means the stage is closed. `BLOCKED_COMPUTE` includes platform or runtime limitations. Stage 0 was closed when stage 1 began.
