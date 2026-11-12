# Progress

## Roadmap

| Stage | Scope | Status |
| --- | --- | --- |
| 0 | Repository setup | DONE |
| 1 | Protocol, resource audit, minimal environment | DONE |
| 2 | Data audit, state recovery, visualization | DONE |
| 3 | Splits, windows, evaluation, non-learning baselines | DONE |
| 4 | Non-graph learning baselines | DONE |
| 5 | Scene graphs and temporal GNNs | DONE |
| 6 | Geometric equivariance | DONE |
| 7 | JEPA and pretraining controls | DONE |
| 8 | Frozen protocol and formal experiments | DONE |
| 9 | Evidence-driven robustness or extension | IN_PROGRESS |
| 10 | Reproduction and final deliverables | NOT_STARTED |

## Current notes

Stage 1 checks are complete on AutoDL. Eleven tests, dependency checks, CPU/CUDA gradients, HDF5 round trips, simulation imports, and a small MuJoCo step passed. The [results](../reports/stage1/autodl/README.md) identify source commit `6344eeb`.

The Windows `egl-probe` build failure was resolved by using Linux. A missing EGL loader on AutoDL was added inside the project environment; existing system graphics packages were not upgraded. The Linux package snapshot is saved in `requirements/linux-cu121.lock.txt`.

Stage 2 scanned all 2,000 Square D0/D1 episodes, produced numeric caches, and verified six complete trajectories against restored simulator states. The [data report](../reports/stage2/README.md) records counts, checksums, coordinate conventions, plots, and replay limitations.

Stage 3 fixed the split and nested training subsets, fitted training-only normalizers, and evaluated position hold and constant velocity on D0 validation. Twenty-five tests and 600 real boundary-window checks passed. Repeated evaluation produced identical episode results and summaries. See the [baseline report](../reports/stage3/README.md).

Stage 4 completed current-state MLP, history MLP, and GRU fitting checks and seed-0 pilots on 80 training episodes. All three passed exact continuation and standalone reload checks. Twenty-nine Linux tests pass. The [learning report](../reports/stage4/README.md) records results on 100 validation episodes, measured costs, and limits. D0 test and D1 remain untouched by prediction evaluation.

Stage 5 completed single-frame and temporal graph pilots, plus message-removal and full-edge controls. All four fit checks and standalone reloads passed; both main graph architectures also passed exact continuation checks. The Linux suite has 36 passing tests. The [graph report](../reports/stage5/README.md) records the mixed comparison with GRU, motion-group results, the sparse proximity group, and actual costs. No configuration search or test-set prediction evaluation was added.

Stage 6 completed canonical-frame Temporal GNN and GRU pilots. Both passed full-chain yaw/translation checks, exact continuation, and independent reloads. The Linux suite has 39 passing tests. The [equivariance report](../reports/stage6/README.md) separates coordinate consistency from the mixed prediction result and limited real-placement diagnostic. D1 and held-out D0 remain reserved for formal testing.

Stage 7 completed six ordinary/canonical Temporal GNN pilots: JEPA, direct-state pretraining, and 40-epoch supervision for each architecture. Both JEPA variants pass the declared collapse screen, and the canonical variants retain full-chain coordinate consistency. All six exact-resume and independent reload checks pass; the Linux suite has 43 passing tests. The [pretraining report](../reports/stage7/README.md) records that JEPA beats direct-state pretraining but trails longer supervision in both architectures. No extra settings were searched to reverse that result. D0 test and D1 were kept reserved for the formal evaluation.

Stage 8 completed all 60 configurations, 84 training phases, and 124 held-out evaluation sets in 14 h 53 min. All three seeds are retained; no run failed or was retried. Ordinary supervised Temporal GNN improves over the MLP/GRU controls on both splits at all fractions, while JEPA and canonicalization do not show consistent extra benefits. All 18 collapse screens and 21 coordinate checks pass. The final audit verifies checkpoint/source/split integrity, recomputes all episode summaries, and exactly reproduces two complete held-out prediction sets. The [formal report](../reports/stage8/README.md) includes all results, costs, caveats, and reproduction commands. The Linux suite still has 48 passing tests. D0 test and D1 have now been evaluated; future test-inspired work is exploratory. No stage 9 training jobs have started.

Stage 9 is preparing a second-task replication after the decision to extend beyond Square. The [Stack protocol](stack_protocol.md) specifies 12 runs: MLP, GRU, Temporal GNN, and a message-ablation control, each with three seeds at the full training fraction. The finite matrix is authorized without a time limit. Data audit, simulator restoration, all four model continuation/fit checks, and resource measurements have passed. The [stage 9 record](../reports/stage9/README.md) tracks the training handoff. Formal training has not yet started. The queue will stop before test evaluation; there are no Stack results. The GRAIL scope decision is also still open.

`CHECKING` marks a stage's final review; `DONE` marks a closed stage.
