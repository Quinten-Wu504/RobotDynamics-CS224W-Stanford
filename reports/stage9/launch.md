# Stack training launch

Archived launch record. Training has since completed; see [the results report](README.md) for evaluation and current status.

This stage tests whether Square's supervised graph result carries over to MimicGen Stack. The [frozen matrix](frozen_matrix.json) has four models and three seeds at the full training fraction. It is a new training task, not zero-shot transfer. No Stack test prediction errors have been opened.

The training-only queue launched at **2026-11-12 12:09:08 UTC** from clean commit `34abd1e`, with driver PID **165850**. The [disconnect check](launch_verification.json) records an actual SSH logout and reconnect: the driver retained PPID 1, its own session, and no terminal, while the first run advanced from epoch 2 to epoch 6 and wrote its checkpoint. Training remains in progress; this is a launch record, not a completed-results report. Later documentation-only commits do not change the frozen training source hashes.

## Data and representation

The [release record](source.json) pins the official revision and SHA-256 values. Both downloaded files match those hashes. Transfers used the Hugging Face mirror after the direct connection from AutoDL failed; the expected checksums came from the official repository metadata.

| Distribution | Episodes | Recorded frames | Episode length range |
| --- | ---: | ---: | ---: |
| Stack D0 | 1,000 | 107,590 | 80–136 |
| Stack D1 | 1,000 | 108,233 | 82–138 |

The decoder checked all used numeric fields, timestamp spacing, normalized actions, controller metadata, cube-relative observation identities, unit quaternions, and XML geometry. No exact state/action trajectory duplicates were found within or across distributions. These hashes do not identify shared demonstration ancestry. Six complete trajectories (645 frames) were restored in the simulator; the largest discrepancy across checked poses, geometry, joints, and gripper fields was 6.7e-16. Images are not model inputs and were not decoded by this audit.

The task uses Panda incremental OSC_POSE commands at 20 Hz. The four-node graph contains end effector, cube A, cube B, and table. Both cubes have time-varying position and orientation inputs; neither is treated as a fixture. Features include all four positions/orientations, box geometry, robot joints/gripper, node identity, and relative vectors to all three movable entities. Every model receives the same information plus the planned action sequence. A shared node packing feeds the graph or is flattened for MLP/GRU.

All three movable entities are prediction targets at 1/5/10 steps (0.05/0.25/0.5 s). The primary score averages nine entity/horizon errors equally within each episode, then weights episodes equally. The table is excluded. Static- versus moving-object behavior will be reported separately. Low offline error does not establish successful closed-loop stacking.

## Frozen comparison and checks

| Model | Parameters | Seeds |
| --- | ---: | --- |
| History MLP | 133,769 | 0, 1, 2 |
| GRU | 125,257 | 0, 1, 2 |
| Temporal GNN | 125,379 | 0, 1, 2 |
| Temporal GNN, messages closed | 125,379 | 0, 1, 2 |

The graph uses all 12 directed non-self edges. The message ablation still computes and then zeros messages, retaining the parameter count and input representation. Its disabled message-network parameters have no effective influence, and explicit relative positions remain available. This is an ablation of learned neighbor exchange, not removal of every relational feature.

The protocol uses 800/100/100 D0 train/validation/test episodes with split seed 224; all 1,000 D1 episodes are reserved for test prediction. Every model trains from scratch for 40 epochs; validation selects the best checkpoint from epochs 11–40. Hyperparameters and seeds are fixed in [stack_protocol.yaml](../../configs/stack_protocol.yaml). There is no data-fraction search or JEPA/equivariance matrix on Stack. The scope tests supervised graph generalization and message dependence; it cannot establish cross-task JEPA benefits.

Data extraction and simulator restoration used clean commit `cc49e97`. All four models passed exact continuation, independent reload, completed-checkpoint replay, and small-sample fitting; final/initial fit MSE ratios were 0.0147, 0.0107, 0.0066, and 0.0068 in the table order. The Linux suite passed 55 tests. A two-epoch probe per model used all 800 training episodes (238,144 windows) and all 100 validation episodes (29,330 windows). Mean update times per epoch were 10.13 s, 19.84 s, 42.60 s, and 42.00 s, respectively. This extrapolates to 3.82 hours of optimizer updates for the 12-run matrix; allow roughly 4–5 hours for training and reloads, with load-dependent variation. Peak sampled RSS was 2.13 GiB and peak allocated CUDA memory 1.04 GiB. These are resource/fit checks, not held-out accuracy results. The duplicate driver check exited with code 1 while leaving the original queue running. Stage 9 remains open; there are no second-task accuracy conclusions yet.

## Running and recovering

The [manifest](cache/manifest.json), [simulator restoration](restoration.json), [four-model checks](preflight/checks.json), [resource measurements](resource_summary.json), [workflow state](workflow_check/progress.json), and [queue replay](workflow_replay.log) preserve the preflight evidence. The workflow completed all four small runs, independently reloaded their validation predictions, and sealed selections without accessing test predictions. Replay used clean commit `b5f9e91` with unchanged model/training code. [Tests](tests.txt), [lint](ruff.txt), [formatting](format.txt), and [dependencies](dependencies.txt) passed. No temporary fallback code or synthetic-data training path remains; necessary data, finite-value, checkpoint, source, and duplicate-run protections are retained.

Use the existing Linux environment in `/root/autodl-tmp/RobotDynamics-CS224W-Stanford`. The pinned dependencies and EGL setup are described in the [environment handoff](../../docs/stage_1_handoff.md). Raw files are in `/root/autodl-fs/RobotDynamics-CS224W-Stanford-data/stack/raw`; numeric caches are in `outputs/stage9/cache`.

The download module verifies and reuses existing files. Interrupted `.partial` files are preserved and must be moved aside before restarting a transfer. A fresh transfer through this module has not been exercised; the recorded downloads used curl with the same pinned URLs and checksums. Audit/check output directories must be new.

```bash
.venv/bin/python -m robot_dynamics.download_stack --directory /root/autodl-fs/RobotDynamics-CS224W-Stanford-data/stack/raw --mirror
.venv/bin/python -m robot_dynamics.stack_data /root/autodl-fs/RobotDynamics-CS224W-Stanford-data/stack/raw --output outputs/stack_audit_repeat
.venv/bin/python -m robot_dynamics.verify_stack /root/autodl-fs/RobotDynamics-CS224W-Stanford-data/stack/raw --output outputs/stack_restoration_repeat.json
.venv/bin/python -m robot_dynamics.check_stack outputs/stage9/cache/stack_d0.hdf5 --manifest outputs/stage9/cache/manifest.json --output outputs/stack_checks_repeat
```

The full training-only queue, using the already audited cache:

```bash
nohup setsid bash scripts/run_stage9.sh outputs/stage9/cache/stack_d0.hdf5 outputs/stage9/cache/manifest.json outputs/stage9/formal > outputs/stage9/formal_driver.log 2>&1 < /dev/null &
```

The queue writes an atomic checkpoint each epoch, checks identities on resume, skips verified finished runs, and performs independent validation reloads before sealing selections. Driver and runner locks reject duplicate queues. If recovery is needed, use the same directory and command; use `>> outputs/stage9/formal_driver.log` to append to the driver log. Do not remove a failed run to hide its history. Logs, configs, normalizers, and checkpoints remain alongside the progress file.

```bash
cat outputs/stage9/formal/progress.json
tail -20 outputs/stage9/formal_driver.log
cat outputs/stage9/formal/training.exit
```

`training.exit` equal to `0`, `status` equal to `training_complete`, 12 completed runs, and a selection seal indicate training and validation reload completion. No test evaluation is included in this script. A nonzero code is a failure to inspect; an absent marker is not proof that a process is still alive. Session detachment protects against SSH disconnect, but server shutdown, storage failure, or platform termination can still interrupt training. The instance is not shut down automatically.

After the training-complete notification, verify the source/configuration/checkpoint records and all seeds, then evaluate the sealed models and both non-learning baselines on D0 test/D1. Publish the episode-level comparisons and limitations before closing Stage 9. GRAIL has not been run; its original proposal scope remains an explicit decision for the stage closeout. Stage 10 has not started.
