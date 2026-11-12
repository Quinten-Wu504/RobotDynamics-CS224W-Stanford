# Stack replication protocol

Stage 9 will add one task after the completed Square study. The second task is MimicGen Stack D0/D1. This document records the design before loading its prediction errors. The finite comparison is authorized without a wall-time cap. Both releases have passed hash checks, full used-field audits, and six complete simulator restorations. All four models pass small-sample fitting, exact continuation, and independent reload checks. Resource and queue preflight checks precede formal launch. The background queue will stop after training and independent validation reloads, leaving test evaluation for the follow-up.

## Why Stack

Square's supervised Temporal GNN beats the tested MLP and GRU controls across all three training fractions on D0 and D1. That supports testing whether the finding survives a different manipulation task. It does not justify searching for settings that rescue JEPA: its extra benefit was not stable in Square.

Stack has two movable cubes instead of a movable nut and fixed peg. It provides a useful test of object-to-object interaction while retaining the MimicGen/robosuite ecosystem. The official implementation increases both cubes' initial x/y bounds from ±0.08 m in D0 to ±0.20 m in D1. The official registry supplies `core/stack_d0.hdf5` and `core/stack_d1.hdf5`. These are source-code facts, not an audit of the released files or proof of contact-specific prediction benefits.

Threading is another released candidate, but Stack more directly changes the fixed-fixture interaction in Square to two movable objects. Only Stack is proposed; there will be no task search based on test performance.

## Comparison

Train from scratch on Stack. This is a cross-task replication of model comparisons, not zero-shot transfer or fine-tuning Square checkpoints.

| Model | Seeds | Role |
| --- | --- | --- |
| History MLP | 0, 1, 2 | Non-graph feedforward control |
| GRU | 0, 1, 2 | Non-graph temporal control |
| Temporal GNN | 0, 1, 2 | Main graph model |
| Temporal GNN with messages closed | 0, 1, 2 | Within-architecture message ablation |

The scope is 12 trained configurations at the full training fraction, plus position hold and constant velocity. If both test distributions are available and pass audit, this gives 28 evaluation sets. It does not replicate Square's entire JEPA/equivariance/data-efficiency matrix. No second-task claim about those factors follows from this experiment.

Retain the four-frame history and planned future actions, with one model conditioning on 1/5/10-step horizons. Verify 20 Hz and the controller in the actual data before calling those horizons 0.05/0.25/0.5 seconds. Use the same optimizer settings and 40-epoch budget as the frozen Square protocol, selecting by validation error over epochs 11–40. Any architecture dimensions changed to accommodate Stack must be recorded with parameter counts. No test-driven hyperparameter search is planned.

The graph has end effector, cube A, cube B, and table nodes. Both cube poses belong in every history frame; cube B must not be treated as a fixed peg. Inputs must preserve orientations, gripper/joint state, geometry, and relative positions for every model. Predict all three movable entities, report each separately, and compute the primary score as the equal-episode mean across three entities and three horizons. The static table is excluded. Also report a moving-object breakdown so a nearly stationary cube cannot conceal poor manipulation predictions.

Use all directed non-self edges for this four-node graph, including cube–table pairs. Close learned messages in the ablation while retaining its inputs and nominal parameter count. Record effective parameter use and actual update time; nominal parameter matching does not mean equal capacity or FLOPs. Report any remaining explicit relational features in the ablation.

## Data and test boundaries

Pin release revision, download sizes and hashes before downloading. Audit every episode's field dimensions, finite values, controller metadata, timestamp spacing, pose conventions, XML geometry, and trajectory hashes. Confirm cube A/B identities against simulator restoration for several full trajectories in both distributions. Preserve original files and do not fill missing semantics with zeros.

The audited release has 1,000 D0 episodes, split 800/100/100 by episode using seed 224, and 1,000 D1 episodes. Fit all normalizers on training episodes only. D1 is final-test-only for prediction evaluation. Data integrity and state-restoration checks may inspect both files, as in Square; prediction errors must remain unopened until all selected checkpoint hashes are sealed.

Use held-out D0 and D1 for the predefined graph-versus-GRU/MLP and graph-versus-message-ablation comparisons. Retain every seed. Pair by episode and seed, report effect sizes and uncertainty, and show all entity/horizon means. Any new test-inspired analysis is exploratory. Existing Square tests remain used tests; Stack does not reset their status. Offline accuracy is not closed-loop control success.

## Implementation and resource gate

The current Square cache, window schema, graph packing, output heads, and evaluation labels explicitly assume two targets and a static peg. They cannot be reused unchanged. The Stack adapter supplies explicit cube histories and three-target outputs while preserving the existing Square paths and tests. Reuse training/checkpoint logic where its interfaces permit, without relabeling cube data as nut/peg fields.

Before a full matrix: validate field recovery, causal windows, normalization, equivalent model inputs, graph batching/message behavior, small-sample fitting, exact resume, and independent checkpoint reload. Time a bounded Stack resource probe using training/validation data only. Save a machine-readable frozen matrix and recovery commands before launching the formal queue.

Square full-data mean update times were 8.36 min for MLP, 17.81 min for GRU, and 36.90 min for Temporal GNN. Using graph cost also for the message ablation gives approximately `3 × (8.36 + 17.81 + 2 × 36.90) = 300 min` of updates. This is only a Square-based reference: Stack's window count, three-target adaptation, I/O, preparation, validation, and test evaluation have not been measured. It is not a five-hour end-to-end promise.

The user authorized the finite second-task matrix without a time limit and requested a detached training queue followed by a handoff. Measure costs before launch and retain all seeds. Use session detachment, queue locks, atomic epoch checkpoints, and resume validation; test the job across an SSH disconnect. Platform shutdown or storage failure can still interrupt it. Do not buy or resize an instance. No Stage 10 work starts during this stage.

GRAIL is not part of this single-task extension. Its state/action availability has not been checked, and permanent removal from the original proposal has not been approved. Resolve that scope decision explicitly before closing Stage 9; do not claim a GRAIL demonstration or transfer result.

## References checked during planning

- [Official dataset overview and release caveats](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html)
- [Official Stack implementation](https://github.com/NVlabs/mimicgen/blob/main/mimicgen/envs/robosuite/stack.py)
- [Official dataset registry](https://github.com/NVlabs/mimicgen/blob/main/mimicgen/__init__.py)
- [Completed Square results and costs](../reports/stage8/README.md)

The upstream links above identify the planning sources. Record the installed implementation revision and actual release metadata during the data audit; do not assume the moving upstream branch matches the pinned environment.

The Stack two-epoch resource probe measured 10.13/19.84/42.60/42.00 seconds per epoch for MLP/GRU/GNN/message-ablation, respectively, on 238,144 training windows. This projects 3.82 hours of updates; the training-only queue is expected to take roughly 4–5 hours including validation/reloads, subject to load. See the [launch and audit record](../reports/stage9/README.md) for checks and actual commands.
