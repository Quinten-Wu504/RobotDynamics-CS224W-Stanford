# Coordinate-equivariant pilots

Stage 6 uses a known scene frame to make the complete predictor equivariant. This is a canonical-frame construction around the existing Temporal GNN, not an EGNN message layer. A canonical GRU receives the same treatment so that a coordinate-processing benefit is not automatically attributed to graphs. No parameter is added to either backbone.

## Supported transformations

The declared group is proper yaw rotation about the gravity axis and arbitrary translation: `p' = Q p + t`, with `Q z = z` and `det(Q)=1`. This is a passive re-expression of the entire scene, including the robot base and controller axes, in another coordinate frame. It does not assert that moving objects relative to a fixed robot produces identical dynamics. Pitch, roll, reflections, changed gravity, and independent object motions are outside this stage's claim.

World positions (including fixture geometry) rotate and translate. Rotation matrices transform by left multiplication. Displacements, causal velocities, normals, gravity, and both translation/axis-angle action vectors rotate without translation. Gripper commands, robot/finger joints, local box half-sizes, horizon indicators, and time remain unchanged. Gravity is unchanged by the declared yaw group. Existing controller source checks establish world-axis pose increments and isotropic per-axis translation/rotation scales; see [field conventions](state_fields.md).

Rotated action components can exceed the original coordinate-aligned [-1,1] box. Coordinate tests do not clip them or execute them in the simulator: clipping would change the command and destroy the passive-transformation test. The original command norm and physical vector are preserved under basis change. This is not a claim about feasibility of a new action in the original controller frame.

## Construction and normalization

Let `o` and `F` be the peg body position and its local-to-world rotation. Every position is represented as `F.T @ (p-o)`, orientation as `F.T @ R`, and vector/action as `F.T @ v`. The same static frame is used across the entire history. The unchanged spatial GNN and per-node GRU operate in these invariant coordinates. The prediction head emits local deltas, which are mapped back with `F`. The GRU control uses precisely the same wrapper.

Under the declared transformation, `o' = Q o+t` and `F' = Q F`, so canonical inputs are unchanged and output deltas rotate by `Q`. Adding the current position gives an equivariant future position. The labeled peg's observed orientation supplies a unique frame; no PCA, frame search, or data-dependent sign choice is used. Node identities and ordering are unchanged by coordinate transformations. This is related to frame-based equivariant network design, but does not implement the averaging algorithm of [Puny et al.](https://arxiv.org/abs/2110.03336).

The original training-only normalizer is kept fixed during transformation checks. Its common isotropic position scale commutes with rotation; its fixed center cancels when subtracting the normalized peg position. Scalar joint normalization is unchanged. Position deltas are decoded using that same scale and the original current-state anchor. This checks preprocessing, temporal processing, prediction head, and physical-unit reconstruction, not just a spatial layer.

Canonicalization discards the peg's absolute world-frame pose. With the robot base absent from the model inputs, this can remove information predictive of robot-relative reachability. Thus better coordinate consistency need not mean better physical generalization. This limitation is part of the hypothesis, not evidence that D1 is solved.

## Fixed experiment

Two models, `canonical_temporal_gnn` and `canonical_gru`, use the stage 4/5 split and optimizer: 80 training episodes, 100 validation episodes, seed 0, four history frames, horizons 1/5/10, 30 epochs, 4,110 updates. Two-episode fitting checks use 200 epochs first. Parameter counts match their original counterparts (126,339 and 120,390). No augmentation, new physical feature, extra training data, or post-result parameter search is planned.

Full-chain coordinate checks use 96 validation windows spread across episodes and horizons, with seed 606 and 16 transformations: pure translation, pure yaw, and combined random yaw/translation up to 0.5 m per axis. The fixed acceptance bound for the canonical models is **2e-5 m (0.002 cm)** maximum absolute future-position discrepancy. Random-weight tests precede training; trained-checkpoint checks follow the pilot. Ordinary models are diagnostic controls, not required to satisfy this bound. Transform/inverse, causal velocity, gravity, and unsupported-reflection checks are included.

Report original D0 validation prediction errors separately from coordinate-consistency error. A coordinate-transformed validation window is not a new physical rollout or an OOD dataset. D1 and held-out D0 remain reserved for stage 8; physical distribution-shift performance is explicitly unmeasured here. Exact continuation and independent reload checks follow the existing protocol.

A separate physical-placement diagnostic uses real D0 validation trajectories. Before inspecting grouped errors, define the convex hull of the 80 training episodes' initial nut XY positions in the peg frame. Group validation episodes by whether their initial placement lies inside or outside that hull, and report the same episode-averaged errors and counts for all four models. This tests finite-training-set placement coverage, not a new data-generating distribution; an outside-hull episode must not be labeled D1 or a general OOD result. The hull is fitted to training positions only and has no error-selected threshold.

## Commands

Use the stage 5 Linux environment and the existing numeric cache:

```bash
CACHE=outputs/stage3/square_d0.hdf5
for model in canonical_temporal_gnn canonical_gru; do
  .venv/bin/python -m robot_dynamics.train_learning "$CACHE" --config configs/learning_fit.yaml --model "$model" --output "outputs/stage6/fit_$model"
done
```

After inspecting fit results:

```bash
for model in canonical_temporal_gnn canonical_gru; do
  .venv/bin/python -m robot_dynamics.train_learning "$CACHE" --config configs/learning_pilot.yaml --model "$model" --output "outputs/stage6/pilot_$model"
  .venv/bin/python -m robot_dynamics.evaluate_learning "$CACHE" --checkpoint "outputs/stage6/pilot_$model/best.pt" --output "outputs/stage6/pilot_${model}_reload.json"
done
.venv/bin/python -m robot_dynamics.check_resume "$CACHE" --models canonical_temporal_gnn canonical_gru --output outputs/stage6/resume_checks
```

Resume an interrupted run in its original directory with `--resume <output>/last.pt`. No stage 7 work is included.

After the pilots finish, compare original-space predictions and run full-chain coordinate checks:

```bash
.venv/bin/python -m robot_dynamics.analyze_equivariance "$CACHE" --output outputs/stage6/analysis
.venv/bin/python -m robot_dynamics.plot_equivariance outputs/stage6/analysis/analysis.json
.venv/bin/python -m robot_dynamics.analyze_support "$CACHE" --output outputs/stage6/analysis/physical_support.json
```

Analysis reloads the original GRU and Temporal GNN from their stage 4/5 output directories, verifies their recorded source and complete prediction hashes, and does not retrain them. All four models must share the exact pilot configuration, normalizer, split, and update count. Figures can also be regenerated from the exported analysis JSON using a base Python environment without torch or a GPU.
