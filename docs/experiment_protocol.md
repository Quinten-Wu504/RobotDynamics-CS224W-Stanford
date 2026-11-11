# Experiment protocol

## Question and scope

Does graph structure improve action-conditioned prediction of robot and object motion? Does geometric equivariance help with limited data or distribution shift? Does JEPA add anything beyond direct supervision or future-state pretraining?

The initial task is MimicGen **core** Square D0/D1, not the small source-demonstration dataset. Models predict motion of the end effector and square nut. Peg and table geometry provide context; their static positions do not contribute to the main error. No closed-loop success claim will be made from offline prediction error.

The stage 2 audit established the field layout, controller semantics, fixture poses, and 20 Hz timing for the pinned release. The [configuration](../configs/square.yaml) records these checks; the [field table](state_fields.md) defines the cache. No prediction model has been trained.

## Samples and timing

For an episode with states indexed from 0, use:

```text
inputs:  state[t-H+1:t+1], action[t:t+k]
targets: position[t+k] - position[t]
```

Slice ends are exclusive. The initial history length is H=4; any later change is selected on validation data and shared across temporal baselines. The current-state baseline uses H=1 and is reported separately. Known future actions are part of the task definition, not inferred labels. History-only forecasting would be a separate task.

First verify k=1. Then use 0.25 s and 0.5 s horizons. Read control frequency from environment metadata and check it against state timestamps or simulator stepping. Convert seconds to steps only when the product is an integer; otherwise stop and document the mismatch instead of rounding silently. The 20 Hz controller default is not evidence of the dataset frequency.

Only generate windows with a complete history, action sequence, and future state in the same episode. Verify that action[t] advances state[t] toward state[t+1]. Compare `obs`, `next_obs`, and restored states where available. If the terminal future state is absent, discard that window; do not duplicate a final frame or generate a label by extrapolation.

## Physical fields

The following representation was established by the data audit. The cache uses named physical fields rather than treating the two distributions' different `object` vectors as interchangeable.

| Field | Meaning and units | Source / check |
| --- | --- | --- |
| End-effector position | World xyz, m | `robot0_eef_pos`; confirm against simulator site |
| End-effector orientation | Hand-body-to-world rotation matrix | `robot0_eef_quat`, xyzw; not the grip site's orientation |
| Gripper state | Two finger joint positions, m | `robot0_gripper_qpos`; verify robot and gripper type |
| Nut position and orientation | World xyz in m, body-to-world rotation | `object[:, :3]` and xyzw `object[:, 3:7]`, checked against restored body pose |
| Target peg pose and size | World pose, geometry in m | Episode XML and model body/geom transforms; `peg1_pos` when present |
| Table surface | World plane point, unit normal, dimensions in m | Arena model and transformed geometry |
| Action | Controller command plus gripper command | `actions` and `env_args`; interpretation below |

Keep pose and shape information: the square nut's center alone cannot describe insertion geometry. Preserve the raw quaternion convention in the extraction record; use rotation matrices in the common model interface to avoid sign ambiguity. MuJoCo XML quaternion ordering and observation quaternion ordering must not be assumed identical.

Square D1 randomizes the peg through XML and adds a peg observation. Restore each episode using its own model and state, then read world coordinates. Do not pad missing D0 fields with zeros or assume a shared `object` vector layout. Include the target peg only; objects moved out of the scene by the task implementation are not active task nodes.

Velocity, if added, uses a backward difference from available history with the measured time interval. No future frame, contact, or task-phase label becomes a current input. Raw simulator joint state is retained for recovery but is not added to only one model family.

## Actions

Both released datasets specify six OSC_POSE commands followed by a gripper command for a Panda with fixed impedance. Their metadata confirms input range [-1, 1], translation scaling ±0.05 m, rotation scaling ±0.5 rad, and delta control. The cache retains both raw commands and scaled increments.

Inspect `env_args` for robot, controller type, impedance mode, `control_delta`, limits, and frequency. Trace the installed controller's scaling and rotation composition. Confirm world-frame translation and orientation-increment axes before applying geometric transformations. For the pinned Panda gripper, the source maps negative action to opening and positive action to closing; verify this is the dataset's gripper. Pose commands are controller targets, not next-frame robot joint positions.

Preserve raw actions alongside decoded commands in the eventual cache. A metadata mismatch blocks training until resolved; it must not be corrected by an undocumented default.

## Splits and model selection

- Split complete D0 episodes into 80% train, 10% validation, and 10% test using split seed 224. Record the actual integer counts and episode IDs before creating windows. Check for duplicate or shared source trajectories when metadata permits.
- D1 is reserved for final testing. Structural metadata checks needed for correct decoding are allowed; D1 prediction metrics cannot guide tuning. Any later D1 development use requires a new documented partition with an untouched final subset.
- Create nested 10%, 25%, and 100% subsets of the D0 training pool from one fixed episode ordering. Every model uses the same manifests. Log the rounding rule and actual counts.
- Fit normalization only on the training episodes available to the run. Validation/test statistics are excluded. Shared isotropic scaling for geometric vectors must preserve any claimed equivariance.
- Select checkpoints and hyperparameters on D0 validation. Freeze the formal configurations before evaluating held-out sets. Formal seeds are 0, 1, and 2; smoke tests and pilot runs are labeled separately.

## Comparisons

Start with position-hold and causal constant-velocity baselines, then current-state MLP, history MLP, GRU, single-frame GNN, and temporal GNN. All paired comparisons get the same physical fields, action sequence, history, and relative geometry. Start the graph with end-effector/gripper, nut, peg, and table context; record edge directions and construction. Distance is not a contact label.

Later compare ordinary/equivariant temporal GNNs with and without JEPA. Record parameter counts, optimizer steps, tuning trials, samples seen, elapsed time, and peak memory. JEPA is evaluated through the same physical prediction task, with future-state pretraining and comparable-compute supervised controls. A low latent loss is not a physical result.

Equivariance claims require a declared transformation group and tests of the whole prediction path, including time modules, normalization, actions, gravity, surface normals, and orientations. Coordinate consistency and D1 generalization are separate measurements.

## Metrics and reporting

For each entity and horizon, reconstruct predicted position and compute `100 * ||predicted_position - true_position||_2` in cm. Average valid windows within each episode, then average episodes equally. Report the number of episodes and windows. Use episode bootstrap intervals (95%, 2,000 resamples, fixed bootstrap seed 224) for each seed, plus the mean and standard deviation across training seeds. Do not treat overlapping windows as independent observations.

Report end effector and nut separately, by horizon, and by motion/interaction segment where labels can be supported. Motion thresholds and any heuristic phase definitions must be derived from training data and frozen before testing. Contact or grasp metrics require credible labels and imbalance-aware measures; they are not part of the initial output.

One-step prediction conditioned on true state, direct multi-step prediction, and free rollout are distinct evaluations. The initial task is direct prediction with known actions. Orientation, if later predicted, is scored with a rotation-space metric, not quaternion component error.

Every result records source commit, configuration, dataset checksum/version, manifests, seed, checkpoint selection, units, horizon, and failures. Report all formal seeds, including failed runs. Configurations and scripts do not count as completed experiments.

## References checked for stage 1

- [MimicGen installation and pinned simulator versions](https://mimicgen.github.io/docs/introduction/installation.html)
- [MimicGen dataset families](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html)
- [robomimic HDF5 structure](https://robomimic.github.io/docs/datasets/overview.html)
- [Square D0/D1 source](https://github.com/NVlabs/mimicgen/blob/c0fbff22ba40060f05a917a80817f3f5c20731e7/mimicgen/envs/robosuite/nut_assembly.py)
- [Pinned OSC defaults](https://github.com/ARISE-Initiative/robosuite/blob/b9d8d3de5e3dfd1724f4a0e6555246c460407daa/robosuite/controllers/config/osc_pose.json)
- [Pinned OSC implementation](https://github.com/ARISE-Initiative/robosuite/blob/b9d8d3de5e3dfd1724f4a0e6555246c460407daa/robosuite/controllers/osc.py)
- [Pinned Panda gripper](https://github.com/ARISE-Initiative/robosuite/blob/b9d8d3de5e3dfd1724f4a0e6555246c460407daa/robosuite/models/grippers/panda_gripper.py)
