# Splits, windows, and baseline evaluation

The stage 3 evaluator uses D0 validation only. D0 test and all D1 episodes are reserved for the frozen formal experiments. No model or threshold is selected using their prediction errors.

## Split and normalization

Sort D0 IDs numerically, permute with NumPy PCG64 seed 224, then take 800 training, 100 validation, and 100 test episodes. The training fractions are prefixes of that same training order: 80, 200, and 800 episodes. Integer counts use floor, with the D0 remainder assigned to test. The manifest records all IDs, lengths, raw-data hashes, and cache hashes. Exact duplicate checks from stage 2 found none; shared source-demonstration ancestry remains unavailable.

Fit one normalizer for each training subset. Pool the end-effector, nut, peg, and table-surface positions once per training state. Use one common xyz centre and one scalar RMS scale across all coordinates. Transform positions as `(p-centre)/scale`, geometry sizes and target displacements as `value/scale`; keep rotations, normals, and bounded raw actions unchanged. Finger and arm joint coordinates use per-channel training means and standard deviations. Constant channels use scale 1 and are recorded. This convention avoids axis-dependent geometric scaling; full model equivariance has not been tested.

The two baselines operate in physical coordinates and need no fitted normalization. Normalizers are produced for later learned models. Their IDs and source hashes are saved so normalization cannot be silently reused across different training subsets.

## Windows and predictions

For H=4 and k in {1, 5, 10}, valid anchors satisfy `H-1 <= t < N-k`. A sample contains history `[t-H+1:t+1]`, episode fixture geometry, planned actions `[t:t+k]`, and position-delta targets `p[t+k]-p[t]`. No window crosses an episode boundary, and the final action without a recorded successor is never used as a label. Window input fields are shared by model families; simulator recovery state is excluded.

- Position hold: `p_hat[t+k] = p[t]`.
- Constant velocity: `p_hat[t+k] = p[t] + k * (p[t]-p[t-1])`.

Velocity uses only the last observed displacement. The uniform 0.05 s interval cancels in this formula. Both baselines ignore planned actions and use identical H=4 eligible anchors. They are direct predictions conditioned on recorded histories, not recursive free rollouts. They are deterministic and have no training seeds, optimizer, or checkpoint.

## Aggregation and motion groups

Compute Euclidean position error in cm separately for end effector and nut. Average windows within an episode, then give episodes equal weight. Bootstrap episodes 2,000 times with seed 224 for percentile 95% intervals. Also report paired constant-velocity-minus-hold differences using the same episode ordering. Empty entity/segment groups do not contribute an episode; counts are reported explicitly. Overlapping windows are never treated as independent bootstrap units.

For each entity, define a low-motion threshold from the 25th percentile of past one-step speeds in the full 800-episode training pool, with a numerical floor of 1e-6 m/s. At prediction time, classify using `||p[t]-p[t-1]|| / dt`. Report all, low-motion, and moving segments. These labels describe recent motion; a low-motion segment can move immediately afterward. They are not contact or task-phase ground truth. Thresholds and rules are fixed before reading validation errors, and the two baselines share them.

## Commands

Run from the existing AutoDL checkout. Copying the 133 MB D0 cache to the local data disk reduces small-read overhead; the input checksum is still verified. Outputs are protected against overwrite. For a repeat run, use fresh output paths. Reuse the committed manifest for comparisons rather than generating a different split.

```bash
cd /root/autodl-tmp/RobotDynamics-CS224W-Stanford
mkdir -p outputs/stage3
cp /root/autodl-fs/RobotDynamics-CS224W-Stanford-data/cache/square_d0.hdf5 outputs/stage3/square_d0.hdf5
.venv/bin/python -m robot_dynamics.splits --output outputs/stage3/manifest.json
.venv/bin/python -m robot_dynamics.normalization outputs/stage3/square_d0.hdf5 --manifest outputs/stage3/manifest.json --output outputs/stage3/normalization.json
.venv/bin/python -m robot_dynamics.baselines outputs/stage3/square_d0.hdf5 --manifest outputs/stage3/manifest.json --normalization outputs/stage3/normalization.json --output-dir outputs/stage3/evaluation
.venv/bin/python -m robot_dynamics.plot_baselines outputs/stage3/square_d0.hdf5 --metrics outputs/stage3/evaluation/metrics.json --output-dir outputs/stage3/figures
```

`robot_dynamics.windows.window` provides the unnormalized sample interface; `normalize_window` applies a chosen training normalizer. The evaluator checks the first and last eligible real-data windows for every validation episode and horizon, including target inverse scaling and action indices. Unit tests cover split overlap, nested subsets, future-state isolation, boundaries, unit conversion, constant velocity, normalization provenance, and equal episode weighting.
