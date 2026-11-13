# GRAIL pickup forecasting

## Scope

This Stage 9 extension applies the scene-graph abstraction to a Unitree G1 pickup subset. It is history-only forecasting with models trained from scratch on GRAIL. No Square/Stack weights are transferred. It is not zero-shot transfer, an action-conditioned world model, an RL experiment, or a closed-loop control result. The supervised graph comparison is the question here; JEPA and equivariance remain the separate Square experiments.

The official release supplies robot and object trajectories. The inspected robot files contain `hand_action_left/right`, but no complete executable body action stream. Neither hand commands nor reconstructed future reference poses enter this experiment. Contact-point dictionaries and placeholder `smpl_joints` are also excluded. No contact or grasp labels are inferred from proximity.

## Data and split

- Source: [NVIDIA GRAIL](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL), revision `1ef0e1c0319f79c927941a5c1828f635837a2f30`, `pickup_table`.
- The pinned category contains 2,991 motions over 685 named assets. Of these, 629 assets have at least two motions. NumPy's generator with seed 224 permutes the sorted eligible assets; the first 120 are retained, with two motions sampled without replacement per asset. The exact names are in the [manifest](../reports/stage9/grail/manifest.json).
- Split by asset before window creation: 84 training assets / 168 motions, 18 validation assets / 36 motions, 18 test assets / 36 motions. Each motion has 250 frames at 25 Hz. These are asset-disjoint splits within tabletop pickup, not category-disjoint or a MimicGen D0/D1 shift.
- All 720 source files are checked against the pinned Hugging Face LFS SHA-256 or Git blob hash before loading. Robot/object source keys and frame counts must agree; nonfinite values, incompatible shapes, nonunit quaternions, and exact duplicate numerical trajectories cause preparation to fail.
- There are 218 distinct source keys among the 240 exported motions. Repeated keys stay within the same asset and split; this dependence is retained. Test uncertainty will resample assets, keeping their two motions together, jointly with training seeds. Overlapping windows will not be treated as independent observations.
- Raw files and the 17.9 MB numerical HDF5 cache stay outside Git. The manifest records per-file hashes and the cache hash. Only code, small audit records, plots, and results are published. This project uses the release for noncommercial course research; upstream trajectory licensing is CC BY-NC 4.0. See the pinned dataset card for the separate asset/checkpoint terms. This experiment downloads neither assets nor upstream model checkpoints.

## States, inputs, and targets

Coordinates are the release's world-frame positions in meters. Public-release quaternions are xyzw, as specified by the [upstream visualizer](https://github.com/NVlabs/GRAIL/blob/main/grail/visualization/prepare_vis_shard.py); they are converted to nine rotation-matrix entries. The static table position, orientation, and size come from each motion's metadata, not an assumed global table height. No simulator or kinematic reconstruction is needed for the three recorded anchors used here.

The nodes are robot root, manipulated object, and table. Each node has 73 features: position (3), rotation matrix (9), backward-difference linear velocity (3), 29 body and 14 hand joint positions (43, on the robot node only), table size (3, on the table node only), node type (3), and relative positions to all three anchors (9). Typed zero blocks represent inapplicable features, not missing data. All models receive exactly this packing. Hand/wrist Cartesian positions, mesh shape, joint velocities, and contact features are not included; the root anchor is a coarser representation than a full articulated robot graph.

The history contains eight frames, spanning 0.28 s. The first allowed anchor is frame 8, so every included history velocity has a real preceding frame. Inputs never use frame `t+1` or later. Targets are root and object displacements at 1, 6, and 12 frames (0.04, 0.24, and 0.48 s); each window includes a three-entry horizon indicator. There are 118,776 training windows and 25,452 validation windows. Horizons use all their valid anchors, with no padding or windows crossing trajectories. One model predicts all three horizons. The table is an input, never a prediction target or part of the primary metric.

Feature means/stds and six displacement scales are fitted only on training trajectories, with a 1e-4 numerical floor. The two speed thresholds are the training-frame 25th percentiles, floored at 1e-6 m/s. Static/moving evaluation groups use the current backward-difference speed, not a future displacement. Validation selection uses physical errors decoded in float64 against the original cached displacements.

## Models and training

The [frozen matrix](../configs/grail_protocol.yaml) contains history MLP, GRU, Temporal GNN, and Temporal GNN with messages disabled, each with seeds 0/1/2: 12 runs. Architectures are fixed before test evaluation. All runs use the full training split, 40 epochs, batch size 512, AdamW at 1e-3, weight decay 1e-4, gradient clipping at 1.0, float32 and deterministic PyTorch kernels. Select the lowest episode-macro validation position error among epochs 11–40. No hyperparameter search or early stopping is planned.

The graph has all six directed non-self edges, two message-passing steps per historical frame, and a shared node-wise GRU. Messages see both endpoint embeddings and their relative positions. The disabled-message control still computes and zeros messages: it preserves nominal parameters and most arithmetic, but its message parameters have no effective influence. Each node still receives its explicit relative-coordinate features. There is no claim of geometric equivariance. The implementation batches this small fixed graph with dense tensor operations rather than adding another graph library interface.

MLP and GRU flatten the same three-node features and receive the same horizon indicator. Parameter counts and actual optimizer steps, times, memory and selected epochs are recorded per run. The parameter counts are 119,206 (MLP), 129,606 (GRU), and 119,427 (both graph variants). The largest/smallest ratio is 1.087. Similar parameter counts do not imply equal runtime; report both.

## Evaluation after training

Training ends after 12 independent validation reloads and a checksum seal of all selected checkpoints. The training script never predicts the test split. Test evaluation begins only in the subsequent closeout after the training-complete notification.

The primary metric equally averages two entities and three horizons, first averaging each motion's windows. Report all seeds, seed mean/sample SD, and paired 95% intervals from 2,000 joint seed/asset bootstrap draws with seed 224. Three predeclared contrasts compare GNN against MLP, GRU and messages-disabled. The intervals are descriptive without a multiple-comparison correction. Asset averaging and motion averaging coincide here because every asset contributes two equal-length motions.

Also report each entity/horizon, low-motion/moving subsets, and hold-position and causal constant-velocity baselines on the identical windows. Non-learning baselines need no training. Include prediction-versus-reference trajectory plots using the first two test assets in sorted order, not selected favorable examples. The resulting numerical/trajectory demonstration covers this Stage 9 item; it does not simulate the G1 or measure successful pickup. Any later changes prompted by test results must be marked exploratory.

## Execution and interruption

The driver uses an exclusive lock, an independent session, ignored SIGHUP, closed terminal input, and file-based logs. It waits until the single GPU has no compute process and at least 8 GB free before each run. Existing jobs are never terminated. Every epoch atomically writes `last.pt`; `best.pt` is saved according to the frozen validation rule. Rerunning the same command resumes interrupted runs and validates completed ones. Changed code, data, normalizer, or configuration blocks resume.

No runtime ceiling was requested. This does not promise immunity to instance shutdown, SIGKILL, storage failure, or resource exhaustion; the process survives ordinary SSH disconnection, and saved checkpoints support recovery. Failures stop the queue with a nonzero exit record rather than silently skipping a seed. Stage 10 remains unstarted.

## Completed run

All 12 runs completed 40 epochs, followed by 14 held-out evaluation sets and two exact full error-array rechecks. The [results](../reports/stage9/grail/README.md) include all seeds, bootstrap intervals, motion groups, costs, learning curves and the four preselected trajectory examples. GNN improves over GRU but does not improve over MLP or messages disabled on the mean; the latter two intervals include zero. The original selection rule was retained despite all six graph runs selecting its lower boundary, epoch 11. No test-driven retraining or checkpoint reselection was added. Stage 9 is in final review, and Stage 10 has not started.
