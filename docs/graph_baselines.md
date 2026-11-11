# Scene graph pilots

Stage 5 compares a single-frame GNN with current-state MLP, and a temporal GNN with history MLP and GRU. It reuses the 80 training episodes, 100 validation episodes, seed 0, four-frame history, three horizons, and all optimizer settings in `configs/learning_pilot.yaml`. Each run has 30 epochs and 4,110 updates. Two-episode fitting checks use the existing 200-epoch configuration first. No test-set evaluation or hyperparameter search is planned here.

## Graph representation

The four nodes are end effector, square nut, peg, and table. Their identity travels with the node during reordering; output is restored to end-effector/nut order. Each node has 52 features, with zero padding for inapplicable fields:

| Slots | Meaning |
| --- | --- |
| 0–2 | World position (table uses surface position) |
| 3–11 | Flattened rotation matrix (table uses its geometry rotation below) |
| 12–14 | Fixture geometry position |
| 15–23 | Fixture geometry rotation |
| 24–26 | Fixture half-size |
| 27–29 | Table surface normal |
| 30–38 | Two gripper positions and seven robot joints, on the end effector |
| 39–42 | Four-way node type |
| 43–45 | Nut minus end-effector position, on the end effector |
| 46–48 | Peg minus nut position, on the nut |
| 49–51 | Peg minus end-effector position, on the end effector |

This packing retains every physical input of stage 4, including its three explicit relative vectors. The same 70 padded planned-action values and three horizon indicators condition every node and the prediction head. Commands beyond the requested horizon remain zero. Fixture state is repeated over history; nothing from a future state is supplied.

The default graph has ten directed edges: both directions for every pair except peg–table. Edge features are normalized source-minus-destination positions. The alternative full graph adds both peg–table directions (12 edges). No radius or contact label is inferred. Self information travels through residual updates, so no self edge is needed.

`torch_geometric.data.Batch` constructs independent copies of the topology for each sample and frame. Topologies are cached by batch size and device; node features are rebuilt for every forward pass. Edge indices are offset by four nodes per graph. Aggregation uses mean incoming messages with deterministic PyTorch `index_add`; the implementation does not require compiled scatter extensions. See the [PyG 2.6.1 batching documentation](https://pytorch-geometric.readthedocs.io/en/2.6.1/advanced/batching.html).

## Models and controls

A shared two-layer node encoder maps features and actions to width 64. Two message-passing rounds share a message MLP and residual update MLP, also width 64. A shared 192-by-192 prediction head emits each dynamic node's three position deltas. The single-frame model uses only the latest state. The temporal model applies a shared 64-unit GRU to each node's four spatial encodings before prediction. It is an ordinary coordinate-dependent GNN; geometric equivariance is not claimed.

Two temporal controls are fixed before results: `temporal_no_messages` multiplies messages by zero, and `temporal_full_edges` adds the static fixture pair. The no-message control retains parameter count and message computation, but its message-network parameters receive zero gradient. It removes learned neighbor exchange; explicit relative vectors still convey relational information. These distinctions matter when interpreting its result. All temporal variants have identical nominal capacity; the single-frame model lacks the recurrent layer. Actual counts and costs are recorded.

Diagnostics check node permutation, batch isolation and individual/batched agreement, finite gradients, past-state dependence, and whether changing a neighbor's orientation affects end-effector predictions. Closing messages must remove that orientation pathway. The trained models receive the same numerical checks on real validation inputs.

Before inspecting results, grouped analysis is fixed to the stage 3 past-speed low/moving rule and a separate end-effector-to-nut distance rule: at most 5 cm at the anchor is `near_object`, otherwise `far_object`. Proximity is an interaction proxy, not a contact annotation. Groups use current/past state only, weight episodes equally, and report sample counts. All seven learned models are evaluated on identical groups; no threshold is selected using model errors.

## Running

Install `torch-geometric==2.6.1` in the existing Linux environment. From the repository root:

```bash
CACHE=outputs/stage3/square_d0.hdf5
for model in gnn temporal_gnn temporal_no_messages temporal_full_edges; do
  .venv/bin/python -m robot_dynamics.train_learning "$CACHE" --config configs/learning_fit.yaml --model "$model" --output "outputs/stage5/fit_$model"
done
```

After inspecting the fitting checks:

```bash
for model in gnn temporal_gnn temporal_no_messages temporal_full_edges; do
  .venv/bin/python -m robot_dynamics.train_learning "$CACHE" --config configs/learning_pilot.yaml --model "$model" --output "outputs/stage5/pilot_$model"
  .venv/bin/python -m robot_dynamics.evaluate_learning "$CACHE" --checkpoint "outputs/stage5/pilot_$model/best.pt" --output "outputs/stage5/pilot_${model}_reload.json"
done
.venv/bin/python -m robot_dynamics.check_resume "$CACHE" --models gnn temporal_gnn --output outputs/stage5/resume_checks
```

Resume any interrupted pilot in its original output directory with `--resume <output>/last.pt`. Stage 4 checkpoints retain their original source hashes and can be evaluated using their recorded commit. Stage 5 comparison additionally requires the current implementation to reproduce their complete saved prediction hashes before using them in group analysis.

The installed environment is recorded in `requirements/linux-stage5.lock.txt`. After the pilots and standalone evaluations finish, generate the comparison and grouped diagnostics with:

```bash
.venv/bin/python -m robot_dynamics.analyze_graphs "$CACHE" --output outputs/stage5/analysis
```

This command expects the stage 4 checkpoints in `outputs/stage4`. It verifies their hashes, recorded source revisions, split, configuration, normalization, and full prediction arrays before computing additional group metrics. It also checks trained graph models for batch/permutation consistency and neighbor dependence on eight validation samples spread across the fixed evaluation order.

Regenerate the published figures from the small committed result file, without data, checkpoints, or a GPU:

```bash
python -m robot_dynamics.plot_graphs reports/stage5/analysis/analysis.json --output outputs/stage5_figures
```
