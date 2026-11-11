# MLP and GRU pilots

Stage 4 uses the fixed stage 3 split. The pilot has 80 training episodes (the nested 10% subset), 100 D0 validation episodes, and seed 0. D0 test and D1 remain untouched. This is a limited development run, not the three-seed formal matrix.

## Inputs and models

Each history frame has 42 values: the shared physical fields from the state cache plus three relative position vectors (nut minus end effector, peg minus nut, peg minus end effector). Fixture geometry contributes 48 values. Raw planned actions are padded to ten commands, with all slots after the requested horizon set to zero. A three-value horizon code distinguishes 1, 5, and 10 steps. No action after the requested horizon is visible.

One model predicts six normalized position deltas, conditioned on the requested horizon. The same weights serve all three horizons. Normalized positions, sizes, scalar joints, and targets follow the stage 3 convention; statistics are fitted only to the episodes actually used by that run. Rotations and raw actions retain their original scale.

- Current-state MLP: last history frame and shared context; two 256-unit ReLU layers.
- History MLP: four flattened history frames and shared context; two 224-unit ReLU layers.
- GRU: four frames through a 96-unit single-layer GRU, followed by the shared context and two 192-unit ReLU layers.

Widths were selected before pilot results to keep parameter counts reasonably close. Exact counts are logged. All three receive the same current physical information and planned actions; only the temporal models receive preceding states. The current-state comparison is therefore an explicit history ablation.

## Training and selection

The two-episode fitting check uses 200 epochs, batch size 128, and training error for selection. It is a capacity/gradient diagnostic and does not report a validation result. Proceed to the 80-episode pilot only after checking that the models fit those trajectories.

The pilot uses 30 epochs, batch size 256, AdamW at 0.001, weight decay 0.0001, and gradient clipping at 1.0. Loss is mean squared normalized position-delta error over the pooled windows and horizons. Uniform window shuffling gives slightly more weight to shorter horizons because they have more eligible anchors. The batch order and optimizer-update count are identical across models. Best checkpoints minimize the validation mean of six episode-averaged errors (two entities × three horizons), in cm. Validation never contributes normalization statistics.

Inputs are prepared in memory and moved to the selected device. Logs separate preprocessing time from training/evaluation time, record optimizer steps, per-epoch timing, sampled process RSS, and peak PyTorch CUDA allocation. CUDA allocation excludes the driver context. Float32 and deterministic algorithms are used; cuDNN and TF32 matrix multiplication are disabled for the reproducibility checks. Timing is specific to this implementation and shared host.

## Commands

From the AutoDL checkout, using the existing local copy of the D0 cache:

```bash
CACHE=outputs/stage3/square_d0.hdf5
for model in current_mlp history_mlp gru; do
  .venv/bin/python -m robot_dynamics.train_learning "$CACHE" --config configs/learning_fit.yaml --model "$model" --output "outputs/stage4/fit_$model"
done
```

After the fitting checks pass:

```bash
for model in current_mlp history_mlp gru; do
  .venv/bin/python -m robot_dynamics.train_learning "$CACHE" --config configs/learning_pilot.yaml --model "$model" --output "outputs/stage4/pilot_$model"
  .venv/bin/python -m robot_dynamics.evaluate_learning "$CACHE" --checkpoint "outputs/stage4/pilot_$model/best.pt" --output "outputs/stage4/pilot_${model}_reload.json"
done
```

To resume an interrupted run in its original directory:

```bash
.venv/bin/python -m robot_dynamics.train_learning "$CACHE" --config configs/learning_pilot.yaml --model gru --output outputs/stage4/pilot_gru --resume outputs/stage4/pilot_gru/last.pt
```

Checkpoints are saved atomically after each epoch and contain model, optimizer, epoch, selection score, logs, and random-generator states. A partial epoch restarts from the last completed epoch. Resume checks model/configuration, input hashes, normalizer, and relevant source files. A fresh run refuses an existing directory. Checkpoints remain in AutoDL outputs and are not committed to Git; small metrics and logs are exported separately. Evaluation accepts only compatible checkpoints from this project's runs.

`configs/learning_resume_check.yaml` runs six epochs on two training episodes. Compare an uninterrupted run with a run stopped by `--stop-after-epoch 3` and resumed from `last.pt`; this checks continuation, not just model loading. Every training run also compares predictions from a fresh model loaded from the selected checkpoint.

References: [PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.5/notes/randomness.html) and [general checkpoints](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html). Exact agreement is checked in the tested software/hardware environment, not assumed across platforms or PyTorch releases.
