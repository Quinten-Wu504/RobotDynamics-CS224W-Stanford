# Predictive pretraining pilots

Stage 7 compares ordinary and canonical-frame Temporal GNNs with and without predictive pretraining. The original 30-epoch supervised runs remain reference points. For each architecture, run JEPA, direct future-state pretraining, and a 40-epoch supervised control. This is a numerical-state adaptation of joint-embedding prediction, not a reproduction of image I-JEPA or a pretrained visual foundation model.

## Data and causality

Use the same nested 80 D0 training episodes, 100 validation episodes, seed 0, four-frame history, and 1/5/10-step windows as stages 4–6. Normalization uses only those 80 training episodes. Pretraining sees no additional episodes. D0 test and D1 remain reserved for stage 8.

The online branch receives history through `t` and planned actions `[t:t+k]`. The target branch observes four frames ending at `t+k`, including overlap with context at short horizons. Its planned-action and horizon-code fields are all zero: no action at or after `t+k` is supplied. Future history is training-only target information and is never passed to the downstream predictor. Tests perturb future states independently of context and exclude states after the target endpoint.

## JEPA objective

The existing spatial encoder, two shared message rounds, and per-node GRU produce 64-dimensional node latents. A shared 64–128–64 predictor predicts the EMA target encoder's future latents for the two dynamic nodes. Static nodes provide context but are excluded from the prediction loss and collapse statistics.

Target weights start as an online-encoder copy, receive no gradients, and update after every pretraining step as `target = 0.99*target + 0.01*online`. The target has no physical prediction head. The loss is the sum of:

- MSE between feature-wise layer-normalized predicted and stop-gradient target latents;
- per-node, across-sample variance penalty `mean(relu(0.2 - sqrt(var + 1e-4)))` on online latents;
- squared off-diagonal covariance, summed over both dynamic nodes and divided by `2*64`.

All three weights are 1. These variance/covariance terms are fixed before running pilots to discourage collapse; this is explicitly **regularized JEPA**, not pure latent MSE. The design draws on [I-JEPA](https://arxiv.org/abs/2301.08243) and [VICReg](https://arxiv.org/abs/2105.04906), with a different state/action task and architecture.

For the canonical variant, both online and target geometric inputs use the same observed peg frame. Latent prediction and regularization operate on invariant coordinates; the existing output mapping restores world-frame deltas during downstream prediction. Full-chain yaw/translation checks remain required after fine-tuning.

## Controls and downstream protocol

Pretrain for 10 epochs, then fine-tune the entire encoder and physical head for 30 epochs. At the phase boundary, reset the physical head to its original seeded weights and create a fresh AdamW optimizer. JEPA's latent predictor and target encoder are not used at inference. No frozen-encoder probe is part of the main matrix.

Direct-state pretraining uses the same future-history observations: 132 normalized physical values, comprising four frames of end-effector position/rotation/finger/joint state and nut position/rotation. A shared 64–128–84 auxiliary head uses padding plus an explicit mask for the smaller nut target. Loss is mean squared error over the 132 valid values; it is not the downstream position-only loss. For the canonical variant those target fields are expressed in the same peg frame. After pretraining, discard this auxiliary head and reset the physical head/optimizer just as for JEPA.

The direct-supervision control trains the physical task continuously for 40 epochs, without a head or optimizer reset. Only the last 30 epochs are eligible for validation-based selection, matching the 30 selection opportunities of the pretraining methods. Thus every new run makes 5,480 updates over the same context-window order; JEPA/state split this into 1,370 pretraining and 4,110 fine-tuning updates. The direct control makes 5,480 supervised updates. All use batch 256, AdamW 0.001, weight decay 0.0001, and gradient clipping 1.0.

Updates and context-data presentations are matched; FLOPs are not assumed equal. JEPA adds a target-encoder pass and a latent predictor, while direct-state pretraining uses an auxiliary reconstruction head. Record phase-specific update times, validation time, memory, inference and auxiliary parameter counts, and target-data exposure. Earlier baselines had only 4,110 updates. A win over those alone cannot establish a JEPA-specific benefit; the direct-state and longer-supervision controls are necessary. Shared GPU timing limits exact compute comparisons.

## Diagnostics and checks

For fixed training windows, log online/target latent standard deviations and covariance effective ranks before pretraining, after every pretraining epoch, and at the end. Compute these separately for each dynamic node, so node identity alone cannot disguise a constant representation. Flag gross collapse if either node's mean standard deviation is below 0.01 or effective rank below 2 after pretraining. These are limited screening thresholds, not proof of useful representations. Also record target/online parameter distance, losses, physical prediction errors, and downstream sensitivity to changed histories/actions. If collapse occurs, retain the failed run and diagnose it before treating the JEPA experiment as valid.

At the saved pretraining endpoint, also apply the same per-node screening thresholds to layer-normalized target and predictor outputs. Raw variance can hide a nearly constant normalized direction, which matters because the predictive loss uses normalized latents. These checks are specified before the full pilots. Input sensitivity uses fixed subsets: zero planned actions or permute past histories across samples while retaining the current state. Report output changes and physical error changes as diagnostics, not as physical OOD performance.

The small configuration tests pretraining, transition, and fine-tuning on two trajectories. Check uninterrupted versus interrupted/resumed runs, including EMA, predictor, optimizer, sampler, and RNG state. At final evaluation, reconstruct a fresh physical model from the chosen checkpoint and require identical predictions. Independent evaluation and coordinate checks use the same saved checkpoint.

## Commands

Use the existing Linux stage 5 environment; no new packages are required.

```bash
CACHE=outputs/stage3/square_d0.hdf5
.venv/bin/python -m pytest -q
.venv/bin/python -m robot_dynamics.train_pretraining "$CACHE" --config configs/pretraining_check.yaml --model temporal_gnn --method jepa --output outputs/stage7/check_jepa
```

After the small checks pass, run the six pilots sequentially:

```bash
for model in temporal_gnn canonical_temporal_gnn; do
  for method in jepa state supervised; do
    .venv/bin/python -m robot_dynamics.train_pretraining "$CACHE" --config configs/pretraining_pilot.yaml --model "$model" --method "$method" --output "outputs/stage7/${model}_${method}"
    .venv/bin/python -m robot_dynamics.evaluate_learning "$CACHE" --checkpoint "outputs/stage7/${model}_${method}/best.pt" --output "outputs/stage7/${model}_${method}_reload.json"
  done
done
```

Resume a stopped job in its original directory with `--resume <output>/last.pt`. Checkpoints include both pretraining modules, phase progress, original head weights, optimizer and RNG state, source/data hashes, and accumulated logs. `pretrained.pt` preserves the phase-boundary state; only downstream validation selects `best.pt`. A stopped partial run does not emit final metrics. Stage 8 is not started by these commands.
