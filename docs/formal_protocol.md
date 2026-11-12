# Formal Square experiments

Stage 8 freezes the model and evaluation choices before looking at held-out prediction errors. The six stage 7 pilots favored longer direct supervision over JEPA; this protocol retains JEPA and both geometries so the formal comparison can measure that result across data sizes and seeds. There is no further validation search, loss-weight search, or test-driven checkpoint selection.

## Matrix and training

The machine-readable protocol is [formal_protocol.yaml](../configs/formal_protocol.yaml). Seeds are 0, 1, and 2. Training subsets are the existing nested 80/200/800 D0 episodes. Every model uses a four-frame history and the same planned actions, with horizon-conditioned predictions for 1/5/10 steps from one model; there are no separate horizon-specific training jobs.

| Role | Models and methods | Data fractions | Seeds | Trained configurations |
| --- | --- | --- | --- | ---: |
| Core | Ordinary/canonical Temporal GNN × supervised/JEPA | 10%, 25%, 100% | 0, 1, 2 | 36 |
| Non-graph controls | History MLP and GRU, supervised | 10%, 25%, 100% | 0, 1, 2 | 18 |
| Auxiliary pretraining controls | Ordinary/canonical Temporal GNN, direct-state pretraining | 10% only | 0, 1, 2 | 6 |

There are **60 trained configurations and 84 training phases**: 36 single-phase supervised runs, 18 JEPA runs with two phases each, and six state-pretraining runs with two phases each. Hold and constant-velocity baselines require no training. With D0 test and D1 evaluation, the pipeline produces 124 prediction sets: 60 × 2 learned-model sets and 2 × 2 non-learning sets. All 60 configurations are run afresh, including seed 0; older pilots remain exploratory references rather than silently substituting for formal runs.

Direct supervision runs for 40 epochs. JEPA and state pretraining use 10 pretraining epochs plus 30 full fine-tuning epochs, resetting the physical head and optimizer at the boundary. Each method selects the lowest equal-episode/entity/horizon validation error from epochs 11–40, with the earliest epoch winning exact ties. MLP and GRU use this same selection window, optimizer, and 40-epoch budget. The existing architectures, regularized JEPA objective, EMA 0.99, batch 256, AdamW learning rate 0.001/weight decay 0.0001, gradient clipping 1, float32, and deterministic settings are unchanged. The common runner makes the non-graph controls' selection opportunities match the graph controls.

Normalizers use only the training subset for that run. Pretraining adds no episodes beyond that subset. Update counts and context presentations match within each fraction, not across fractions. FLOPs are not assumed equal; logs retain pretraining/fine-tuning updates, auxiliary and inference parameters, loading/validation/update times, and memory. Direct-state pretraining is an explicitly limited 10% auxiliary control; no claim about its larger-data behavior will be made.

## Test-use rules and analysis

The existing 100 D0 validation episodes select checkpoints. The 100 held-out D0 episodes and all 1,000 D1 episodes remain untouched by prediction evaluation until **every** formal run has finished and independently reproduced its validation predictions. The queue then seals all selected checkpoint hashes, normalizer hashes, validation prediction hashes, and the frozen matrix. The test evaluator refuses to run without that seal. Replacing a checkpoint or changing the protocol invalidates the evaluation. No test result feeds back into training or selection.

Report end-effector and nut errors in cm at all three horizons, equal-weighted by episode, along with the six-error mean. Retain every seed, seed mean/SD/range, and individual episode files. Predeclared paired comparisons cover graph versus MLP/GRU, canonical versus ordinary models, JEPA versus supervision/state pretraining, and the difference in JEPA's effect between geometries. The paired bootstrap uses 2,000 resamples with seed 224, jointly resampling training seeds and shared episode IDs. Three seeds give limited uncertainty estimates; overlapping intervals or small effects are not evidence of equivalence.

Motion groups retain the stage 3 thresholds fitted on the full D0 training split; this common grouping rule is independent of model fraction and prediction errors. The stage 5 near-object rule remains an end-effector/nut center distance of at most 5 cm at the anchor. These groups use current/past state, include episode/window counts, and are descriptive when sparse. Proximity is not a contact label. Non-learning baselines receive the same histories and ignore actions, as before.

The complete canonical prediction chain is checked on the same 96 validation windows and 16 gravity-preserving yaw/translation changes, with tolerance 2e-5 m. JEPA's raw and normalized representations are screened at the pretraining endpoint using the stage 7 thresholds. A failed diagnostic is retained and marks the generated analysis `complete_with_diagnostic_flags`; it is never silently dropped or relabeled as a passing run. Coordinate residuals are separate from physical D1 errors. All model predictions are decoded against raw physical target deltas, evaluated episode by episode to bound memory.

## Execution and recovery

The existing AutoDL environment is sufficient; no new packages are required. Run from `/root/autodl-tmp/RobotDynamics-CS224W-Stanford`. A resource-only check used all 800 training episodes but just one pretraining and one fine-tuning epoch: 348,868 windows, 72.7 s pretraining updates, 63.2 s fine-tuning updates, 64.4 s input preparation, and 1.15 GiB peak allocated CUDA memory. A full canonical JEPA run at this size is estimated at about 46 minutes. The entire pipeline is provisionally 12–16 hours plus variable system/I/O overhead; this is an estimate, not a completion deadline.

The finite full matrix is authorized to run to completion without an automatic wall-time kill. The shell process is detached from the SSH terminal. Locks prevent duplicate queues; each epoch saves a checkpoint atomically. Restarting the same command skips verified completed runs and resumes interrupted runs from `last.pt`. A failure stops the pipeline with a nonzero exit marker and preserved logs rather than skipping a seed. Server shutdown, disk failure, or platform termination can still interrupt the job; recovery preserves completed work.

The small workflow check uses two training episodes, one seed, three total epochs, and four D0 validation episodes as stand-ins for both evaluation paths. It never opens D0 test or D1 prediction evaluation:

```bash
bash scripts/run_stage8.sh outputs/stage3/square_d0.hdf5 outputs/stage3/square_d0.hdf5 outputs/stage8/workflow_check configs/formal_workflow_check.yaml --validation-only-check
```

Full execution, including training, sealing, both test sets, non-learning baselines, diagnostics, tables, and figures:

```bash
nohup setsid bash scripts/run_stage8.sh outputs/stage3/square_d0.hdf5 outputs/stage8/square_d1.hdf5 outputs/stage8/formal > outputs/stage8/formal_driver.log 2>&1 < /dev/null &
```

`outputs/stage8/square_d1.hdf5` is a byte-for-byte local copy of the already audited numeric cache at `/root/autodl-fs/RobotDynamics-CS224W-Stanford-data/cache/square_d1.hdf5`, not the raw release file. The evaluator checks its manifest hash. Monitoring does not require keeping the original SSH connection open:

```bash
tail -20 outputs/stage8/formal_driver.log
cat outputs/stage8/formal/progress.json
cat outputs/stage8/formal/pipeline.exit
```

An exit marker of `0` means the script finished; inspect `analysis/analysis.json` for diagnostic flags before treating the scientific checks as passed. The final tables, figures, and generated summary are under `outputs/stage8/formal/analysis`. Per-run metrics/checkpoints, independent validation reloads, test episode CSVs, the selection seal, and queue events remain alongside them. A nonzero exit is a failure requiring inspection; a missing exit marker means the pipeline has not recorded completion. Do not start another run in a new directory to bypass an interrupted queue.

Stage 8 completed all 60 configurations and 124 prediction sets. The [formal report](../reports/stage8/README.md) records the audited results, uncertainty, and limitations; the [launch record](../reports/stage8/launch.md) preserves the preflight checks. The background job has ended and did not start stage 9, change repository visibility, shut down the server, or provision additional resources. D0 test and D1 are now used tests; later test-inspired extensions must be reported as exploratory.
