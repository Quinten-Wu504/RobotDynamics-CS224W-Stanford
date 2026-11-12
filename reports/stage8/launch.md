# Formal experiment launch

Archived launch record, written before the formal evaluations. See [the final report](README.md) for completion and results.

Stage 8 is **in progress**. The complete 60-configuration queue is being launched on the existing AutoDL instance as a detached process. This directory records the frozen plan and checks performed before launch; it contains **no formal D0-test or D1 prediction results**. Those results will be reviewed after the background pipeline finishes.

The [protocol](../../docs/formal_protocol.md) and [frozen matrix](frozen_matrix.json) define 36 core configurations, 18 MLP/GRU controls, and six small-data state-pretraining controls. Seeds are 0/1/2; the main matrix covers 80/200/800 training episodes. The 60 configurations contain 84 training phases. All three horizons share each trained model. Test evaluation begins only after all selections have been sealed, then produces 124 learned/non-learning prediction sets across held-out D0 and D1.

## Checks before launch

| Check | Result |
| --- | --- |
| Linux suite | 48 passed |
| Windows base suite | 30 passed; 8 skip entries for unavailable torch/PyG coverage |
| Lint, formatting, dependencies, shell syntax | Passed |
| Full small workflow | 8 models, 20 prediction sets, summaries and all three figure types completed |
| Exact interruption/resume | All 8 architecture/method combinations passed |
| Interrupted result finalization | All 8 combinations reproduced predictions and summaries without new training |
| Interrupted first startup | Incomplete output preserved before restarting that run |
| Duplicate launch | Rejected by the pipeline lock with exit code 3 |
| Replaying a completed queue | Selected checkpoints and training metrics unchanged |
| Formal test access during checks | None; four D0 validation episodes stood in for evaluation |

The final workflow used two training episodes, one seed, and three epochs. Its prediction errors and figures are **pipeline checks, not scientific model comparisons**. Both the initial and final workflow completed without errors; final checks used clean source commit `6f8f9a1`. The [verification record](verification.json) includes hashes, and [resume checks](resume_checks/checks.json) record every comparison. [Small-workflow artifacts](workflow_check_final/analysis/README.md) exercise the same reporting path used by the full script. The figures were visually inspected. Code review retained provenance, causality, finite-value, checkpoint, overwrite, and test-access protections.

## Resource measurement

A separate all-800-episode probe used one JEPA pretraining epoch and one fine-tuning epoch, exclusively on D0 training/validation. It processed 348,868 training windows. Input preparation took 64.4 s, pretraining updates 72.7 s, and fine-tuning updates 63.2 s; peak allocated CUDA memory was 1.15 GiB and sampled RSS 2.57 GiB. The [configuration](resource_probe.yaml) and [metrics](resource_probe/metrics.json) label this as a resource probe. Its validation error is not a formal result.

At that rate, a full canonical JEPA run with 800 episodes is about 46 minutes. The entire finite pipeline is estimated at roughly 12–16 hours plus variable I/O/system overhead. It runs sequentially on the existing RTX 4090 D. No extra instance, service, dependency, or larger matrix was added. The numeric D1 cache was copied locally and checked against its existing manifest hash; that copy did not inspect D1 prediction errors.

## Background job and results

Entry point: [scripts/run_stage8.sh](../../scripts/run_stage8.sh). The launch uses `nohup setsid` with stdin disconnected. SSH logout does not stop it, and `--complete-matrix` sets no automatic wall-time kill. The instance itself still needs to remain running; a machine/platform interruption is handled through saved checkpoints rather than a promise that interruption is impossible.

Paths on AutoDL, relative to `/root/autodl-tmp/RobotDynamics-CS224W-Stanford`:

- Driver log: `outputs/stage8/formal_driver.log`
- Live queue and active configuration: `outputs/stage8/formal/progress.json`
- Process ID: `outputs/stage8/formal/pipeline.pid`
- Completion code: `outputs/stage8/formal/pipeline.exit`
- Final report and figures: `outputs/stage8/formal/analysis/`
- Per-model training logs: `outputs/stage8/formal/<run_id>.log`

An exit code of `0` means the pipeline reached its end; the analysis status and diagnostic flags still need review. A nonzero code preserves the failure and stops rather than skipping a model. If interrupted, restart the same command and output directory from the protocol; completed runs are verified and skipped. Do not launch another copy in a new directory.

Training, independent validation reloads, checkpoint sealing, both test sets, non-learning baselines, representation/coordinate diagnostics, CSV/JSON summaries, and PNG/PDF plots are all part of the script. No stage 9 work or automatic server shutdown follows. The formal results are not yet reviewed or published; stage 8 is not complete.
