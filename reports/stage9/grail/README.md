# GRAIL launch record

The pickup demonstration is a separate part of Stage 9. Its scope is now fixed: history-only forecasting from scratch on 240 tabletop pickup motions, with four models and three seeds. No GRAIL prediction result is available yet. The completed [Stack report](../README.md) remains unchanged as evidence about that task.

See the [protocol](../../../docs/grail_protocol.md), [configuration](../../../configs/grail_protocol.yaml), [asset-disjoint manifest](manifest.json), and [data audit](data_audit.json). All 720 downloaded files passed their pinned upstream hashes. The 240 trajectories contain 60,000 frames; 218 source keys stay within their asset split. No exact numerical duplicates or invalid fields were found. The audit is a release-format and numerical check, not a physics replay or confirmation that every pickup succeeds.

![Two training trajectories: root, object and recorded table positions](data_check.png)

The rows show the first two sorted training motions. The position traces were inspected for plausible alignment and motion; the figure does not verify contacts or controller behavior.

## Reproduce preparation

Use the existing Linux CUDA environment from the project setup, then add the file reader:

```bash
.venv/bin/python -m pip install -r requirements/grail.txt
.venv/bin/python -u -m robot_dynamics.prepare_grail \
  --raw outputs/grail/raw --output outputs/grail/prepared
```

The downloader is pinned to the recorded revision. The default endpoint is Hugging Face; `--endpoint https://hf-mirror.com` can retrieve the same hash-verified files where the main endpoint is inaccessible. The current AutoDL launch uses the identical cache prepared locally because the main Hugging Face endpoint timed out on that server. The source records and original files remain locally available.

## Train or resume on AutoDL

From `/root/autodl-tmp/RobotDynamics-CS224W-Stanford`, the deployed cache and manifest are `outputs/grail/cache.h5` and `outputs/grail/manifest.json`:

```bash
mkdir -p outputs/grail/formal
nohup setsid bash scripts/run_grail.sh \
  outputs/grail/cache.h5 outputs/grail/manifest.json outputs/grail/formal \
  > outputs/grail/formal/driver.log 2>&1 < /dev/null &
```

Use the same command after an interruption; the locks prevent a duplicate active queue. It resumes from the last completed epoch, checks finished runs, and continues remaining seeds. Do not modify the training files or data mid-queue.

```bash
cat outputs/grail/formal/progress.json
tail -n 30 outputs/grail/formal/driver.log
cat outputs/grail/formal/training.exit
```

`training.exit` is absent while the driver runs; zero plus `status: training_complete` and a 12-run `selection_seal.json` indicates successful completion. A failed process leaves its error and nonzero exit code. `waiting_for_gpu` means the queue is alive but another GPU job is still running. Logs are per run; checkpoints, normalizers and metrics are under the named model/seed directory. The training process does not evaluate test predictions or shut down the instance.

After training, verify all 12 configurations, logs, checkpoints and seals, then evaluate the reserved test assets, both non-learning baselines, grouped errors and trajectory displays. No test score or transfer claim should be inferred from the launch checks.
