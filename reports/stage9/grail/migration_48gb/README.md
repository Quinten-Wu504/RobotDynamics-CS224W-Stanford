# Migration to the 48 GB server

The original GRAIL queue was waiting for another GPU task and had completed no formal runs. It was stopped before migration; [its final progress record](old_queue.json) explicitly says `migrated_to_new_server`. No other process on that server was stopped. The old driver's exit file alone is not evidence of training completion: a complete queue also requires `training_complete` and the checkpoint selection seal.

The new server has one RTX 4090 D reporting 49,140 MiB of GPU memory and a 90 GiB container memory limit. Its pre-existing task finished during setup. The same Git repository, isolated Python environment and numerical cache are transferred directly from the original server. A separate local-to-server cache transfer has the identical SHA-256, `96164e2d94a3442fc925a9a3e2a7b84d250576acc5787bfeb714eaa22e59292a`. The cache and split match the published manifest. An interrupted external clone and a partial fresh-environment download are not used for the experiment.

The frozen model architectures, batch size 512, 40 epochs, 12 model/seed configurations, selected assets and validation rule are unchanged. This is a hardware migration, not another hyperparameter search. The larger GPU provides headroom; allocating all available memory would not by itself improve the experiment. Runs remain sequential so their timings are not confounded by overlapping training jobs.

Before launch, the CUDA workflow checks and one-epoch full-training-pool probes use only training/validation data. They test continued training, standalone checkpoint reloads and peak allocation; probe checkpoints are not formal results or reused for formal training. Held-out test predictions remain deferred until the training-complete notification.

## Checks and resource measurements

The source commit for validation and formal training is `ef13ad69315c9b3adf67153ab431a0e7eb84b6f4`. Python is 3.10.8 and PyTorch is 2.5.1+cu121. The [environment snapshot](environment.txt) records the transferred package versions. All [60 tests](tests.txt) and [Ruff checks](ruff.txt) pass on the new server.

All four models pass CUDA independent reload and uninterrupted/resumed training checks; the final parameters and selected validation predictions match exactly in these checks. The two-motion fitting MSE ratio is 0.158907. See the [CUDA check summary](cuda_check/checks.json) and [full-pool probe record](validation.json). Each probe uses all 168 training motions and 36 validation motions (118,776 and 25,452 windows) for one epoch, with the formal batch size and model unchanged.

| Model | Peak allocated CUDA memory (GiB) | Training update time per epoch (s) |
|---|---:|---:|
| history_mlp | 1.044 | 1.49 |
| gru | 1.044 | 5.98 |
| temporal_gnn | 1.521 | 9.74 |
| temporal_no_messages | 1.521 | 9.88 |

The [one-second device samples](gpu_samples.csv) reached 2,121 MiB (2.07 GiB), including CUDA context and allocator reserve. Sampling can miss shorter spikes; PyTorch's allocation peak and sampled device usage are distinct measurements. The earlier 8 GB free-memory gate was a conservative launch margin, not a measured minimum. The new GPU has ample headroom.

Scaling the measured update and validation times to 40 epochs and three seeds gives 54.9 minutes before loading, checkpoint writes and independent reloads. About one hour for the whole queue is a practical estimate, not a runtime guarantee. GPU speed and epoch costs can vary.

## Active queue

The formal driver is running on the new 48 GB server, PID **7691**, with runner PID **7696**. The [post-disconnection verification](launch_verification.json) observes `training`, one completed configuration, no other GPU compute process and no test evaluation. The first MLP seed completed all 40 epochs and passed its independent reload before GRU seed 0 began; the [launch progress snapshot](launch_progress.json) records the observed state rather than final completion.

The launch SSH client was closed, then a new connection confirmed that the driver still runs in its own session, with ignored SIGHUP, `/dev/null` input and file logs. Its outer shell is adopted by PID 1; the driver itself remains session leader. A duplicate launch was refused with exit 1. These checks demonstrate survival of SSH disconnection, not immunity to server shutdown or external kill signals. The launcher now records explicit 143/130 exit codes for TERM/INT; each completed epoch remains recoverable.

On the new server, the project root and launch/recovery command are unchanged:

```bash
cd /root/autodl-tmp/RobotDynamics-CS224W-Stanford
cat outputs/grail/formal/progress.json
tail -n 20 outputs/grail/formal/driver.log
```

If the queue is interrupted, restart it from that directory:

```bash
mkdir -p outputs/grail/formal
nohup setsid bash scripts/run_grail.sh \
  outputs/grail/cache.h5 outputs/grail/manifest.json outputs/grail/formal \
  > outputs/grail/formal/driver.log 2>&1 < /dev/null &
```

The original server's queue stays stopped. The new queue stops after training, all 12 independent validation reloads and checkpoint sealing; it does not run held-out evaluation or shut down the instance. Results and trajectory displays remain pending until the training-complete notification. Stage 10 has not started.

To reproduce the CUDA recovery checks in a new output directory:

```bash
.venv/bin/python -m scripts.check_grail outputs/grail/cache.h5 \
  --manifest outputs/grail/manifest.json \
  --output outputs/grail/cuda_check_again --device cuda
```
