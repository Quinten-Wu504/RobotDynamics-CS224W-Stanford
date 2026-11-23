# Reproducing the project

There are three useful entry points: redraw the published results, run the small Square example, or repeat a formal experiment. The first needs no raw data or PyTorch. The second trains a real graph model on eight training episodes and checks it on two validation episodes. The third uses the frozen matrices and their original data/provenance requirements.

## Environment

Run commands from the repository root. The small example was tested in a fresh Windows Python 3.10.9 CPU environment, separately from the development environment. The Linux project environment also passes the full suite. The pinned wheel set targets Python 3.10–3.12; newer Python releases have not been validated.

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -r requirements/demo.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip check
python -m pytest -q
```

The CPU wheel is sufficient for the demonstration. The [official PyTorch version page](https://pytorch.org/get-started/previous-versions/) also lists the CUDA 12.1 wheel used for the formal experiments. GPU experiments and the Linux queue scripts use the setup in [environment.md](environment.md); simulator replay has additional dependencies. The numeric Square decoder and demo do not require a simulator or GPU.

## Redraw results without training

For plots only, install `requirements/base.txt`, then:

```bash
python -m robot_dynamics.final_results --output outputs/final_results
```

The output contains all 62 aggregate rows, all 84 learned-run cost rows, source-file hashes, and PNG/PDF figures for data efficiency, horizons, distributions, costs, proximity groups, GRAIL validation curves and trajectory examples. It reads the archived analyses; GRAIL episode errors are also reaggregated and checked for exact agreement. It does not execute fresh test predictions. The output directory must be new.

The underlying numerical archives and their earlier independent checks are linked from the [final report](../reports/stage10/README.md). Task metrics differ: Square and GRAIL each average six entity/horizon cells, while Stack averages nine. Do not average their scores into a single benchmark number.

## Small Square example

The [notebook](../notebooks/robot_dynamics_demo.ipynb) provides the same steps with explanations and inline plots. [Open it in Colab](https://colab.research.google.com/github/Quinten-Wu504/RobotDynamics-CS224W-Stanford/blob/main/notebooks/robot_dynamics_demo.ipynb). It pins its source commit and defaults to CPU. Its cells were executed with a separate clean clone and a fresh local environment; the hosted Colab interface was not run. The installation and initial cache preparation were validated separately. Colab can load notebooks from GitHub, but runtime availability and duration are not guaranteed ([Colab FAQ](https://research.google.com/colaboratory/faq.html)).

The initial raw download is 1,621,351,476 bytes. Allow about 4 GB free storage for data, cache and outputs, in addition to the environment. The downloader uses upstream revision `33016f8a62c02334f929f2913af8fdd2a8a129e1` and verifies SHA-256 `41fc24bce0f88343099c0b1b5bf6eee08cbc35851e71276d4509a01c9b75481c`. Follow the [upstream dataset terms](https://huggingface.co/datasets/amandlek/mimicgen_datasets); the repository does not redistribute raw data.

```bash
python scripts/download_square.py --directory outputs/demo_data/raw --tasks square_d0
python -m robot_dynamics.cache_square outputs/demo_data/raw/square_d0.hdf5 --output outputs/demo_data/square_d0.hdf5
python -m robot_dynamics.demo_square prepare outputs/demo_data/square_d0.hdf5 --output outputs/square_demo
python -m robot_dynamics.train_learning outputs/square_demo/square_demo.hdf5 --manifest outputs/square_demo/manifest.json --config configs/demo_square.yaml --model temporal_gnn --output outputs/square_demo/temporal_gnn
python -m robot_dynamics.evaluate_learning outputs/square_demo/square_demo.hdf5 --manifest outputs/square_demo/manifest.json --checkpoint outputs/square_demo/temporal_gnn/best.pt --output outputs/square_demo/reload.json
python -m robot_dynamics.demo_square plot outputs/square_demo
```

Only the first eight original training episodes and first two original validation episodes are copied into the small cache. The formal test set is absent. Normalization uses training episodes only. The model predicts displacement from four observed state frames and the appropriate planned actions, at horizons of 1/5/10 steps. Graph nodes, edge features and message layers are the research implementation, not a simplified surrogate.

The five-epoch configuration selects its best validation epoch without the formal experiment's 11–40 restriction. This is intentional for a short teaching example. Its score is not a formal result and does not compare architectures. The [recorded run](../reports/stage10/demo/verification.json) took about eight seconds of training and reload on CPU, with roughly 0.55 GiB peak observed process memory. Download, installation, initial decoding and plotting are additional costs and depend on the machine/network.

Outputs include `graph.png`, `trajectory.png`, `verification.json`, the normalizer, per-epoch metrics and `best.pt`/`last.pt`. The standalone evaluation must reproduce the training run's full validation prediction hash. The plotted episode is the first fixed validation episode; each point is a direct 0.5 s forecast, not an autoregressive rollout.

## Checkpoints and interruption

The demo creates its checkpoint locally; no private server access or checkpoint download is needed. The notebook's final cell exports a ZIP containing weights, normalizer, manifest, metrics and figures, excluding raw/cache datasets. Preserve the pinned source revision and regenerate the same inputs to use it later.

For interruption testing, use a new run directory and stop at epoch 2, then resume that directory with the identical configuration:

```bash
python -m robot_dynamics.train_learning outputs/square_demo/square_demo.hdf5 --manifest outputs/square_demo/manifest.json --config configs/demo_square.yaml --model temporal_gnn --output outputs/square_resume --stop-after-epoch 2
python -m robot_dynamics.train_learning outputs/square_demo/square_demo.hdf5 --manifest outputs/square_demo/manifest.json --config configs/demo_square.yaml --model temporal_gnn --output outputs/square_resume --resume outputs/square_resume/last.pt
```

There is no public download of the original formal weights. They remain in the project owner's AutoDL storage. This is a limitation of exact checkpoint replay, not of the runnable demo or published numerical reanalysis. Their locations and the corresponding integrity records are:

| Task | AutoDL project-relative directory | Selected weights and provenance |
| --- | --- | --- |
| Square | `outputs/stage8/formal/<run_id>/` | `best.pt`, `normalization.json`; [selection seal](../reports/stage8/formal/selection_seal.json) |
| Stack | `outputs/stage9/formal/<run_id>/` | `best.pt`, `normalization.json`; [selection seal](../reports/stage9/formal/selection_seal.json) |
| GRAIL | `outputs/grail/formal/<run_id>/` | `best.pt`, `normalization.json`; [selection seal](../reports/stage9/grail/formal/selection_seal.json) |

The project root there is `/root/autodl-tmp/RobotDynamics-CS224W-Stanford`. Exact replay also needs the original byte-identical numeric caches, manifests and source code. The stage reports identify the source commits. Square/Stack are on the original server; GRAIL is on the second server. Stage 10 reverified a GRAIL checkpoint; the original Square/Stack server was not available for another replay. Their earlier full rechecks remain archived.

## Full experiments

These are substantially longer runs, separate from the notebook. They were not repeated in Stage 10. The [Square protocol](formal_protocol.md), [Stack protocol](stack_protocol.md), [Stack launch record](../reports/stage9/launch.md), and [GRAIL launch record](../reports/stage9/grail/launch.md) give their complete pipelines. The representative main comparison is supervised Temporal GNN, all three seeds; Square also includes canonical models and JEPA/state-pretraining controls. The machine-readable matrices are [Square](../configs/formal_protocol.yaml), [Stack](../configs/stack_protocol.yaml), and [GRAIL](../configs/grail_protocol.yaml).

Freshly generated HDF5 caches can have different byte hashes because metadata records their construction. Do not bypass an old checkpoint's integrity checks or replace hashes in its seal. A new training reproduction must bind its own audited cache to a new manifest and preserve the original episode split. For Square, use the existing audit/summary/split tools in a separate reproduction checkout:

```bash
python scripts/download_square.py --directory outputs/reproduction/raw
python -m robot_dynamics.data_audit outputs/reproduction/raw/square_d0.hdf5 --output outputs/reproduction/d0_audit.json
python -m robot_dynamics.data_audit outputs/reproduction/raw/square_d1.hdf5 --output outputs/reproduction/d1_audit.json
python -m robot_dynamics.cache_square outputs/reproduction/raw/square_d0.hdf5 --output outputs/reproduction/square_d0.hdf5
python -m robot_dynamics.cache_square outputs/reproduction/raw/square_d1.hdf5 --output outputs/reproduction/square_d1.hdf5
python -m robot_dynamics.summarize_audit --audits outputs/reproduction/d0_audit.json outputs/reproduction/d1_audit.json --caches outputs/reproduction/square_d0.hdf5 outputs/reproduction/square_d1.hdf5 --output-dir outputs/reproduction/audit_summary
python -m robot_dynamics.splits --audit-dir outputs/reproduction/audit_summary --output outputs/reproduction/manifest.json
python -m robot_dynamics.normalization outputs/reproduction/square_d0.hdf5 --manifest outputs/reproduction/manifest.json --output outputs/reproduction/normalization.json
```

The formal Square runner currently reads `reports/stage3/manifest.json` and `reports/stage3/normalization.json`. In that separate checkout, verify the new manifest's `d0` and `d1_final_test` values equal the published split, replace only those two provenance/normalization files with the generated versions, and commit the reproduction setup on your own branch before running the formal queue. Keep the published reports and original seals intact. This full fresh-data matrix has not been rerun in Stage 10; the tested fresh-data path is the small example above. Stack and GRAIL accept an explicit newly generated manifest path.

Published full-matrix times were 14 h 53 min for Square, 3 h 50 min for Stack training/reloads, and 56 min 47 s for GRAIL training/reloads. These are observations, not runtime promises. Use the configured recovery scripts and original output directory after interruption. They survive ordinary SSH disconnection but cannot prevent machine shutdown or storage failure.

All formal test sets have already been examined. A change motivated by their errors is exploratory; it must not be presented as a new untouched-test confirmation. None of these offline predictions establishes closed-loop robot success.
