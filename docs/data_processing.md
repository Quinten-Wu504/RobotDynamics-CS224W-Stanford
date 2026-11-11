# Square data processing

Use the tested Linux environment from [environment setup](stage_1_handoff.md). These commands use the existing AutoDL project and file storage. All data remains outside Git. The downloader pins Hugging Face revision `33016f8a62c02334f929f2913af8fdd2a8a129e1` of `amandlek/mimicgen_datasets` and checks file sizes and SHA-256 against the repository's LFS metadata.

```bash
cd /root/autodl-tmp/RobotDynamics-CS224W-Stanford
export LD_LIBRARY_PATH="$PWD/.venv/native/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
DATA_ROOT=/root/autodl-fs/RobotDynamics-CS224W-Stanford-data
mkdir -p "$DATA_ROOT/cache" outputs/stage2
.venv/bin/python scripts/download_square.py --directory "$DATA_ROOT/raw"
for task in square_d0 square_d1; do
  .venv/bin/python -m robot_dynamics.data_audit "$DATA_ROOT/raw/$task.hdf5" --output "outputs/stage2/${task}_audit.json"
  .venv/bin/python -m robot_dynamics.cache_square "$DATA_ROOT/raw/$task.hdf5" --output "$DATA_ROOT/cache/$task.hdf5"
  .venv/bin/python -m robot_dynamics.verify_square "$DATA_ROOT/raw/$task.hdf5" --output "outputs/stage2/${task}_restoration.json"
  .venv/bin/python -m robot_dynamics.plot_square "$DATA_ROOT/cache/$task.hdf5" --verification "outputs/stage2/${task}_restoration.json" --output "outputs/stage2/${task}_trajectories"
done
.venv/bin/python -m robot_dynamics.summarize_audit --audits outputs/stage2/square_d0_audit.json outputs/stage2/square_d1_audit.json --caches "$DATA_ROOT/cache/square_d0.hdf5" "$DATA_ROOT/cache/square_d1.hdf5" --output-dir outputs/stage2/summary
```

Existing downloads are verified and reused. Other outputs are never overwritten. An interrupted download leaves a `.partial` file; move it aside before restarting that file. An interrupted cache has `complete=false` and must not be used. Use a fresh output path for reruns. The summary directory must not already exist.

The full audit reads every dataset in every episode, including image arrays. The cache keeps numeric state only. The three fixed verification episodes are `demo_0`, `demo_499`, and `demo_999` in each distribution; these were chosen by index, without prediction results. Every recorded state in those episodes is restored. A separate first-20-action replay measures dynamic differences without replacing the recorded labels. Plots overlay recorded and restored positions and show target-peg geometry and table height.

The raw files total about 3.28 GB. Allow additional space for two numeric caches, full audit JSON, and optional pilot outputs. No training process or GPU rendering is needed for extraction and plots. Source commits and dirty-tree status are written into audit, cache, and restoration records; keep the checkout unchanged during a run.

The raw data is distributed through the [official MimicGen dataset page](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html) and [Hugging Face repository](https://huggingface.co/datasets/amandlek/mimicgen_datasets). Follow the upstream dataset usage terms. This project redistributes neither the raw HDF5 files nor camera images.
