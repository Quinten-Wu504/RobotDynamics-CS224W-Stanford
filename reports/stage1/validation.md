# Stage 1 checks

Code snapshot: `74cb4d306f480c4bcb14c8b5398505fad7e312ec`.

Commands below were run from the repository root with `.venv\Scripts\python.exe` on Windows. This is an environment/configuration check, not a prediction experiment.

| Check | Result |
| --- | --- |
| `python -m pytest -q` | 9 passed in 0.96 s |
| `python -m ruff check robot_dynamics scripts tests` | Passed |
| `python -m ruff format --check robot_dynamics scripts tests` | Passed; five Python files |
| `python -m compileall -q robot_dynamics scripts tests` | Passed |
| `python -m pip check` | No broken requirements |
| `python -m robot_dynamics.config` | Configuration loads; five unresolved data checks reported |
| `python -m robot_dynamics.config --require-audited` | Exit 1 as expected; data audit incomplete |
| `python scripts/smoke_environment.py --device numpy --output outputs/stage1/base_smoke.json` | Passed from a clean working tree; [result](base_smoke.json) |
| `python -m pip install egl-probe==1.0.2` | Failed building wheel; see [error excerpt](../../docs/environment.md) |

The tests cover frequency-based horizon conversion, invalid frequencies, episode-level splits, and incomplete audit flags. No real trajectories were used, so they do not establish correct dataset time alignment or absence of data leakage in a future loader.

The base smoke result records Python/package versions and the source commit. PyTorch gradients, CUDA execution, full simulator imports, rendering, and dataset replay were not tested. The base dependency snapshot is [windows-base.lock.txt](../../requirements/windows-base.lock.txt); it is not a complete simulator lock.

Stage 1 remains blocked. No dataset, training run, checkpoint, or prediction metric was produced.
