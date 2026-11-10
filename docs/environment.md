# Environment notes

## Local machine

Audit date: 2026-11-10, America/Los_Angeles.

| Item | Observed |
| --- | --- |
| OS | Windows 10 Pro, build 19045, x86-64 |
| CPU | Intel Core i7-10750H, 6 physical cores / 12 threads |
| RAM | 15.76 GiB total; 2.46 GiB available at the audit |
| GPU | NVIDIA Quadro T1000 with Max-Q Design, 4096 MiB |
| Driver | 573.57; nvidia-smi reports CUDA compatibility 12.8 |
| Project drive | 475.69 GiB total; 77.8 GiB free |
| Python found | System 3.12.6; Anaconda 3.10.9 |
| Project Python | 3.10.9, isolated `.venv`, no system site-packages |
| WSL | Unavailable: `WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED` |

CPU model was read from the processor registry entry. Core count, RAM, platform, and project disk space came from `scripts/audit_resources.py`; GPU details came from `nvidia-smi`. Restricted WMI calls initially failed, so those errors were not treated as missing hardware. Available memory is a snapshot, not a reservation. The CUDA number in nvidia-smi does not prove that PyTorch or a CUDA toolkit is installed.

## Version choices

| Component | Selection | Reason |
| --- | --- | --- |
| Python | 3.10 (local 3.10.9) | Existing interpreter, compatible MuJoCo Windows wheel |
| MimicGen | `c0fbff22ba40060f05a917a80817f3f5c20731e7` | Resolved upstream main once, then pinned |
| robosuite | `b9d8d3de5e3dfd1724f4a0e6555246c460407daa` (1.4.1) | Commit specified by MimicGen |
| robomimic | `d0b37cf214bd24fb590d182edb6384333f67b661` (0.3.0) | Commit specified by MimicGen |
| MuJoCo | 2.3.2 | MimicGen's recommended simulator version |
| NumPy / SciPy | 1.26.4 / 1.11.4 | Keep the older simulation stack on NumPy 1.x |
| Numba | 0.59.1 | Compatible with this Python and NumPy selection |
| PyTorch / torchvision | 2.5.1 / 0.20.1, CUDA 12.1 wheels | Official matched pair; pending device validation |

MimicGen's installation guide lists macOS/Linux and warns against robosuite 1.5+. The documented source revisions were preferred over current releases. Python 3.10 is a practical choice here, not an upstream claim that the whole stack is tested on Windows.

`requirements/windows-base.lock.txt` is the full installed base dependency snapshot, including build tools. It does not contain a working simulation or PyTorch installation. `requirements/simulation.txt` and `requirements/torch.txt` pin the intended full stack; its transitive dependency lock must be exported on the successful Linux host. `robosuite_task_zoo` and pygame are not needed for the Square environment checks.

## Checks and blocker

Base dependency installation succeeded. The configuration checks passed, and `pip check` found no broken requirements. The config readiness command correctly rejected all five unfinished data checks. Formatting was applied and checked with Ruff. No models or data were used.

The single attempted install of required dependency `egl-probe==1.0.2` failed:

```text
Building wheel for egl-probe (setup.py): finished with status 'error'
CMake Error: Unknown argument -j
subprocess.check_call("cmake ..; make -j", cwd=build_dir, shell=True)
subprocess.CalledProcessError: Command 'cmake ..; make -j' returned non-zero exit status 1.
ERROR: Failed building wheel for egl-probe
```

Its build command assumes POSIX shell syntax. The selected robomimic revision requires `egl_probe>=1.0.1`. The failure blocks the full simulator stack on this host; it is not a demonstrated GPU memory failure. The raw local log is `outputs-egl-install.log` (ignored by Git). No third-party setup files were patched, no dependencies were silently omitted, and no further simulation install attempts were made after this failure.

Stage 1 is **BLOCKED_COMPUTE**. The protocol and base configuration are available, but full simulation imports, a MuJoCo step, and PyTorch CPU/CUDA checks remain unverified. No dataset was downloaded and no training job is running. The [handoff](stage_1_handoff.md) covers the remaining checks.

## Sources

- [MimicGen installation](https://mimicgen.github.io/docs/introduction/installation.html)
- [robomimic pinned dependencies](https://github.com/ARISE-Initiative/robomimic/blob/d0b37cf214bd24fb590d182edb6384333f67b661/setup.py)
- [MuJoCo 2.3.2 files](https://pypi.org/project/mujoco/2.3.2/#files)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [egl-probe source](https://github.com/StanfordVL/egl_probe)
