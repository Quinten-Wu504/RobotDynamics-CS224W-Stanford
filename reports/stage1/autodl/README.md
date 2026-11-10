# AutoDL checks

Source commit: `6344eeb4187457639ff65332fdeaf3a1c8dc3ebc`. Both smoke runs used a clean working tree.

| Check | Result |
| --- | --- |
| Dependency consistency | Passed; [pip output](pip_check.txt) |
| Configuration and container-limit tests | 11 passed in 0.70 s; [output](tests.txt) |
| Ruff and formatting | Passed; [lint](ruff.txt), [format](format.txt) |
| CPU gradients and HDF5 | Passed; [result](cpu.json) |
| CUDA gradients, simulation imports, MuJoCo step | Passed on RTX 4090 D; [result](cuda_simulation.json) |
| Data readiness gate | Exit 1 as expected; [output](readiness.txt) |

The [resource record](resources.txt) includes both host-visible totals and the actual 16-core / 80 GiB cgroup allocation. GPU memory is 24564 MiB. The Python dependency snapshot is [linux-cu121.lock.txt](../../../requirements/linux-cu121.lock.txt).

The first combined smoke check failed at EGL import; its [log](smoke.log) is retained. Extracting Ubuntu `libegl1=1.4.0-1` into the project environment fixed it. [Native package metadata](native_runtime.txt) records the checksum; the [successful log](smoke_egl.log) records the rerun. Existing system graphics packages were not upgraded.

The optional task-zoo and private macro warnings are retained. The Square D0/D1 classes imported successfully. These checks do not validate rendering, real dataset recovery, controller metadata, or model accuracy. No data was downloaded and no training run was started.
