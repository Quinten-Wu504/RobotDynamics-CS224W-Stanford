import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["numpy", "cpu", "cuda"], default="numpy")
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    device_name = args.device
    cuda_runtime = None
    if args.device != "numpy":
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.manual_seed(0)
        x = torch.randn(32, 16, device=args.device, requires_grad=True)
        loss = x.square().mean()
        loss.backward()
        if not torch.isfinite(x.grad).all():
            raise RuntimeError("Non-finite gradient in device smoke test")
        cuda_runtime = torch.version.cuda
        if args.device == "cuda":
            torch.cuda.synchronize()
            device_name = torch.cuda.get_device_name(0)

    with h5py.File("smoke.h5", "w", driver="core", backing_store=False) as file:
        file["values"] = np.arange(6).reshape(2, 3)
        np.testing.assert_array_equal(file["values"][:], np.arange(6).reshape(2, 3))

    packages = ["numpy", "h5py", "scipy", "matplotlib", "PyYAML"]
    for module in ["scipy", "matplotlib", "yaml"]:
        importlib.import_module(module)
    if args.device != "numpy":
        importlib.import_module("torchvision")
        packages.extend(["torch", "torchvision"])
    simulation_classes = []
    if args.simulation:
        for module in ["mujoco", "robosuite", "robomimic", "mimicgen"]:
            importlib.import_module(module)
            packages.append(module)
        import mujoco
        from mimicgen.envs.robosuite.nut_assembly import Square_D0, Square_D1

        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
            '<geom type="sphere" size="0.05"/></body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError("Non-finite MuJoCo state")
        simulation_classes = [Square_D0.__name__, Square_D1.__name__]

    result = {
        "kind": "environment_smoke_test",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            package: importlib.metadata.version(package) for package in packages
        },
        "device": args.device,
        "device_name": device_name,
        "cuda_runtime": cuda_runtime,
        "simulation_imports_and_step": args.simulation,
        "simulation_classes": simulation_classes,
        "gradient": "passed" if args.device != "numpy" else "not_run",
        "hdf5_roundtrip": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
