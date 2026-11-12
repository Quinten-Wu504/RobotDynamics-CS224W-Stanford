import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from robot_dynamics.data_audit import source_version
from robot_dynamics.formal import write_json


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            equal(left[k], right[k]) for k in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = yaml.safe_load(Path("configs/stack_protocol.yaml").read_text())
    records = []
    for model in protocol["models"]:
        cfg = {
            **protocol["training"],
            "purpose": "check",
            "seed": 0,
            "train_episodes": 2,
            "validation_episodes": 4,
            "epochs": 4,
            "selection_start": 2,
        }
        cfg_path = args.output / f"{model}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        def run(directory, *extra):
            command = [
                sys.executable,
                "-m",
                "robot_dynamics.train_stack",
                str(args.cache),
                "--manifest",
                str(args.manifest),
                "--config",
                str(cfg_path),
                "--model",
                model,
                "--output",
                str(directory),
                *extra,
            ]
            with (args.output / f"{directory.name}.log").open("a") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )

        full, resumed = args.output / f"{model}_full", args.output / f"{model}_resumed"
        run(full)
        run(resumed, "--stop-after-epoch", "1")
        run(resumed, "--resume")
        a = torch.load(full / "last.pt", weights_only=True, map_location="cpu")
        b = torch.load(resumed / "last.pt", weights_only=True, map_location="cpu")
        for key in (
            "model",
            "optimizer",
            "generator",
            "torch_rng",
            "cuda_rng",
            "best_score",
            "epoch",
            "identity",
        ):
            if not equal(a[key], b[key]):
                raise ValueError(f"{model}: resume mismatch in {key}")
        m1, m2 = [json.loads((d / "metrics.json").read_text()) for d in (full, resumed)]
        if m1["prediction_sha256"] != m2["prediction_sha256"]:
            raise ValueError("Resumed best predictions differ")
        run(resumed, "--reload-only")
        # Reconstructing metrics after the final checkpoint must also work.
        run(resumed, "--resume")
        if (
            json.loads((resumed / "metrics.json").read_text())["prediction_sha256"]
            != m1["prediction_sha256"]
        ):
            raise ValueError("Completed-checkpoint replay differs")
        fit = args.output / f"{model}_fit"
        cfg.update(purpose="fit", epochs=40, selection_start=1)
        cfg_path.write_text(yaml.safe_dump(cfg))
        run(fit)
        m = json.loads((fit / "metrics.json").read_text())
        ratio = m["best_train_mse"] / m["initial_train_mse_this_invocation"]
        if ratio >= 0.1:
            raise ValueError(
                f"{model}: small-sample fit did not reduce MSE tenfold: {ratio}"
            )
        run(fit, "--reload-only")
        records.append(
            {
                "model": model,
                "exact_resume": True,
                "independent_reload": True,
                "completed_checkpoint_replay": True,
                "fit_mse_ratio": ratio,
            }
        )
        write_json(args.output / "checks.json", {"models": records, **source_version()})


if __name__ == "__main__":
    main()
