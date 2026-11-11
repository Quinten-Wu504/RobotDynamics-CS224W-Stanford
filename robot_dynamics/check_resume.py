import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from robot_dynamics.data_audit import source_version


def identical(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            identical(left[k], right[k]) for k in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            identical(a, b) for a, b in zip(left, right)
        )
    return left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for model in ("current_mlp", "history_mlp", "gru"):
        full = args.output / f"{model}_continuous"
        resumed = args.output / f"{model}_resumed"
        base = [
            sys.executable,
            "-m",
            "robot_dynamics.train_learning",
            str(args.cache),
            "--config",
            "configs/learning_resume_check.yaml",
            "--model",
            model,
        ]
        for tag, extra in [
            ("continuous", ["--output", str(full)]),
            ("interrupted", ["--output", str(resumed), "--stop-after-epoch", "3"]),
            (
                "resumed",
                ["--output", str(resumed), "--resume", str(resumed / "last.pt")],
            ),
        ]:
            with (args.output / f"{model}_{tag}.log").open("w") as stream:
                subprocess.run(
                    base + extra, stdout=stream, stderr=subprocess.STDOUT, check=True
                )
        a = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
        b = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
        ma = json.loads((full / "metrics.json").read_text())
        mb = json.loads((resumed / "metrics.json").read_text())
        checks = {
            "model": model,
            "last_model_exact": identical(a["model"], b["model"]),
            "optimizer_exact": identical(a["optimizer"], b["optimizer"]),
            "sampler_state_exact": identical(a["generator"], b["generator"]),
            "best_predictions_exact": ma["prediction_sha256"]
            == mb["prediction_sha256"],
            "best_metrics_exact": ma["summary"] == mb["summary"],
        }
        if not all(v for k, v in checks.items() if k != "model"):
            raise ValueError(checks)
        records.append(checks)
        print(checks, flush=True)
    (args.output / "checks.json").write_text(
        json.dumps({"checks": records, **source_version()}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
