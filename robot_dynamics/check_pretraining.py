import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from robot_dynamics.check_resume import identical
from robot_dynamics.data_audit import source_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-non-graph", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    models = ["temporal_gnn", "canonical_temporal_gnn"]
    if args.include_non_graph:
        models += ["history_mlp", "gru"]
    for model in models:
        for method in (
            ("supervised",)
            if model in ("history_mlp", "gru")
            else ("jepa", "state", "supervised")
        ):
            key = f"{model}_{method}"
            full = args.output / f"{key}_continuous"
            resumed = args.output / f"{key}_resumed"
            base = [
                sys.executable,
                "-m",
                "robot_dynamics.train_pretraining",
                str(args.cache),
                "--config",
                "configs/pretraining_check.yaml",
                "--model",
                model,
                "--method",
                method,
            ]
            stages = [
                ("continuous", ["--output", str(full)]),
                (
                    "pretraining_stop",
                    ["--output", str(resumed), "--stop-after-epoch", "2"],
                ),
                (
                    "finetuning_stop",
                    [
                        "--output",
                        str(resumed),
                        "--resume",
                        str(resumed / "last.pt"),
                        "--stop-after-epoch",
                        "4",
                    ],
                ),
                (
                    "resumed",
                    ["--output", str(resumed), "--resume", str(resumed / "last.pt")],
                ),
            ]
            for tag, extra in stages:
                with (args.output / f"{key}_{tag}.log").open("w") as log:
                    subprocess.run(
                        base + extra, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
            a = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
            b = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
            ma = json.loads((full / "metrics.json").read_text())
            mb = json.loads((resumed / "metrics.json").read_text())
            checks = {
                name: identical(a[name], b[name])
                for name in (
                    "model",
                    "auxiliary",
                    "optimizer",
                    "generator",
                    "torch_rng",
                    "cuda_rng",
                    "initial_head",
                )
            }
            checks["best_prediction_hash"] = (
                ma["prediction_sha256"] == mb["prediction_sha256"]
            )
            checks["best_summary"] = ma["summary"] == mb["summary"]
            record = {"model": model, "method": method, "checks": checks}
            records.append(record)
            (args.output / "checks.json").write_text(
                json.dumps({"records": records, **source_version()}, indent=2) + "\n"
            )
            if not all(checks.values()):
                raise ValueError(record)
            print(record, flush=True)


if __name__ == "__main__":
    main()
