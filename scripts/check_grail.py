import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from robot_dynamics.formal import write_json
from robot_dynamics.grail_models import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(Path("configs/grail_protocol.yaml").read_text())["training"]
    cfg.update(
        purpose="check",
        device="cpu",
        threads=2,
        seed=0,
        epochs=2,
        selection_start=1,
        train_episodes=4,
        validation_episodes=4,
    )
    config = args.output / "check.yaml"
    config.write_text(yaml.safe_dump(cfg))

    def run(model, name, flags=(), configuration=config):
        command = [
            sys.executable,
            "-u",
            "-m",
            "robot_dynamics.train_grail",
            str(args.cache),
            "--manifest",
            str(args.manifest),
            "--config",
            str(configuration),
            "--model",
            model,
            "--output",
            str(args.output / name),
            *flags,
        ]
        with (args.output / f"{name}.log").open("a") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    checks = {}
    for model in MODELS:
        run(model, model)
        run(model, model, ("--reload-only",))
        resumed = model + "_resumed"
        run(model, resumed, ("--stop-after-epoch", "1"))
        run(model, resumed, ("--resume",))
        full = torch.load(
            args.output / model / "last.pt", weights_only=True, map_location="cpu"
        )
        continued = torch.load(
            args.output / resumed / "last.pt", weights_only=True, map_location="cpu"
        )
        exact = all(
            torch.equal(v, continued["model"][k]) for k, v in full["model"].items()
        )
        a = json.loads((args.output / model / "metrics.json").read_text())
        b = json.loads((args.output / resumed / "metrics.json").read_text())
        drift = max(
            (v - continued["model"][k]).abs().max().item()
            for k, v in full["model"].items()
        )
        for key, value in full["model"].items():
            torch.testing.assert_close(
                value, continued["model"][key], rtol=1e-5, atol=1e-6
            )
        if abs(a["best_score_cm"] - b["best_score_cm"]) > 1e-4:
            raise ValueError(
                f"Continuation validation drift exceeds tolerance: {model}"
            )
        checks[model] = {
            "exact_resume": exact,
            "resume_max_parameter_drift": drift,
            "resume_tolerance": {"rtol": 1e-5, "atol": 1e-6},
            "selected_prediction_hash_equal": a["prediction_sha256"]
            == b["prediction_sha256"],
            "independent_reload": True,
            "parameter_count": a["parameter_count"],
            "prediction_sha256": a["prediction_sha256"],
            "resources": a["resources"],
        }
        print(
            f"{model}: continuation and independent reload passed; maximum parameter drift {drift:.3g}",
            flush=True,
        )
    fit_config = args.output / "fit.yaml"
    cfg.update(purpose="fit", train_episodes=2, validation_episodes=2, epochs=30)
    fit_config.write_text(yaml.safe_dump(cfg))
    run("temporal_gnn", "fit", configuration=fit_config)
    fit = json.loads((args.output / "fit/metrics.json").read_text())
    ratio = fit["best_train_mse"] / fit["initial_train_mse_this_invocation"]
    if ratio >= 0.5:
        raise ValueError(
            f"Two-motion fitting check did not reduce loss sufficiently: {ratio}"
        )
    write_json(
        args.output / "checks.json",
        {
            "passed": True,
            "models": checks,
            "fit_final_initial_mse_ratio": ratio,
            "test_predictions_computed": False,
        },
    )
    print(f"All checks passed; two-motion fitting MSE ratio {ratio:.6f}", flush=True)


if __name__ == "__main__":
    main()
