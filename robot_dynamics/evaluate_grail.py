import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
import torch
import yaml

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.formal import write_json
from robot_dynamics.grail_data import make_arrays, fit_statistics, validate_manifest
from robot_dynamics.grail_models import GrailPredictor
from robot_dynamics.grail_results import (
    displacement,
    grouped_rows,
    summarize,
    asset_values,
)
from robot_dynamics.train_grail import code_hashes, reload_validation
from robot_dynamics.train_learning import setup, tensors, predict


def save_rows(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def audit(runs, cache_path, manifest_path):
    plan = json.loads((runs / "matrix.json").read_text())
    seal = json.loads((runs / "selection_seal.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    protocol_path = Path("configs/grail_protocol.yaml")
    protocol = yaml.safe_load(protocol_path.read_text())
    expected_runs = [
        {
            "run_id": f"{model}_s{seed}",
            "model": model,
            "config": {**protocol["training"], "seed": seed},
        }
        for seed in protocol["seeds"]
        for model in protocol["models"]
    ]
    if (
        plan["runs"] != expected_runs
        or len(expected_runs) != 12
        or len(seal["runs"]) != 12
    ):
        raise ValueError("Formal matrix changed or is incomplete")
    if (
        seal["matrix_sha256"] != file_sha256(runs / "matrix.json")
        or plan["protocol_sha256"] != file_sha256(protocol_path)
        or plan["manifest_sha256"] != file_sha256(manifest_path)
        or plan["cache_sha256"] != file_sha256(cache_path)
        or manifest["cache_sha256"] != plan["cache_sha256"]
        or plan["training_code"] != code_hashes()
        or plan["runner_sha256"] != file_sha256(Path("robot_dynamics/run_grail.py"))
    ):
        raise ValueError("Frozen source or data identity changed")
    progress = json.loads((runs / "progress.json").read_text())
    if (
        progress["status"] != "training_complete"
        or sorted(progress["completed"]) != sorted(r["run_id"] for r in expected_runs)
        or (runs / "training.exit").read_text().strip() != "0"
    ):
        raise ValueError("Training did not complete successfully")
    with h5py.File(cache_path, "r") as cache:
        validate_manifest(manifest, cache)
        fitted = fit_statistics(cache, manifest["splits"]["train"])
    checked = []
    for run, selected in zip(expected_runs, seal["runs"]):
        name, cfg = run["run_id"], run["config"]
        directory = runs / name
        result = json.loads((directory / "metrics.json").read_text())
        checkpoint = torch.load(
            directory / "best.pt", weights_only=True, map_location="cpu"
        )
        stats = json.loads((directory / "normalization.json").read_text())
        expected = {
            "run_id": name,
            "best_checkpoint_sha256": file_sha256(directory / "best.pt"),
            "validation_prediction_sha256": result["prediction_sha256"],
            "normalization_sha256": file_sha256(directory / "normalization.json"),
        }
        if (
            selected != expected
            or result["config"] != cfg
            or result["model"] != run["model"]
            or result["train_ids"] != manifest["splits"]["train"]
            or stats != fitted
        ):
            raise ValueError(f"Invalid sealed run or training-only statistics: {name}")
        if [r["epoch"] for r in result["logs"]] != list(range(1, 41)):
            raise ValueError("Missing training epochs")
        chosen = min(
            (r for r in result["logs"] if r["epoch"] >= 11),
            key=lambda r: r["validation_cm"],
        )
        if (
            checkpoint["epoch"] != chosen["epoch"]
            or result["best_epoch"] != chosen["epoch"]
            or checkpoint["best_score"] != chosen["validation_cm"]
        ):
            raise ValueError("Checkpoint does not follow frozen selection rule")
        for filename, digest in plan["training_code"].items():
            original = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{checkpoint['source_commit']}:robot_dynamics/{filename}",
                ]
            )
            if hashlib.sha256(original).hexdigest() != digest:
                raise ValueError("Historical checkpoint source mismatch")
        setup(cfg["seed"], cfg["threads"])
        reloaded = reload_validation(
            cache_path, manifest_path, directory, torch.device("cuda")
        )
        checked.append(
            {
                "run_id": name,
                "best_epoch": chosen["epoch"],
                "validation_reload": reloaded,
            }
        )
    return plan, manifest, fitted, checked


def evaluate(cache_path, ids, stats, model, method, manifest, shown):
    rows, trajectories, digest = [], [], hashlib.sha256()
    with h5py.File(cache_path, "r") as cache:
        for name in ids:
            arrays = make_arrays(cache, [name], stats)
            prediction = (
                predict(model, tensors(arrays, torch.device("cuda")))
                if model is not None
                else None
            )
            positions = cache[name]["position"][:].astype("float64")
            delta = displacement(prediction, arrays, positions, stats, method)
            asset = manifest["audit"][name]["asset"]
            episode_rows, errors = grouped_rows(delta, arrays, name, asset, stats)
            rows.extend(episode_rows)
            digest.update(name.encode())
            digest.update(errors.tobytes())
            if asset in shown:
                for i in np.flatnonzero(arrays["horizon"] == 12):
                    t = int(arrays["anchor"][i])
                    for j, entity in enumerate(("robot_root", "object")):
                        predicted = positions[t, j] + delta[i].reshape(2, 3)[j]
                        trajectories.append(
                            {
                                "episode": name,
                                "asset": asset,
                                "entity": entity,
                                "anchor": t,
                                "target_frame": t + 12,
                                **{
                                    f"true_{axis}": float(positions[t + 12, j, d])
                                    for d, axis in enumerate("xyz")
                                },
                                **{
                                    f"pred_{axis}": float(predicted[d])
                                    for d, axis in enumerate("xyz")
                                },
                            }
                        )
    assets, values = asset_values(rows)
    return (
        rows,
        trajectories,
        {
            "episode_ids": ids,
            "asset_ids": assets,
            "asset_means_cm": values.tolist(),
            "mean_cm": float(values.mean()),
            "error_sha256": digest.hexdigest(),
            "summary": summarize(rows),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--recheck", nargs="+", default=[])
    args = parser.parse_args()
    source = source_version()
    if source["working_tree_dirty"]:
        raise ValueError("Commit the evaluator before accessing evaluation data")
    plan, manifest, stats, verified = audit(args.runs, args.cache, args.manifest)
    role = "validation_workflow" if args.validation_only else "test"
    # Keep complete asset pairs even in the workflow check.
    ids = manifest["splits"]["validation" if args.validation_only else "test"]
    shown = sorted({manifest["audit"][name]["asset"] for name in ids})[:2]
    if args.validation_only:
        ids = [name for name in ids if manifest["audit"][name]["asset"] in shown]
    output = args.runs / (
        "evaluation_workflow" if args.validation_only else "evaluation"
    )
    output.mkdir(exist_ok=True)
    identity = {
        "selection_seal_sha256": file_sha256(args.runs / "selection_seal.json"),
        "evaluator_sha256": file_sha256(Path(__file__)),
        "results_code_sha256": file_sha256(Path("robot_dynamics/grail_results.py")),
        "role": role,
        "episode_ids": ids,
    }
    access = output / "access.json"
    if access.exists():
        if json.loads(access.read_text())["identity"] != identity:
            raise ValueError("Evaluation identity changed")
    else:
        write_json(
            access,
            {
                "identity": identity,
                "source": source,
                "opened_unix_time": time.time(),
                "training_audit": verified,
            },
        )
    runs = [
        r for r in plan["runs"] if not args.validation_only or r["config"]["seed"] == 0
    ]
    runs += [
        {"run_id": k, "model": k, "config": {"seed": None}}
        for k in ("hold", "constant_velocity")
    ]
    if args.recheck and not set(args.recheck) <= {r["run_id"] for r in runs}:
        raise ValueError("Unknown recheck run")
    completed = []
    for run in runs:
        name, seed = run["run_id"], run["config"]["seed"]
        if args.recheck and name not in args.recheck:
            continue
        setup(seed or 0, 4)
        model = None
        if seed is not None:
            checkpoint = torch.load(
                args.runs / name / "best.pt", weights_only=True, map_location="cpu"
            )
            model = GrailPredictor(run["model"]).cuda()
            model.load_state_dict(checkpoint["model"])
        rows, trajectories, result = evaluate(
            args.cache,
            ids,
            stats,
            model,
            run["model"],
            manifest,
            shown if seed in (None, 0) else [],
        )
        path = output / f"{name}.json"
        if args.recheck:
            old = json.loads(path.read_text())
            if any(old[k] != result[k] for k in result):
                raise ValueError(f"Independent full evaluation changed: {name}")
            write_json(
                output / f"{name}_recheck.json",
                {
                    "passed": True,
                    "run_id": name,
                    "error_sha256": result["error_sha256"],
                    "source": source,
                },
            )
        else:
            if path.exists():
                raise FileExistsError(
                    f"Evaluation already exists: {path}; use --recheck"
                )
            csv_path = output / f"{name}_episodes.csv"
            save_rows(csv_path, rows)
            if trajectories:
                trajectory_path = output / f"{name}_trajectories.csv"
                save_rows(trajectory_path, trajectories)
                result["trajectory_rows_sha256"] = file_sha256(trajectory_path)
            write_json(
                path,
                {
                    **result,
                    "model": run["model"],
                    "seed": seed,
                    "role": role,
                    "identity": identity,
                    "source": source,
                    "episode_rows_sha256": file_sha256(csv_path),
                },
            )
            completed.append(path.name)
        print(f"{role}: {name} complete", flush=True)
    if not args.recheck:
        write_json(
            output / "complete.json",
            {
                "results": completed,
                "identity": identity,
                "source": source,
                "finished_unix_time": time.time(),
            },
        )


if __name__ == "__main__":
    main()
