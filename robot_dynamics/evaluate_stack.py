import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from robot_dynamics.baselines import mean_interval
from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.evaluate_formal import check_source, save_rows, summarize
from robot_dynamics.formal import write_json
from robot_dynamics.stack_data import ENTITIES, episode_arrays
from robot_dynamics.stack_models import StackPredictor
from robot_dynamics.train_learning import setup, tensors, predict
from robot_dynamics.train_stack import reload_validation


def grouped_rows(errors, arrays, name, thresholds):
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite physical prediction error")
    rows = []
    for k in (1, 5, 10):
        for j, entity in enumerate(ENTITIES):
            low = arrays["speed"][:, j] <= thresholds[j]
            for label, mask in [
                ("all", np.ones(len(low), bool)),
                ("low_motion", low),
                ("moving", ~low),
            ]:
                selected = mask & (arrays["horizon"] == k)
                if selected.any():
                    rows.append(
                        {
                            "episode": name,
                            "entity": entity,
                            "horizon_steps": k,
                            "segment": label,
                            "windows": int(selected.sum()),
                            "mean_error_cm": float(errors[selected, j].mean()),
                        }
                    )
    return rows


def physical_errors(prediction, arrays, positions, scale, baseline=None):
    t, k = arrays["anchor"], arrays["horizon"]
    target = positions[t + k] - positions[t]
    if baseline == "hold":
        delta = np.zeros_like(target)
    elif baseline == "constant_velocity":
        delta = (positions[t] - positions[t - 1]) * k[:, None, None]
    else:
        delta = prediction.reshape(-1, 3, 3).astype("float64") * scale
    return 100 * np.linalg.norm(delta - target, axis=-1)


def audit(runs, manifest_path, d0, validation_only):
    plan = json.loads((runs / "matrix.json").read_text())
    seal = json.loads((runs / "selection_seal.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    if seal["matrix_sha256"] != file_sha256(runs / "matrix.json") or plan[
        "manifest_sha256"
    ] != file_sha256(manifest_path):
        raise ValueError("Frozen matrix or manifest changed")
    if len(plan["runs"]) != (4 if validation_only else 12) or len(seal["runs"]) != len(
        plan["runs"]
    ):
        raise ValueError("Incomplete sealed matrix")
    state = json.loads((runs / "progress.json").read_text())
    if state["status"] != "training_complete" or set(state["completed"]) != {
        r["run_id"] for r in plan["runs"]
    }:
        raise ValueError("Training queue is incomplete")
    for path, digest in seal["code_sha256"].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"Sealed implementation changed: {path}")
    verified = []
    for run, selected in zip(plan["runs"], seal["runs"]):
        name, cfg = run["run_id"], run["config"]
        if (cfg["purpose"] == "formal") == validation_only or cfg["purpose"] not in (
            "formal",
            "check",
        ):
            raise ValueError("Workflow/test roles do not match")
        directory = runs / name
        metrics = json.loads((directory / "metrics.json").read_text())
        checkpoint = torch.load(
            directory / "best.pt", weights_only=True, map_location="cpu"
        )
        stats = json.loads((directory / "normalization.json").read_text())
        candidates = [
            entry
            for entry in metrics["logs"]
            if entry["epoch"] >= cfg["selection_start"]
        ]
        chosen = min(candidates, key=lambda entry: entry["validation_cm"])
        expected = {
            "run_id": name,
            "best_checkpoint_sha256": file_sha256(directory / "best.pt"),
            "validation_prediction_sha256": metrics["prediction_sha256"],
            "normalization_sha256": file_sha256(directory / "normalization.json"),
        }
        if (
            selected != expected
            or metrics["config"] != cfg
            or metrics["model"] != run["model"]
        ):
            raise ValueError("Sealed run/configuration mismatch")
        if [entry["epoch"] for entry in metrics["logs"]] != list(
            range(1, cfg["epochs"] + 1)
        ):
            raise ValueError("Missing epochs")
        if (
            checkpoint["epoch"] != chosen["epoch"]
            or metrics["best_epoch"] != chosen["epoch"]
            or checkpoint["best_score"] != chosen["validation_cm"]
        ):
            raise ValueError(
                "Checkpoint was not selected by the frozen validation rule"
            )
        train_ids = manifest["d0"]["train"][: cfg["train_episodes"]]
        if stats["train_episode_ids"] != train_ids or metrics["train_ids"] != train_ids:
            raise ValueError("Normalizer/training split mismatch")
        check_source(checkpoint)
        setup(cfg["seed"], cfg["threads"])
        reload = reload_validation(
            d0, manifest_path, directory, torch.device(cfg["device"])
        )
        verified.append(
            {"run_id": name, "best_epoch": chosen["epoch"], "validation_reload": reload}
        )
    return plan, manifest, verified


def evaluate(cache_path, ids, stats, model, method, device):
    rows, digest = [], hashlib.sha256()
    with h5py.File(cache_path, "r") as cache:
        if not cache.attrs["complete"] or cache.attrs["task"] != "stack":
            raise ValueError("Invalid Stack cache")
        for name in ids:
            group = cache[name]
            arrays = episode_arrays(group, stats)
            prediction = (
                predict(model, tensors(arrays, device)) if model is not None else None
            )
            errors = physical_errors(
                prediction, arrays, group["position"][:], stats["scale"], method
            )
            digest.update(name.encode())
            digest.update(errors.tobytes())
            rows.extend(
                grouped_rows(errors, arrays, name, stats["motion_threshold_m_per_s"])
            )
    values = [
        np.mean(
            [
                r["mean_error_cm"]
                for r in rows
                if r["episode"] == name and r["segment"] == "all"
            ]
        )
        for name in ids
    ]
    return rows, {
        "episode_ids": ids,
        "error_sha256": digest.hexdigest(),
        "summary": summarize(rows),
        "macro": mean_interval(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("d0_cache", type=Path)
    parser.add_argument("--d1-cache", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--verify-run")
    args = parser.parse_args()
    plan, manifest, verified = audit(
        args.runs, args.manifest, args.d0_cache, args.validation_only
    )
    if file_sha256(args.d0_cache) != manifest["datasets"]["stack_d0"]["cache_sha256"]:
        raise ValueError("D0 cache hash mismatch")
    if args.validation_only:
        splits = {
            "validation_a": (args.d0_cache, manifest["d0"]["validation"][:2]),
            "validation_b": (args.d0_cache, manifest["d0"]["validation"][2:4]),
        }
    else:
        if (
            args.d1_cache is None
            or file_sha256(args.d1_cache)
            != manifest["datasets"]["stack_d1"]["cache_sha256"]
        ):
            raise ValueError("D1 cache hash mismatch")
        splits = {
            "d0_test": (args.d0_cache, manifest["d0"]["test"]),
            "d1_final_test": (args.d1_cache, manifest["d1_final_test"]),
        }
    output = args.runs / "evaluation"
    output.mkdir(exist_ok=True)
    identity = {
        "selection_seal_sha256": file_sha256(args.runs / "selection_seal.json"),
        "evaluator_sha256": file_sha256(Path(__file__)),
        "validation_only": args.validation_only,
    }
    access_path = output / "access.json"
    if access_path.exists():
        if json.loads(access_path.read_text())["identity"] != identity:
            raise ValueError("Evaluation identity changed")
    else:
        write_json(
            access_path,
            {
                "identity": identity,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "training_audit": verified,
                **source_version(),
            },
        )
    reference = json.loads(
        (args.runs / plan["runs"][0]["run_id"] / "normalization.json").read_text()
    )
    jobs = plan["runs"] + [
        {"run_id": b, "model": b} for b in ("hold", "constant_velocity")
    ]
    completed = []
    for run in jobs:
        name, kind = run["run_id"], run["model"]
        if args.verify_run and args.verify_run != name:
            continue
        baseline = kind in ("hold", "constant_velocity")
        stats = (
            reference
            if baseline
            else json.loads((args.runs / name / "normalization.json").read_text())
        )
        if stats != reference:
            raise ValueError("Full-data normalizers differ across models/seeds")
        setup(0, 4)
        device = torch.device("cuda")
        model = None
        if not baseline:
            checkpoint = torch.load(
                args.runs / name / "best.pt", weights_only=True, map_location="cpu"
            )
            model = StackPredictor(kind).to(device)
            model.load_state_dict(checkpoint["model"])
        for split, (path, ids) in splits.items():
            result_path = output / f"{name}_{split}.json"
            rows_path = output / f"{name}_{split}_episodes.csv"
            if result_path.exists() and not args.verify_run:
                saved = json.loads(result_path.read_text())
                if (
                    saved["identity"] != identity
                    or saved["episode_rows_sha256"] != file_sha256(rows_path)
                    or saved["episode_ids"] != ids
                ):
                    raise ValueError("Previously evaluated artifact changed")
                completed.append(result_path.name)
                continue
            rows, result = evaluate(
                path, ids, stats, model, kind if baseline else None, device
            )
            result.update(
                run_id=name,
                model=kind,
                seed=None if baseline else run["config"]["seed"],
                split=split,
                identity=identity,
                **source_version(),
            )
            if args.verify_run:
                saved = json.loads(result_path.read_text())
                if any(
                    result[k] != saved[k]
                    for k in ("error_sha256", "macro", "summary", "episode_ids")
                ):
                    raise ValueError("Independent held-out reevaluation differs")
                write_json(
                    output / f"{name}_{split}_recheck.json",
                    {
                        "passed": True,
                        "error_sha256": result["error_sha256"],
                        **source_version(),
                    },
                )
            else:
                save_rows(rows_path, rows)
                result["episode_rows_sha256"] = file_sha256(rows_path)
                write_json(result_path, result)
            completed.append(result_path.name)
            print(name, split, result["macro"]["mean_cm"], flush=True)
    if args.verify_run:
        if not completed:
            raise ValueError("Unknown recheck run")
    else:
        write_json(
            output / "complete.json",
            {
                "results": completed,
                "count": len(completed),
                "identity": identity,
                **source_version(),
            },
        )


if __name__ == "__main__":
    main()
