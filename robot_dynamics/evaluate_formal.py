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

from robot_dynamics.analyze_equivariance import coordinate_checks
from robot_dynamics.analyze_pretraining import latent_checks
from robot_dynamics.baselines import mean_interval
from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.formal import matrix, stable_digest, validate_completed, write_json
from robot_dynamics.learning_data import make_arrays
from robot_dynamics.learning_models import DynamicsPredictor
from robot_dynamics.train_learning import setup, predict, tensors
from robot_dynamics.windows import read_episode, window, anchors


def grouped_errors(errors, episode, arrays, positions, thresholds, near_distance):
    anchor = arrays["anchor"]
    speed = np.linalg.norm(positions[anchor] - positions[anchor - 1], axis=-1) / 0.05
    near = (
        np.linalg.norm(positions[anchor, 0] - positions[anchor, 1], axis=-1)
        <= near_distance
    )
    rows = []
    for horizon in (1, 5, 10):
        eligible = arrays["horizon"] == horizon
        for j, entity in enumerate(("end_effector", "square_nut")):
            low = speed[:, j] <= thresholds[j]
            for segment, mask in [
                ("all", np.ones(len(anchor), bool)),
                ("low_motion", low),
                ("moving", ~low),
                ("near_object", near),
                ("far_object", ~near),
            ]:
                selected = eligible & mask
                if selected.any():
                    rows.append(
                        {
                            "episode": episode,
                            "entity": entity,
                            "horizon_steps": horizon,
                            "segment": segment,
                            "windows": int(selected.sum()),
                            "mean_error_cm": float(errors[selected, j].mean()),
                        }
                    )
    return rows


def summarize(rows):
    result = []
    for entity, horizon, segment in sorted(
        {(r["entity"], r["horizon_steps"], r["segment"]) for r in rows}
    ):
        selected = [
            r
            for r in rows
            if (r["entity"], r["horizon_steps"], r["segment"])
            == (entity, horizon, segment)
        ]
        result.append(
            {
                "entity": entity,
                "horizon_steps": horizon,
                "segment": segment,
                "episodes": len(selected),
                "windows": sum(r["windows"] for r in selected),
                **mean_interval([r["mean_error_cm"] for r in selected]),
            }
        )
    return result


def save_rows(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def check_source(checkpoint):
    for name, digest in checkpoint["identity"]["code"].items():
        blob = subprocess.check_output(
            ["git", "show", f"{checkpoint['source_commit']}:robot_dynamics/{name}"]
        )
        if (
            hashlib.sha256(blob).hexdigest() != digest
            or file_sha256(Path("robot_dynamics") / name) != digest
        ):
            raise ValueError("Training implementation changed before final evaluation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("d0_cache", type=Path)
    parser.add_argument("--d1-cache", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/formal_protocol.yaml")
    )
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--validation-only-check", action="store_true")
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if args.validation_only_check != (protocol["purpose"] == "workflow_check"):
        raise ValueError(
            "Workflow checks must use validation only; formal runs must use frozen tests"
        )
    plan = matrix(args.protocol)
    if json.loads((args.runs / "matrix.json").read_text()) != plan:
        raise ValueError("Frozen matrix mismatch")
    seal = json.loads((args.runs / "selection_seal.json").read_text())
    if seal["matrix_sha256"] != stable_digest(plan) or len(seal["selected"]) != len(
        plan["runs"]
    ):
        raise ValueError("All selections must be sealed before test access")
    for run, selected in zip(plan["runs"], seal["selected"]):
        metrics = validate_completed(run, args.runs / run["run_id"])
        if (
            selected["run_id"] != run["run_id"]
            or selected["checkpoint_sha256"] != metrics["best_checkpoint_sha256"]
        ):
            raise ValueError("Selected checkpoint differs from seal")
    manifest = json.loads(Path("reports/stage3/manifest.json").read_text())
    if file_sha256(args.d0_cache) != manifest["datasets"]["square_d0"]["cache_sha256"]:
        raise ValueError("D0 cache checksum mismatch")
    if args.validation_only_check:
        splits = {
            "validation_check_a": (args.d0_cache, manifest["d0"]["validation"][:2]),
            "validation_check_b": (args.d0_cache, manifest["d0"]["validation"][2:4]),
        }
    else:
        if (
            args.d1_cache is None
            or file_sha256(args.d1_cache)
            != manifest["datasets"]["square_d1"]["cache_sha256"]
        ):
            raise ValueError("D1 cache checksum mismatch")
        splits = {
            "d0_test": (args.d0_cache, manifest["d0"]["test"]),
            "d1_final_test": (args.d1_cache, manifest["d1_final_test"]),
        }
    thresholds = json.loads(
        Path(protocol["evaluation"]["motion_threshold_source"]).read_text()
    )["subsets"][protocol["evaluation"]["motion_threshold_subset"]][
        "motion_threshold_m_per_s"
    ]
    output = args.runs / "evaluation"
    output.mkdir(exist_ok=True)
    access_path = output / "access.json"
    identity = {
        "selection_seal_sha256": file_sha256(args.runs / "selection_seal.json"),
        "protocol_sha256": plan["protocol_sha256"],
        "validation_only_check": args.validation_only_check,
    }
    if access_path.exists():
        if json.loads(access_path.read_text())["identity"] != identity:
            raise ValueError("Evaluation identity changed")
    else:
        write_json(
            access_path,
            {
                "identity": identity,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **source_version(),
            },
        )
    jobs = plan["runs"] + [
        {"run_id": method, "model": method, "method": method}
        for method in protocol["evaluation"]["baselines"]
    ]
    for run in jobs:
        name = run["run_id"]
        baseline = run["method"] in ("hold", "constant_velocity")
        if baseline:
            stats = json.loads(Path("reports/stage3/normalization.json").read_text())[
                "subsets"
            ]["1.0"]
            checkpoint_hash, model = None, None
        else:
            directory = args.runs / name
            checkpoint = torch.load(
                directory / "best.pt", map_location="cpu", weights_only=True
            )
            check_source(checkpoint)
            checkpoint_hash = file_sha256(directory / "best.pt")
            stats = json.loads((directory / "normalization.json").read_text())
            setup(run["config"]["seed"], run["config"]["cpu_threads"])
            model = DynamicsPredictor(run["model"]).cuda()
            model.load_state_dict(checkpoint["model"])
        for split, (cache_path, ids) in splits.items():
            result_path = output / f"{name}_{split}.json"
            rows_path = output / f"{name}_{split}_episodes.csv"
            if result_path.exists():
                old = json.loads(result_path.read_text())
                if (
                    old["identity"] != identity
                    or old["checkpoint_sha256"] != checkpoint_hash
                    or old["episode_rows_sha256"] != file_sha256(rows_path)
                ):
                    raise ValueError("Saved evaluation changed")
                continue
            started = time.perf_counter()
            rows, prediction_hash = [], hashlib.sha256()
            with h5py.File(cache_path, "r") as cache:
                for episode in ids:
                    arrays = make_arrays(cache, [episode], stats)
                    raw = read_episode(cache[episode])
                    positions = np.stack(
                        [raw["eef_position"], raw["nut_position"]], axis=1
                    )
                    t, k = arrays["anchor"], arrays["horizon"]
                    truth = positions[t + k] - positions[t]
                    if baseline:
                        delta = (
                            np.zeros_like(truth)
                            if run["method"] == "hold"
                            else (positions[t] - positions[t - 1]) * k[:, None, None]
                        )
                    else:
                        normalized = predict(model, tensors(arrays, "cuda"))
                        prediction_hash.update(normalized.tobytes())
                        delta = stats["position_scale"] * normalized.reshape(
                            -1, 2, 3
                        ).astype(np.float64)
                    errors = 100 * np.linalg.norm(delta - truth, axis=-1)
                    if not np.isfinite(errors).all():
                        raise FloatingPointError("Nonfinite test predictions")
                    rows.extend(
                        grouped_errors(
                            errors,
                            episode,
                            arrays,
                            positions,
                            thresholds,
                            protocol["evaluation"]["near_object_distance_m"],
                        )
                    )
            save_rows(rows_path, rows)
            macro = [
                np.mean(
                    [
                        r["mean_error_cm"]
                        for r in rows
                        if r["episode"] == episode and r["segment"] == "all"
                    ]
                )
                for episode in ids
            ]
            write_json(
                result_path,
                {
                    "run_id": name,
                    "split": split,
                    "identity": identity,
                    "checkpoint_sha256": checkpoint_hash,
                    "episode_ids": ids,
                    "episode_rows_sha256": file_sha256(rows_path),
                    "prediction_sha256": prediction_hash.hexdigest()
                    if not baseline
                    else None,
                    "macro": mean_interval(macro),
                    "summary": summarize(rows),
                    "seconds": time.perf_counter() - started,
                    **source_version(),
                },
            )
            print(name, split, "evaluated", flush=True)
        if not baseline:
            diagnostic_path = output / f"{name}_diagnostics.json"
            if diagnostic_path.exists():
                previous = json.loads(diagnostic_path.read_text())
                if (
                    previous["identity"] != identity
                    or previous["checkpoint_sha256"] != checkpoint_hash
                ):
                    raise ValueError(
                        "Saved diagnostics differ from selected checkpoint"
                    )
                continue
            diagnostics = {
                "run_id": name,
                "identity": identity,
                "checkpoint_sha256": checkpoint_hash,
            }
            with h5py.File(args.d0_cache, "r") as cache:
                if run["model"] == "canonical_temporal_gnn":
                    samples, horizons = [], []
                    ids = manifest["d0"]["validation"]
                    for i, j in enumerate(
                        np.linspace(
                            0,
                            len(ids) - 1,
                            protocol["evaluation"]["coordinate_windows"],
                            dtype=int,
                        )
                    ):
                        ep = read_episode(cache[ids[j]])
                        k = (1, 5, 10)[i % 3]
                        eligible = anchors(len(ep["time"]), 4, k)
                        samples.append(
                            window(ep, int(eligible[(i * 17) % len(eligible)]), 4, k)
                        )
                        horizons.append(k)
                    diagnostics["coordinate_check"] = coordinate_checks(
                        model, samples, horizons, stats, "cuda"
                    )
                if run["method"] == "jepa":
                    train = make_arrays(cache, stats["train_episode_ids"], stats)
                    diagnostics["latent_checks"] = latent_checks(
                        directory, model, train, cache, stats, "cuda"
                    )
            write_json(diagnostic_path, diagnostics)
    write_json(
        output / "complete.json",
        {
            "identity": identity,
            "trained_models": len(plan["runs"]),
            "evaluated_predictors": len(jobs),
            "prediction_sets": len(jobs) * len(splits),
            "splits": list(splits),
            **source_version(),
        },
    )


if __name__ == "__main__":
    main()
