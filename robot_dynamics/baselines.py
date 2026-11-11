import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np

from robot_dynamics.config import horizon_steps, load_config, unresolved_audit
from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.splits import validate_split
from robot_dynamics.windows import (
    anchors,
    position_error_cm,
    predict_positions,
    read_episode,
    window,
    normalize_window,
)


def mean_interval(values, seed=224, resamples=2000):
    values = np.asarray(values)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(
        axis=1
    )
    return {
        "mean_cm": float(values.mean()),
        "ci95_cm": np.quantile(boot, [0.025, 0.975]).tolist(),
    }


def evaluate_episode(positions, history, horizons, dt, thresholds):
    rows = []
    for horizon in horizons:
        indices = anchors(len(positions), history, horizon)
        if not len(indices):
            raise ValueError("Episode is too short for configured window")
        speed = (
            np.linalg.norm(positions[indices] - positions[indices - 1], axis=-1) / dt
        )
        for method in ("hold", "constant_velocity"):
            errors = position_error_cm(
                predict_positions(positions, indices, horizon, method),
                positions[indices + horizon],
            )
            for entity in range(2):
                low = speed[:, entity] <= thresholds[entity]
                for segment, mask in [
                    ("all", np.ones(len(indices), dtype=bool)),
                    ("low_motion", low),
                    ("moving", ~low),
                ]:
                    if np.any(mask):
                        rows.append(
                            {
                                "model": method,
                                "entity": ("end_effector", "square_nut")[entity],
                                "horizon_steps": horizon,
                                "segment": segment,
                                "windows": int(mask.sum()),
                                "mean_error_cm": float(errors[mask, entity].mean()),
                            }
                        )
    return rows


def summarize(rows):
    summaries = []
    keys = sorted({(r["entity"], r["horizon_steps"], r["segment"]) for r in rows})
    for entity, horizon, segment in keys:
        subset = [
            r
            for r in rows
            if (r["entity"], r["horizon_steps"], r["segment"])
            == (entity, horizon, segment)
        ]
        by_model = {}
        for method in ("hold", "constant_velocity"):
            selected = sorted(
                (r for r in subset if r["model"] == method), key=lambda r: r["episode"]
            )
            values = [r["mean_error_cm"] for r in selected]
            by_model[method] = values
            summaries.append(
                {
                    "entity": entity,
                    "horizon_steps": horizon,
                    "segment": segment,
                    "model": method,
                    "episodes": len(selected),
                    "windows": sum(r["windows"] for r in selected),
                    **mean_interval(values),
                }
            )
        difference = np.asarray(by_model["constant_velocity"]) - by_model["hold"]
        summaries.append(
            {
                "entity": entity,
                "horizon_steps": horizon,
                "segment": segment,
                "model": "constant_velocity_minus_hold",
                "episodes": len(difference),
                **mean_interval(difference),
            }
        )
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/square.yaml"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = json.loads(args.manifest.read_text())
    norm = json.loads(args.normalization.read_text())
    config = load_config(args.config)
    if unresolved_audit(config) or config["history_steps"] < 2:
        raise ValueError("Audited data and at least two history frames required")
    if norm["manifest_sha256"] != file_sha256(args.manifest):
        raise ValueError("Normalization belongs to a different split")
    expected = manifest["datasets"]["square_d0"]
    if (
        file_sha256(args.cache) != expected["cache_sha256"]
        or norm["cache_sha256"] != expected["cache_sha256"]
    ):
        raise ValueError("Cache provenance mismatch")
    stats = norm["subsets"]["1.0"]
    if stats["train_episode_ids"] != manifest["d0"]["train"]:
        raise ValueError("Motion thresholds were not fitted to the training split")
    horizons = [1] + horizon_steps(
        config["horizons_seconds"], config["control_frequency_hz"]
    )
    rows = []
    checked_windows = 0
    with h5py.File(args.cache, "r") as cache:
        validate_split(manifest["d0"], list(cache))
        dt = float(cache.attrs["dt_seconds"])
        if not cache.attrs["complete"] or not np.isclose(
            dt, 1 / config["control_frequency_hz"]
        ):
            raise ValueError("Cache completion or timing mismatch")
        for name in manifest["d0"]["validation"]:
            g = cache[name]
            episode = read_episode(g)
            positions = np.stack([g["eef_position"][:], g["nut_position"][:]], axis=1)
            for k in horizons:
                valid = anchors(len(positions), config["history_steps"], k)
                for t in (valid[0], valid[-1]):
                    sample = window(episode, int(t), config["history_steps"], k)
                    normalized = normalize_window(sample, stats)
                    np.testing.assert_allclose(
                        normalized["target_delta"] * stats["position_scale"],
                        positions[t + k] - positions[t],
                        atol=1e-12,
                    )
                    np.testing.assert_array_equal(
                        sample["actions"], episode["actions"][t : t + k]
                    )
                    checked_windows += 1
            for row in evaluate_episode(
                positions,
                config["history_steps"],
                horizons,
                dt,
                stats["motion_threshold_m_per_s"],
            ):
                rows.append({"episode": name, **row})
    with (args.output_dir / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    result = {
        "real_window_boundary_checks": checked_windows,
        "split": "d0_validation",
        "episode_ids": manifest["d0"]["validation"],
        "config": config,
        "config_sha256": file_sha256(args.config),
        "manifest_sha256": file_sha256(args.manifest),
        "normalization_sha256": file_sha256(args.normalization),
        "source_dataset_sha256": expected["source_sha256"],
        "cache_sha256": expected["cache_sha256"],
        "threshold_m_per_s": stats["motion_threshold_m_per_s"],
        "methods": {"hold": "p[t]", "constant_velocity": "p[t] + k * (p[t] - p[t-1])"},
        "conditioning": "Both baselines ignore the available planned actions; same H=4 eligible windows.",
        "aggregation": "mean Euclidean cm per episode, then equal episode mean; empty segments omitted",
        "bootstrap": {
            "unit": "episode",
            "resamples": 2000,
            "seed": 224,
            "interval": "percentile 95%",
        },
        "training": "none; deterministic baselines, no repeated training seeds or checkpoint",
        "runtime_seconds": time.perf_counter() - started,
        "summary": summary,
        **source_version(),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "episodes": len(result["episode_ids"]),
                "runtime_seconds": result["runtime_seconds"],
                "summary": [r for r in result["summary"] if r["segment"] == "all"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
