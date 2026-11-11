import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import torch

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.frame_transforms import (
    transform_sample,
    transform_vectors,
    yaw_rotation,
)
from robot_dynamics.learning_data import make_arrays, encode_sample
from robot_dynamics.learning_models import DynamicsPredictor
from robot_dynamics.train_learning import setup, tensors, predict
from robot_dynamics.windows import read_episode, window, anchors


def encode_samples(samples, stats, horizons):
    encoded = [encode_sample(s, stats, k) for s, k in zip(samples, horizons)]
    return {
        "state": np.stack([x[0] for x in encoded]),
        "context": np.stack([x[1] for x in encoded]),
        "target": np.stack([x[2] for x in encoded]),
    }


def current_positions(samples):
    return np.stack(
        [
            np.stack([s["history"][k][-1] for k in ("eef_position", "nut_position")])
            for s in samples
        ]
    )


def coordinate_checks(model, samples, horizons, stats, device):
    data = encode_samples(samples, stats, horizons)
    base = current_positions(samples) + stats["position_scale"] * predict(
        model, tensors(data, device)
    ).reshape(-1, 2, 3)
    target = current_positions(samples) + np.stack([s["target_delta"] for s in samples])
    original_error = float(100 * np.linalg.norm(base - target, axis=-1).mean())
    rng = np.random.default_rng(606)
    transforms = [(0.0, np.array([0.2, -0.3, 0.1])), (1.1, np.zeros(3))]
    transforms += [
        (float(rng.uniform(-np.pi, np.pi)), rng.uniform(-0.5, 0.5, 3))
        for _ in range(14)
    ]
    records = []
    for angle, translation in transforms:
        rotation = yaw_rotation(angle)
        changed = [transform_sample(s, rotation, translation) for s in samples]
        changed_data = encode_samples(changed, stats, horizons)
        future = current_positions(changed) + stats["position_scale"] * predict(
            model, tensors(changed_data, device)
        ).reshape(-1, 2, 3)
        expected = transform_vectors(base, rotation) + translation
        error = future - expected
        records.append(
            {
                "yaw_radians": angle,
                "translation_m": translation.tolist(),
                "max_abs_m": float(np.abs(error).max()),
                "max_l2_m": float(np.linalg.norm(error, axis=-1).max()),
                "mean_l2_m": float(np.linalg.norm(error, axis=-1).mean()),
                "transformed_prediction_error_cm": float(
                    100
                    * np.linalg.norm(
                        future - (transform_vectors(target, rotation) + translation),
                        axis=-1,
                    ).mean()
                ),
            }
        )
    maximum = max(r["max_abs_m"] for r in records)
    return {
        "windows": len(samples),
        "transformations": len(transforms),
        "acceptance_max_abs_m": 2e-5,
        "max_abs_m": maximum,
        "within_bound": maximum <= 2e-5,
        "original_sample_mean_error_cm": original_error,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("outputs/stage6"))
    parser.add_argument("--stage4", type=Path, default=Path("outputs/stage4"))
    parser.add_argument("--stage5", type=Path, default=Path("outputs/stage5"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path("reports/stage3/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    cache_hash = file_sha256(args.cache)
    ids = manifest["d0"]["validation"]
    samples, horizons, sample_ids = [], [], []
    with h5py.File(args.cache, "r") as cache:
        for i, j in enumerate(np.linspace(0, len(ids) - 1, 96, dtype=int)):
            name = ids[j]
            ep = read_episode(cache[name])
            k = (1, 5, 10)[i % 3]
            eligible = anchors(len(ep["time"]), 4, k)
            t = int(eligible[(i * 17) % len(eligible)])
            samples.append(window(ep, t, 4, k))
            horizons.append(k)
            sample_ids.append({"episode": name, "horizon": k, "anchor": t})
    runs, diagnostics, rows = {}, {}, []
    locations = {
        "gru": args.stage4,
        "temporal_gnn": args.stage5,
        "canonical_gru": args.runs,
        "canonical_temporal_gnn": args.runs,
    }
    common_cfg, common_stats_hash, arrays = None, None, None
    for kind, parent in locations.items():
        directory = parent / f"pilot_{kind}"
        metrics = json.loads((directory / "metrics.json").read_text())
        checkpoint = torch.load(
            directory / "best.pt", map_location="cpu", weights_only=True
        )
        identity = checkpoint["identity"]
        cfg = identity["config"]
        stats = json.loads((directory / "normalization.json").read_text())
        stats_hash = file_sha256(directory / "normalization.json")
        if (
            file_sha256(directory / "best.pt") != metrics["best_checkpoint_sha256"]
            or cache_hash != identity["cache_sha256"]
            or file_sha256(manifest_path) != identity["manifest_sha256"]
            or stats_hash != identity["normalization_sha256"]
        ):
            raise ValueError("Checkpoint/data/normalizer mismatch")
        for name, digest in identity["code"].items():
            blob = subprocess.check_output(
                ["git", "show", f"{checkpoint['source_commit']}:robot_dynamics/{name}"]
            )
            if hashlib.sha256(blob).hexdigest() != digest:
                raise ValueError("Recorded source mismatch")
        if common_cfg is None:
            common_cfg, common_stats_hash = cfg, stats_hash
            with h5py.File(args.cache, "r") as cache:
                arrays = make_arrays(cache, ids, stats)
        if cfg != common_cfg or stats_hash != common_stats_hash:
            raise ValueError("Unequal pilot configuration or normalization")
        if (
            metrics["evaluation_ids"] != ids
            or metrics["train_ids"] != manifest["d0"]["train_subsets"]["0.1"]
            or metrics["resources"]["optimizer_steps_total"] != 4110
        ):
            raise ValueError("Split/update budget mismatch")
        setup(cfg["seed"], cfg["cpu_threads"])
        model = DynamicsPredictor(kind).to(cfg["device"])
        model.load_state_dict(checkpoint["model"])
        prediction = predict(model, tensors(arrays, cfg["device"]))
        if (
            hashlib.sha256(prediction.tobytes()).hexdigest()
            != metrics["prediction_sha256"]
        ):
            raise ValueError("Full prediction hash differs from recorded pilot")
        if kind.startswith("canonical_"):
            reload = json.loads((parent / f"pilot_{kind}_reload.json").read_text())
            if (
                reload["prediction_sha256"] != metrics["prediction_sha256"]
                or reload["summary"] != metrics["summary"]
            ):
                raise ValueError("Standalone reload differs")
        check = coordinate_checks(model, samples, horizons, stats, cfg["device"])
        diagnostics[kind] = {"full_prediction_hash_exact": True, **check}
        if kind.startswith("canonical_") and not check["within_bound"]:
            (args.output / f"failed_{kind}.json").write_text(
                json.dumps(check, indent=2) + "\n"
            )
            raise ValueError(
                f"Full-chain equivariance failed: {kind}: {check['max_abs_m']}"
            )
        runs[kind] = {
            k: metrics[k]
            for k in (
                "parameter_count",
                "best_epoch",
                "best_score_cm",
                "best_train_mse",
                "train_summary",
                "resources",
                "logs",
                "source_commit",
                "working_tree_dirty",
                "best_checkpoint_sha256",
            )
        }
        rows.extend({"model": kind, **r} for r in metrics["summary"])
        print(kind, check["max_abs_m"], metrics["best_score_cm"], flush=True)
    result = {
        "runs": runs,
        "summary": rows,
        "coordinate_checks": diagnostics,
        "coordinate_samples": sample_ids,
        "physical_ood": "Not evaluated; D1 and D0 test remain reserved for stage 8.",
        "normalization_sha256": common_stats_hash,
        **source_version(),
    }
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
