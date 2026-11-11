import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from robot_dynamics.analyze_equivariance import coordinate_checks
from robot_dynamics.baselines import mean_interval
from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.jepa import (
    JointEmbedding,
    encode,
    encoder_inputs,
    representation_stats,
    target_arrays,
)
from robot_dynamics.learning_data import make_arrays, error_rows
from robot_dynamics.learning_models import DynamicsPredictor
from robot_dynamics.train_learning import setup, tensors, predict, aggregate
from robot_dynamics.windows import read_episode, window, anchors


def subset(arrays, indices):
    return {
        k: v[indices] if isinstance(v, np.ndarray) else v for k, v in arrays.items()
    }


@torch.no_grad()
def sensitivity(model, arrays, scale, device):
    selected = subset(arrays, np.linspace(0, len(arrays["state"]) - 1, 1024, dtype=int))
    data = tensors(selected, device)
    base = predict(model, data).reshape(-1, 2, 3)
    target = selected["target"].reshape(-1, 2, 3)
    base_error = float(np.linalg.norm(base - target, axis=-1).mean() * scale * 100)
    result = {"windows": 1024, "base_sample_mean_error_cm": base_error}
    for condition in ("zero_actions", "permuted_past"):
        changed = {k: v.clone() for k, v in data.items()}
        if condition == "zero_actions":
            changed["context"][:, 48:118] = 0
        else:
            permutation = np.random.default_rng(707).permutation(len(base))
            changed["state"][:, :-1] = data["state"][permutation, :-1]
        output = predict(model, changed).reshape(-1, 2, 3)
        error = float(np.linalg.norm(output - target, axis=-1).mean() * scale * 100)
        result[condition] = {
            "mean_output_change_cm": float(
                np.linalg.norm(output - base, axis=-1).mean() * scale * 100
            ),
            "sample_mean_error_cm": error,
            "error_change_cm": error - base_error,
        }
    return result


@torch.no_grad()
def latent_checks(directory, model, train, cache, stats, device):
    checkpoint = torch.load(
        directory / "pretrained.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model"])
    auxiliary = JointEmbedding(model).to(device)
    auxiliary.load_state_dict(checkpoint["auxiliary"])
    selected = subset(
        train,
        np.linspace(
            0, len(train["state"]) - 1, min(1024, len(train["state"])), dtype=int
        ),
    )
    future = target_arrays(cache, selected, stats)
    data = tensors(selected, device)
    fs, fc = encoder_inputs(
        model,
        torch.from_numpy(future["future_state"]).to(device),
        torch.from_numpy(future["future_context"]).to(device),
    )
    online = encode(model, data["state"], data["context"])[:, :2]
    target = auxiliary.target.encode(fs, fc)[:, :2]
    prediction = auxiliary.predictor(online)
    representations = {
        "online_raw": representation_stats(online),
        "target_raw": representation_stats(target),
        "target_normalized": representation_stats(F.layer_norm(target, (64,))),
        "prediction_normalized": representation_stats(F.layer_norm(prediction, (64,))),
    }
    flags = [
        f"{branch}:{node}"
        for branch, rows in representations.items()
        for node, row in enumerate(rows)
        if row["mean_std"] < 0.01 or row["effective_rank"] < 2
    ]
    context = data["context"].clone()
    context[:, 48:118] = 0
    changed = auxiliary.predictor(encode(model, data["state"], context)[:, :2])
    return {
        "epoch": checkpoint["epoch"],
        "windows": len(online),
        "representations": representations,
        "gross_collapse_flags": flags,
        "normalized_prediction_zero_action_rms": float(
            (F.layer_norm(changed, (64,)) - F.layer_norm(prediction, (64,)))
            .square()
            .mean()
            .sqrt()
        ),
        "normalized_predictive_mse": float(
            F.mse_loss(F.layer_norm(prediction, (64,)), F.layer_norm(target, (64,)))
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("outputs/stage7"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path("reports/stage3/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    ids = manifest["d0"]["validation"]
    train_ids = manifest["d0"]["train_subsets"]["0.1"]
    cache_hash = file_sha256(args.cache)
    locations = {}
    for model, stage in [("temporal_gnn", 5), ("canonical_temporal_gnn", 6)]:
        locations[f"{model}_supervised30"] = (
            model,
            Path(f"outputs/stage{stage}/pilot_{model}"),
        )
        for method in ("supervised", "state", "jepa"):
            locations[f"{model}_{method}"] = (model, args.runs / f"{model}_{method}")
    runs, episode_scores, summary = {}, {}, []
    arrays = train = common_stats_hash = None
    with h5py.File(args.cache, "r") as cache:
        samples, horizons = [], []
        for i, j in enumerate(np.linspace(0, len(ids) - 1, 96, dtype=int)):
            ep = read_episode(cache[ids[j]])
            horizon = (1, 5, 10)[i % 3]
            eligible = anchors(len(ep["time"]), 4, horizon)
            samples.append(
                window(ep, int(eligible[(i * 17) % len(eligible)]), 4, horizon)
            )
            horizons.append(horizon)
        for key, (kind, directory) in locations.items():
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
                raise ValueError("Checkpoint/data provenance mismatch")
            for name, digest in identity["code"].items():
                blob = subprocess.check_output(
                    [
                        "git",
                        "show",
                        f"{checkpoint['source_commit']}:robot_dynamics/{name}",
                    ]
                )
                if hashlib.sha256(blob).hexdigest() != digest:
                    raise ValueError("Recorded source mismatch")
            if (
                metrics["train_ids"] != train_ids
                or metrics["evaluation_ids"] != ids
                or cfg["history"] != 4
                or cfg["horizons"] != [1, 5, 10]
                or cfg["seed"] != 0
            ):
                raise ValueError("Unequal pilot split/window/seed")
            if arrays is None:
                common_stats_hash = stats_hash
                arrays = make_arrays(cache, ids, stats)
                train = make_arrays(cache, train_ids, stats)
            if stats_hash != common_stats_hash:
                raise ValueError("Unequal normalization")
            setup(cfg["seed"], cfg["cpu_threads"])
            device = cfg["device"]
            model = DynamicsPredictor(kind).to(device)
            model.load_state_dict(checkpoint["model"])
            prediction = predict(model, tensors(arrays, device))
            if (
                hashlib.sha256(prediction.tobytes()).hexdigest()
                != metrics["prediction_sha256"]
            ):
                raise ValueError("Full prediction hash mismatch")
            old = key.endswith("supervised30")
            if not old:
                reload = json.loads((args.runs / f"{key}_reload.json").read_text())
                if (
                    reload["prediction_sha256"] != metrics["prediction_sha256"]
                    or reload["summary"] != metrics["summary"]
                ):
                    raise ValueError("Independent reload mismatch")
                if metrics["pretrain_steps"] + metrics["downstream_steps"] != 5480:
                    raise ValueError("Unexpected update budget")
            rows = error_rows(prediction, arrays, stats["position_scale"], kind)
            if aggregate(rows) != metrics["summary"]:
                raise ValueError("Recomputed physical summary mismatch")
            episode_scores[key] = np.array(
                [
                    np.mean(
                        [
                            float(r["mean_error_cm"])
                            for r in rows
                            if r["episode"] == episode
                        ]
                    )
                    for episode in ids
                ]
            )
            if any(sum(r["episode"] == episode for r in rows) != 6 for episode in ids):
                raise ValueError("Incomplete episode summaries")
            result = {
                k: metrics[k]
                for k in (
                    "parameter_count",
                    "best_epoch",
                    "best_score_cm",
                    "resources",
                    "source_commit",
                    "best_checkpoint_sha256",
                )
            }
            result.update(
                macro=mean_interval(episode_scores[key]),
                full_prediction_hash_exact=True,
                sensitivity=sensitivity(model, arrays, stats["position_scale"], device),
            )
            result["context_window_presentations"] = metrics["train_windows"] * (
                30 if old else 40
            )
            result["future_target_window_presentations"] = metrics["train_windows"] * (
                10 if key.endswith(("_jepa", "_state")) else 0
            )
            if not old:
                for field in (
                    "auxiliary_parameter_count",
                    "pretrain_steps",
                    "downstream_steps",
                    "supervised_updates_total",
                ):
                    result[field] = metrics[field]
            if kind.startswith("canonical_"):
                result["coordinate_check"] = coordinate_checks(
                    model, samples, horizons, stats, device
                )
                if not result["coordinate_check"]["within_bound"]:
                    raise ValueError("Full-chain coordinate check failed")
            if key.endswith("_jepa"):
                if (
                    file_sha256(directory / "pretrained.pt")
                    != metrics["pretrained_checkpoint_sha256"]
                ):
                    raise ValueError("Pretraining checkpoint mismatch")
                result["latent_checks"] = latent_checks(
                    directory, model, train, cache, stats, device
                )
            runs[key] = result
            summary.extend({"run": key, **row} for row in metrics["summary"])
            print(
                key,
                result["macro"],
                result.get("latent_checks", {}).get("gross_collapse_flags"),
                flush=True,
            )
    contrasts = {}
    for kind in ("temporal_gnn", "canonical_temporal_gnn"):
        for control in ("state", "supervised", "supervised30"):
            contrasts[f"{kind}:jepa_minus_{control}"] = mean_interval(
                episode_scores[f"{kind}_jepa"] - episode_scores[f"{kind}_{control}"]
            )
    contrasts["interaction_jepa_vs_supervised40"] = mean_interval(
        (
            episode_scores["canonical_temporal_gnn_jepa"]
            - episode_scores["canonical_temporal_gnn_supervised"]
        )
        - (
            episode_scores["temporal_gnn_jepa"]
            - episode_scores["temporal_gnn_supervised"]
        )
    )
    result = {
        "runs": runs,
        "summary": summary,
        "paired_contrasts": contrasts,
        "normalization_sha256": common_stats_hash,
        "scope": "D0 validation only; seed 0, 80 training episodes. Episode bootstrap does not measure seed uncertainty. Negative differences favor JEPA.",
        **source_version(),
    }
    (args.output / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
