import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import torch

from robot_dynamics.baselines import mean_interval
from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.graph_models import GRAPH_MODELS, scene_nodes
from robot_dynamics.learning_data import make_arrays
from robot_dynamics.learning_models import DynamicsPredictor
from robot_dynamics.plot_graphs import plot_results
from robot_dynamics.train_learning import setup, tensors, predict


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def graph_checks(model, data):
    indices = torch.linspace(
        0, len(data["state"]) - 1, 8, device=data["state"].device
    ).long()
    state, context = data["state"][indices], data["context"][indices]
    graph = model.graph
    with torch.no_grad():
        reference = model(state, context)
        individual = torch.cat(
            [model(s[None], c[None]) for s, c in zip(state, context)]
        )
        nodes = scene_nodes(state, context)
        permutation = torch.tensor([2, 0, 3, 1], device=state.device)
        permuted = graph.forward_graph(
            nodes[:, :, permutation],
            context,
            torch.argsort(permutation)[graph.edges],
            permutation,
        )
        torch.testing.assert_close(reference, individual, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(reference, permuted, rtol=1e-5, atol=1e-6)
        altered = nodes.clone()
        # Swap measured nut orientations across samples, leaving every other field fixed.
        altered[:, :, 1, 3:12] = nodes.roll(3, 0)[:, :, 1, 3:12]
        neighbor_effect = (
            (reference[:, :3] - graph.forward_graph(altered, context)[:, :3])
            .abs()
            .max()
            .item()
        )
        enabled = graph.messages
        graph.messages = False
        closed_effect = (
            (
                graph.forward_graph(nodes, context)[:, :3]
                - graph.forward_graph(altered, context)[:, :3]
            )
            .abs()
            .max()
            .item()
        )
        graph.messages = enabled
        if closed_effect != 0 or (enabled and neighbor_effect <= 0):
            raise ValueError("Unexpected trained neighbor pathway")
    return {
        "batch_max_abs": (reference - individual).abs().max().item(),
        "permutation_max_abs": (reference - permuted).abs().max().item(),
        "neighbor_rotation_max_abs": neighbor_effect,
        "messages_closed_neighbor_max_abs": closed_effect,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("outputs/stage5"))
    parser.add_argument("--baseline-runs", type=Path, default=Path("outputs/stage4"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path("reports/stage3/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    cache_hash = file_sha256(args.cache)
    baseline = json.loads(Path("reports/stage3/evaluation/metrics.json").read_text())
    thresholds = json.loads(Path("reports/stage3/normalization.json").read_text())[
        "subsets"
    ]["1.0"]["motion_threshold_m_per_s"]
    models = ("current_mlp", "history_mlp", "gru") + GRAPH_MODELS
    episodes, overall, checks, runs = [], [], {}, {}
    cached_arrays = {}
    for kind in models:
        directory = (
            args.runs if kind in GRAPH_MODELS else args.baseline_runs
        ) / f"pilot_{kind}"
        metrics = json.loads((directory / "metrics.json").read_text())
        checkpoint = torch.load(
            directory / "best.pt", map_location="cpu", weights_only=True
        )
        identity = checkpoint["identity"]
        cfg = identity["config"]
        if file_sha256(directory / "best.pt") != metrics["best_checkpoint_sha256"]:
            raise ValueError("Checkpoint differs from recorded pilot")
        if (
            cache_hash != identity["cache_sha256"]
            or file_sha256(manifest_path) != identity["manifest_sha256"]
        ):
            raise ValueError("Input provenance mismatch")
        # Old baseline checkpoints are checked against their immutable source revision.
        for name, digest in identity["code"].items():
            blob = subprocess.check_output(
                ["git", "show", f"{checkpoint['source_commit']}:robot_dynamics/{name}"]
            )
            if hashlib.sha256(blob).hexdigest() != digest:
                raise ValueError("Recorded source hash mismatch")
        if (
            cfg
            != json.loads((args.baseline_runs / "pilot_gru/metrics.json").read_text())[
                "config"
            ]
        ):
            raise ValueError("Pilot budget mismatch")
        if (
            metrics["evaluation_ids"] != manifest["d0"]["validation"]
            or metrics["train_ids"] != manifest["d0"]["train_subsets"]["0.1"]
        ):
            raise ValueError("Pilot split mismatch")
        stats = json.loads((directory / "normalization.json").read_text())
        digest = file_sha256(directory / "normalization.json")
        if digest != identity["normalization_sha256"]:
            raise ValueError("Normalizer mismatch")
        if digest not in cached_arrays:
            with h5py.File(args.cache, "r") as cache:
                cached_arrays[digest] = make_arrays(
                    cache, metrics["evaluation_ids"], stats
                )
        arrays = cached_arrays[digest]
        setup(cfg["seed"], cfg["cpu_threads"])
        model = DynamicsPredictor(kind).to(cfg["device"])
        model.load_state_dict(checkpoint["model"])
        data = tensors(arrays, cfg["device"])
        prediction = predict(model, data)
        if (
            hashlib.sha256(prediction.tobytes()).hexdigest()
            != metrics["prediction_sha256"]
        ):
            raise ValueError(
                "Prediction mismatch; do not reuse old metrics with changed behavior"
            )
        checks[kind] = {"full_prediction_hash_exact": True}
        if kind in GRAPH_MODELS:
            reloaded = json.loads((args.runs / f"pilot_{kind}_reload.json").read_text())
            if (
                reloaded["prediction_sha256"] != metrics["prediction_sha256"]
                or reloaded["summary"] != metrics["summary"]
            ):
                raise ValueError("Standalone reload mismatch")
            checks[kind].update(graph_checks(model, data))
        runs[kind] = {
            key: metrics[key]
            for key in (
                "parameter_count",
                "best_epoch",
                "best_score_cm",
                "resources",
                "train_summary",
                "best_train_mse",
                "source_commit",
                "working_tree_dirty",
                "best_checkpoint_sha256",
                "logs",
            )
        }
        for row in metrics["summary"]:
            ref = next(
                r
                for r in baseline["summary"]
                if r["model"] == "hold"
                and r["segment"] == "all"
                and r["entity"] == row["entity"]
                and r["horizon_steps"] == row["horizon_steps"]
            )
            if row["windows"] != ref["windows"]:
                raise ValueError("Window count mismatch")
            overall.append({"model": kind, **row})
        scale = stats["position_scale"]
        errors = (
            100
            * scale
            * np.linalg.norm((prediction - arrays["target"]).reshape(-1, 2, 3), axis=-1)
        )
        positions = arrays["state"][:, :, :6].reshape(-1, 4, 2, 3)
        speed = (
            scale * np.linalg.norm(positions[:, -1] - positions[:, -2], axis=-1) / 0.05
        )
        near = (
            scale * np.linalg.norm(positions[:, -1, 0] - positions[:, -1, 1], axis=-1)
            <= 0.05
        )
        for i, name in enumerate(arrays["ids"]):
            for horizon in (1, 5, 10):
                eligible = (arrays["episode"] == i) & (arrays["horizon"] == horizon)
                for j, entity in enumerate(("end_effector", "square_nut")):
                    low = speed[:, j] <= thresholds[j]
                    for segment, mask in (
                        ("low_motion", low),
                        ("moving", ~low),
                        ("near_object", near),
                        ("far_object", ~near),
                    ):
                        select = eligible & mask
                        if select.any():
                            episodes.append(
                                {
                                    "model": kind,
                                    "episode": name,
                                    "entity": entity,
                                    "horizon_steps": horizon,
                                    "segment": segment,
                                    "windows": int(select.sum()),
                                    "mean_error_cm": float(errors[select, j].mean()),
                                }
                            )
    groups, differences = [], []
    for entity, horizon, segment in sorted(
        {(r["entity"], r["horizon_steps"], r["segment"]) for r in episodes}
    ):
        selections = {}
        for kind in models:
            rows = sorted(
                [
                    r
                    for r in episodes
                    if (r["model"], r["entity"], r["horizon_steps"], r["segment"])
                    == (kind, entity, horizon, segment)
                ],
                key=lambda r: r["episode"],
            )
            selections[kind] = rows
            groups.append(
                {
                    "model": kind,
                    "entity": entity,
                    "horizon_steps": horizon,
                    "segment": segment,
                    "episodes": len(rows),
                    "windows": sum(r["windows"] for r in rows),
                    **mean_interval([r["mean_error_cm"] for r in rows]),
                }
            )
        for left, right in (
            ("gnn", "current_mlp"),
            ("temporal_gnn", "gru"),
            ("temporal_gnn", "history_mlp"),
            ("temporal_no_messages", "temporal_gnn"),
            ("temporal_full_edges", "temporal_gnn"),
        ):
            a, b = selections[left], selections[right]
            if [(r["episode"], r["windows"]) for r in a] != [
                (r["episode"], r["windows"]) for r in b
            ]:
                raise ValueError("Group membership differs")
            differences.append(
                {
                    "comparison": f"{left}_minus_{right}",
                    "entity": entity,
                    "horizon_steps": horizon,
                    "segment": segment,
                    "episodes": len(a),
                    **mean_interval(
                        [x["mean_error_cm"] - y["mean_error_cm"] for x, y in zip(a, b)]
                    ),
                }
            )
    write_csv(args.output / "group_episodes.csv", episodes)
    write_csv(
        args.output / "overall.csv",
        [{**r, "ci95_cm": str(r["ci95_cm"])} for r in overall],
    )
    result = {
        "overall": overall,
        "groups": groups,
        "paired_differences": differences,
        "checks": checks,
        "runs": runs,
        "motion_threshold_m_per_s": thresholds,
        "near_object_distance_m": 0.05,
        **source_version(),
    }
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_results(result, args.output)


if __name__ == "__main__":
    main()
