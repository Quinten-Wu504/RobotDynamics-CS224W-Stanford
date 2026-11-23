"""Small Square reproduction example; never evaluates the formal test split."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.splits import validate_split


def prepare(cache_path, output):
    original = json.loads(Path("reports/stage3/manifest.json").read_text())
    train = original["d0"]["train"][:8]
    validation = original["d0"]["validation"][:2]
    output.mkdir(parents=True, exist_ok=False)
    subset = output / "square_demo.hdf5"
    with h5py.File(cache_path, "r") as source:
        if (
            not source.attrs["complete"]
            or source.attrs["source_sha256"]
            != original["datasets"]["square_d0"]["source_sha256"]
        ):
            raise ValueError(
                "Expected a completed cache from the pinned Square D0 source"
            )
        with h5py.File(subset, "x") as target:
            target.attrs.update(dict(source.attrs))
            target.attrs["complete"] = False
            target.attrs["purpose"] = "small reproduction demo, not formal results"
            for name in train + validation:
                source.copy(name, target)
            target.attrs["complete"] = True
    split = {
        "train": train,
        "validation": validation,
        "test": [],
        "train_subsets": {
            str(f): train[: int(f * len(train))] for f in (0.1, 0.25, 1.0)
        },
    }
    validate_split(split, train + validation)
    result = {
        "purpose": "8 training / 2 validation motions; five-epoch pipeline demonstration",
        "d0": split,
        "datasets": {
            "square_d0": {
                "cache_sha256": file_sha256(subset),
                "source_sha256": original["datasets"]["square_d0"]["source_sha256"],
            }
        },
        "parent_manifest_sha256": file_sha256(Path("reports/stage3/manifest.json")),
        "parent_cache_sha256": file_sha256(cache_path),
        **source_version(),
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"train": train, "validation": validation, "test": []}, indent=2))


def plot(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from robot_dynamics.graph_models import scene_edges, scene_nodes
    from robot_dynamics.learning_data import make_arrays
    from robot_dynamics.learning_models import DynamicsPredictor
    from robot_dynamics.train_learning import predict, setup, tensors

    run = output / "temporal_gnn"
    metrics = json.loads((run / "metrics.json").read_text())
    reload = json.loads((output / "reload.json").read_text())
    if reload["prediction_sha256"] != metrics["prediction_sha256"]:
        raise ValueError("Standalone checkpoint predictions differ")
    cfg = metrics["config"]
    setup(cfg["seed"], cfg["cpu_threads"])
    stats = json.loads((run / "normalization.json").read_text())
    checkpoint = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
    with h5py.File(output / "square_demo.hdf5", "r") as cache:
        arrays = make_arrays(cache, metrics["evaluation_ids"], stats)
        name = arrays["ids"][0]
        positions = np.stack(
            [cache[name][k][:] for k in ("eef_position", "nut_position")], axis=1
        )
        times = cache[name]["time"][:]
    data = tensors(arrays, cfg["device"])
    model = DynamicsPredictor("temporal_gnn").to(cfg["device"])
    model.load_state_dict(checkpoint["model"])
    prediction = predict(model, data).reshape(-1, 2, 3) * stats["position_scale"]
    chosen = (arrays["episode"] == 0) & (arrays["horizon"] == 10)
    anchors = arrays["anchor"][chosen]
    forecast = positions[anchors] + prediction[chosen]
    target = positions[anchors + 10]
    x = times[anchors + 10] - times[0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for entity in range(2):
        for coordinate in range(3):
            ax = axes[entity, coordinate]
            ax.plot(x, target[:, entity, coordinate], color="black", label="Reference")
            ax.plot(x, forecast[:, entity, coordinate], label="Demo GNN")
            ax.set(
                title=f"{('End effector', 'Nut')[entity]} {'xyz'[coordinate]}",
                xlabel="Target time (s)",
                ylabel="Position (m)",
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(f"{name}: direct 0.5 s predictions, not a rollout or formal result")
    fig.savefig(output / "trajectory.png", dpi=150)
    plt.close(fig)
    nodes = scene_nodes(data["state"][:1], data["context"][:1])[0, -1, :, :3]
    nodes = nodes.cpu().numpy() * stats["position_scale"] + stats["position_center"]
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    for start, end in scene_edges().T.tolist():
        ax.annotate(
            "",
            xy=nodes[end, :2],
            xytext=nodes[start, :2],
            arrowprops={"arrowstyle": "->", "alpha": 0.35},
        )
    ax.scatter(nodes[:, 0], nodes[:, 1], s=80)
    for point, label in zip(nodes, ("End effector", "Nut", "Peg", "Table")):
        ax.annotate(label, point[:2], xytext=(5, 5), textcoords="offset points")
    ax.set(
        xlabel="World x (m)",
        ylabel="World y (m)",
        title="Four-node scene graph: first validation window",
    )
    ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(output / "graph.png", dpi=150)
    plt.close(fig)
    verification = {
        "purpose": "small pipeline demonstration, not a formal accuracy comparison",
        "standalone_reload_exact": True,
        "prediction_sha256": metrics["prediction_sha256"],
        "checkpoint_sha256": file_sha256(run / "best.pt"),
        "train_ids": metrics["train_ids"],
        "validation_ids": metrics["evaluation_ids"],
        "test_evaluated": False,
        "nodes": 4,
        "directed_edges": int(scene_edges().shape[1]),
        "resources": metrics["resources"],
        **source_version(),
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("cache", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    figures = commands.add_parser("plot")
    figures.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.cache, args.output)
    else:
        plot(args.output)


if __name__ == "__main__":
    main()
