import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from robot_dynamics.data_audit import source_version
from robot_dynamics.formal import write_json
from robot_dynamics.grail_results import asset_values, paired_interval, summarize


MODELS = (
    "history_mlp",
    "gru",
    "temporal_gnn",
    "temporal_no_messages",
    "hold",
    "constant_velocity",
)
LABELS = (
    "History MLP",
    "GRU",
    "Temporal GNN",
    "Messages disabled",
    "Position hold",
    "Constant velocity",
)


def read_rows(path, digest):
    content = (
        path.read_bytes()
        if path.exists()
        else gzip.decompress(path.with_suffix(".csv.gz").read_bytes())
    )
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"CSV hash differs: {path}")
    return list(csv.DictReader(io.StringIO(content.decode())))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(runs, manifest):
    evaluation = runs / "evaluation"
    complete = json.loads((evaluation / "complete.json").read_text())
    plan = json.loads((runs / "matrix.json").read_text())
    expected = {r["run_id"] + ".json" for r in plan["runs"]} | {
        "hold.json",
        "constant_velocity.json",
    }
    if set(complete["results"]) != expected or len(complete["results"]) != 14:
        raise ValueError("Incomplete test evaluation matrix")
    values, records, grouped, trajectories = (
        {},
        defaultdict(list),
        defaultdict(list),
        {},
    )
    for filename in complete["results"]:
        result = json.loads((evaluation / filename).read_text())
        if (
            result["role"] != "test"
            or result["identity"] != complete["identity"]
            or set(result["episode_ids"]) != set(manifest["splits"]["test"])
        ):
            raise ValueError("Test role, identity or motion membership mismatch")
        stem = Path(filename).stem
        rows = read_rows(
            evaluation / f"{stem}_episodes.csv", result["episode_rows_sha256"]
        )
        if {r["episode"] for r in rows} != set(result["episode_ids"]):
            raise ValueError("Episode rows do not cover the test split")
        if any(r["asset"] != manifest["audit"][r["episode"]]["asset"] for r in rows):
            raise ValueError("Asset membership changed")
        assets, macro = asset_values(rows)
        if (
            assets != result["asset_ids"]
            or not np.allclose(macro, result["asset_means_cm"], rtol=0, atol=1e-12)
            or not np.isclose(macro.mean(), result["mean_cm"], rtol=0, atol=1e-12)
        ):
            raise ValueError("Published macro differs from episode rows")
        summary = summarize(rows)
        if summary != result["summary"]:
            raise ValueError("Published group summaries differ from episode rows")
        model, seed = result["model"], result["seed"]
        if (model, seed) in values:
            raise ValueError("Duplicate model/seed")
        values[model, seed] = (assets, macro)
        records[model].append((seed, float(macro.mean())))
        for row in summary:
            grouped[model, row["entity"], row["horizon_steps"], row["segment"]].append(
                row
            )
        if "trajectory_rows_sha256" in result:
            trajectories[model] = read_rows(
                evaluation / f"{stem}_trajectories.csv",
                result["trajectory_rows_sha256"],
            )
    summary = []
    for model in MODELS:
        entries = sorted(records[model])
        if [seed for seed, _ in entries] != (
            [None] if model in MODELS[4:] else [0, 1, 2]
        ):
            raise ValueError("Missing seeds")
        v = np.array([x for _, x in entries])
        summary.append(
            {
                "model": model,
                "seeds": [s for s, _ in entries],
                "seed_means_cm": v.tolist(),
                "mean_cm": float(v.mean()),
                "seed_sd_cm": float(v.std(ddof=1)) if len(v) > 1 else None,
            }
        )
    groups = []
    for (model, entity, horizon, segment), entries in sorted(grouped.items()):
        populations = {(r["episodes"], r["assets"], r["windows"]) for r in entries}
        if len(populations) != 1:
            raise ValueError("Group populations differ across seeds")
        episodes, assets, windows = populations.pop()
        means = [r["mean_cm"] for r in entries]
        groups.append(
            {
                "model": model,
                "entity": entity,
                "horizon_steps": horizon,
                "segment": segment,
                "episodes": episodes,
                "assets": assets,
                "windows": windows,
                "mean_cm": float(np.mean(means)),
                "seed_sd_cm": float(np.std(means, ddof=1)) if len(means) > 1 else None,
            }
        )
    contrasts = []
    for control in ("history_mlp", "gru", "temporal_no_messages"):
        differences = []
        for seed in (0, 1, 2):
            graph_assets, graph = values["temporal_gnn", seed]
            control_assets, baseline = values[control, seed]
            if graph_assets != control_assets:
                raise ValueError("Unpaired assets")
            differences.append(graph - baseline)
        contrasts.append(
            {
                "comparison": f"temporal_gnn-minus-{control}",
                **paired_interval(differences),
            }
        )
    costs = []
    for run in plan["runs"]:
        metrics = json.loads((runs / run["run_id"] / "metrics.json").read_text())
        costs.append(
            {
                "run_id": run["run_id"],
                "model": run["model"],
                "seed": run["config"]["seed"],
                "parameter_count": metrics["parameter_count"],
                "optimizer_steps": metrics["optimizer_steps_total"],
                "best_epoch": metrics["best_epoch"],
                "update_minutes": sum(r["update_seconds"] for r in metrics["logs"])
                / 60,
                "validation_minutes": sum(
                    r["validation_seconds"] for r in metrics["logs"]
                )
                / 60,
                "peak_cuda_gib": metrics["resources"]["peak_cuda_allocated_bytes"]
                / 2**30,
                "rss_gib": metrics["resources"]["max_observed_rss_bytes"] / 2**30,
            }
        )
    return {
        "summary": summary,
        "groups": groups,
        "paired_contrasts": contrasts,
        "costs": costs,
        "bootstrap_unit": "asset (both motions together), jointly with training seed",
    }, trajectories


def plots(output, result, trajectories):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = {r["model"]: r for r in result["summary"]}
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(6)
    means = [summary[m]["mean_cm"] for m in MODELS]
    sd = [summary[m]["seed_sd_cm"] or 0 for m in MODELS]
    ax.bar(
        x,
        means,
        yerr=sd,
        capsize=4,
        color=["#a4b7c8", "#7498b7", "#287f73", "#90b7a7", "#c4b7a2", "#b29d86"],
    )
    for i, model in enumerate(MODELS[:4]):
        ax.scatter(
            np.full(3, i),
            summary[model]["seed_means_cm"],
            color="black",
            s=12,
            zorder=3,
        )
    ax.set(
        xticks=x,
        xticklabels=["MLP", "GRU", "GNN", "No messages", "Hold", "Const. velocity"],
        ylabel="Position error (cm)",
        title="GRAIL asset-held-out test: mean and seed SD",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"comparison.{suffix}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, entity in zip(axes, ("robot_root", "object")):
        for model, label in zip(MODELS, LABELS):
            selected = sorted(
                (
                    r
                    for r in result["groups"]
                    if r["entity"] == entity
                    and r["model"] == model
                    and r["segment"] == "all"
                ),
                key=lambda r: r["horizon_steps"],
            )
            ax.plot(
                [r["horizon_steps"] / 25 for r in selected],
                [r["mean_cm"] for r in selected],
                "o-",
                label=label,
            )
        ax.set(
            title=entity.replace("_", " ").title(),
            xlabel="Forecast horizon (s)",
            ylabel="Position error (cm)",
        )
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"horizons.{suffix}", dpi=160)
    plt.close(fig)
    names = sorted({r["episode"] for r in trajectories["temporal_gnn"]})
    for number, name in enumerate(names, 1):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for row, entity in enumerate(("robot_root", "object")):
            reference = [
                r
                for r in trajectories["temporal_gnn"]
                if r["episode"] == name and r["entity"] == entity
            ]
            time = [int(r["target_frame"]) / 25 for r in reference]
            for column, axis in enumerate("xyz"):
                ax = axes[row, column]
                ax.plot(
                    time,
                    [float(r[f"true_{axis}"]) for r in reference],
                    color="black",
                    label="Reference",
                    linewidth=1.8,
                )
                for model, label, color in (
                    ("temporal_gnn", "GNN seed 0", "#287f73"),
                    ("gru", "GRU seed 0", "#527ca6"),
                    ("constant_velocity", "Constant velocity", "#bf8b47"),
                ):
                    values = [
                        r
                        for r in trajectories[model]
                        if r["episode"] == name and r["entity"] == entity
                    ]
                    if [r["target_frame"] for r in values] != [
                        r["target_frame"] for r in reference
                    ]:
                        raise ValueError("Trajectory plot alignment differs")
                    ax.plot(
                        time,
                        [float(r[f"pred_{axis}"]) for r in values],
                        label=label,
                        color=color,
                        alpha=0.85,
                    )
                ax.set(
                    xlabel="Target time (s)",
                    ylabel=f"{axis} (m)",
                    title=entity.replace("_", " "),
                )
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"{name}\nDirect 0.48 s forecasts from observed histories; not a free rollout",
            fontsize=11,
        )
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"trajectory_{number}.{suffix}", dpi=150)
        plt.close(fig)
    write_json(
        output / "trajectory_examples.json",
        {
            "episodes": names,
            "selection": "All two motions from each of the first two sorted test assets",
            "horizon_steps": 12,
            "plotted_seed": 0,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, trajectories = analyze(args.runs, json.loads(args.manifest.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "analysis.json", {**result, "source": source_version()})
    for name in ("summary", "groups", "paired_contrasts", "costs"):
        write_csv(args.output / f"{name}.csv", result[name])
    plots(args.output, result, trajectories)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
