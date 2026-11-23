"""Regenerate the course-report figures from the archived experiment records."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.analyze_stack import plots as stack_plots
from robot_dynamics.analyze_grail import analyze as grail_analyze, plots as grail_plots


SOURCES = {
    "square": Path("reports/stage8/formal/analysis/analysis.json"),
    "stack": Path("reports/stage9/formal/analysis/analysis.json"),
    "grail": Path("reports/stage9/grail/formal/analysis/analysis.json"),
}
MODELS = ("history_mlp", "gru", "temporal_gnn", "temporal_no_messages")
LABELS = ("History MLP", "GRU", "Temporal GNN", "Messages disabled")


def export(output):
    output.mkdir(parents=True, exist_ok=False)
    data = {name: json.loads(path.read_text()) for name, path in SOURCES.items()}
    rows = []
    for task, result in data.items():
        for row in result["summary"]:
            rows.append(
                {
                    "task": task,
                    "split": row.get("split", "asset_test"),
                    "train_episodes": row.get(
                        "train_episodes", 800 if task == "stack" else 168
                    )
                    if row["seed_sd_cm"] is not None
                    else 0,
                    "model": row["model"],
                    "method": row.get(
                        "method",
                        "supervised" if row["seed_sd_cm"] is not None else "baseline",
                    ),
                    "mean_cm": row["mean_cm"],
                    "seed_sd_cm": row["seed_sd_cm"],
                    "seed_values_cm": json.dumps(row["seed_means_cm"]),
                }
            )
    for baseline in data["square"]["baselines"]:
        rows.append(
            {
                "task": "square",
                "split": baseline["split"],
                "train_episodes": 0,
                "model": baseline["run_id"],
                "method": "baseline",
                "mean_cm": baseline["macro"]["mean_cm"],
                "seed_sd_cm": None,
                "seed_values_cm": json.dumps([baseline["macro"]["mean_cm"]]),
            }
        )
    with (output / "all_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "all_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Final result tables",
        "",
        "Errors in cm; lower is better. Each task uses its own metric and split. Seed SD is not a confidence interval.",
        "",
        "| Task | Split | Training motions | Model | Method | Mean | Seed SD |",
        "| --- | --- | ---: | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        sd = f"{row['seed_sd_cm']:.4f}" if row["seed_sd_cm"] is not None else "—"
        lines.append(
            f"| {row['task']} | {row['split']} | {row['train_episodes']} | {row['model']} | {row['method']} | {row['mean_cm']:.4f} | {sd} |"
        )
    (output / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    for ax, task in zip(axes, data):
        splits = ("asset_test",) if task == "grail" else ("d0_test", "d1_final_test")
        model_names = MODELS[:3] if task == "square" else MODELS
        width = 0.8 / len(model_names)
        for index, model in enumerate(model_names):
            selected = [
                next(
                    r
                    for r in rows
                    if r["task"] == task
                    and r["split"] == split
                    and r["model"] == model
                    and r["method"] == "supervised"
                    and r["train_episodes"] == (168 if task == "grail" else 800)
                )
                for split in splits
            ]
            ax.bar(
                np.arange(len(splits)) + (index - (len(model_names) - 1) / 2) * width,
                [r["mean_cm"] for r in selected],
                width,
                yerr=[r["seed_sd_cm"] for r in selected],
                capsize=3,
                label=LABELS[index],
            )
        ax.set(
            xticks=range(len(splits)),
            xticklabels=[
                "Asset test"
                if s == "asset_test"
                else "D0 test"
                if s == "d0_test"
                else "D1"
                for s in splits
            ],
            ylabel="Mean error (cm)",
            title=f"{task.title()} · {'9' if task == 'stack' else '6'} target/horizon cells",
        )
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Supervised comparisons · separate task scales · error bars: seed SD")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"overview.{suffix}", dpi=160)
    plt.close(fig)

    costs = []
    for task, result in data.items():
        for record in result["costs"]:
            if task == "square":
                seconds = (
                    record["resources"]["initial_phase_update_seconds"]
                    + record["resources"]["finetune_phase_update_seconds"]
                )
                count = record["parameter_count"]
            elif task == "stack":
                seconds, count = record["update_seconds"], record["parameters"]
            else:
                seconds, count = (
                    60 * record["update_minutes"],
                    record["parameter_count"],
                )
            costs.append(
                {
                    "task": task,
                    "run_id": record["run_id"],
                    "model": record["model"],
                    "method": record.get("method", "supervised"),
                    "seed": record["seed"],
                    "train_episodes": record.get(
                        "train_episodes", 800 if task == "stack" else 168
                    ),
                    "parameters": count,
                    "update_minutes": seconds / 60,
                }
            )
    with (output / "all_costs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(costs[0]))
        writer.writeheader()
        writer.writerows(costs)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for ax, task in zip(axes, data):
        model_names = MODELS[:3] if task == "square" else MODELS
        values = [
            np.mean(
                [
                    r["update_minutes"]
                    for r in costs
                    if r["task"] == task
                    and r["model"] == model
                    and r["method"] == "supervised"
                    and r["train_episodes"] == (168 if task == "grail" else 800)
                ]
            )
            for model in model_names
        ]
        ax.bar(range(len(values)), values, color=[f"C{i}" for i in range(len(values))])
        ax.set(
            xticks=range(len(values)),
            xticklabels=["MLP", "GRU", "GNN", "No msg"][: len(values)],
            ylabel="Mean update minutes/run",
            title=task.upper() if task == "grail" else task.title(),
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Measured training cost · excludes loading, validation and reload")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"cost_comparison.{suffix}", dpi=160)
    plt.close(fig)

    square_output = output / "square"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "robot_dynamics.plot_formal",
            str(SOURCES["square"]),
            "--output",
            str(square_output),
        ],
        check=True,
    )
    stack_output = output / "stack"
    stack_output.mkdir()
    stack_plots(data["stack"], stack_output)
    grail_output = output / "grail"
    grail_output.mkdir()
    grail, trajectories = grail_analyze(
        Path("reports/stage9/grail/formal"),
        json.loads(Path("reports/stage9/grail/manifest.json").read_text()),
    )
    for key in ("summary", "groups", "paired_contrasts", "costs", "training_curves"):
        if grail[key] != data["grail"][key]:
            raise ValueError(f"GRAIL archive reanalysis differs: {key}")
    grail_plots(grail_output, grail, trajectories)

    groups = data["square"]["groups"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for i, split in enumerate(("d0_test", "d1_final_test")):
        for j, entity in enumerate(("end_effector", "square_nut")):
            ax = axes[i, j]
            for model, color in (("gru", "tab:orange"), ("temporal_gnn", "tab:green")):
                for segment, style in (("near_object", "-"), ("far_object", "--")):
                    selected = sorted(
                        [
                            r
                            for r in groups
                            if r["split"] == split
                            and r["entity"] == entity
                            and r["model"] == model
                            and r["method"] == "supervised"
                            and r["train_episodes"] == 800
                            and r["segment"] == segment
                        ],
                        key=lambda r: r["horizon_steps"],
                    )
                    counts = "/".join(str(r["episodes_per_seed"]) for r in selected)
                    ax.plot(
                        [r["horizon_steps"] / 20 for r in selected],
                        [r["mean_cm"] for r in selected],
                        style,
                        color=color,
                        marker="o",
                        label=f"{model}: {segment} (n={counts})",
                    )
            ax.set(
                title=f"{split}: {entity}", xlabel="Horizon (s)", ylabel="Error (cm)"
            )
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
    fig.suptitle("Square proximity groups · distance ≤ 5 cm is a proxy, not contact")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"proximity.{suffix}", dpi=160)
    plt.close(fig)
    provenance = {
        "inputs": {
            name: {"path": str(path).replace("\\", "/"), "sha256": file_sha256(path)}
            for name, path in SOURCES.items()
        },
        "summary_rows": len(rows),
        "grail_reanalysis_exact": True,
        "new_training": False,
        "files": {
            str(p.relative_to(output)).replace("\\", "/"): file_sha256(p)
            for p in output.rglob("*")
            if p.is_file()
        },
        **source_version(),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported {len(rows)} result rows and figures to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    export(parser.parse_args().output)


if __name__ == "__main__":
    main()
