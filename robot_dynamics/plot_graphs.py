import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(result, output):
    overall, groups, runs = result["overall"], result["groups"], result["runs"]
    models = tuple(runs)
    graph_models = [k for k in models if k not in ("current_mlp", "history_mlp", "gru")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, entity in zip(axes, ("end_effector", "square_nut")):
        for kind in models:
            rows = [r for r in overall if r["model"] == kind and r["entity"] == entity]
            ax.plot(
                [r["horizon_steps"] / 20 for r in rows],
                [r["mean_cm"] for r in rows],
                marker="o",
                label=kind,
            )
        ax.set(
            title=entity.replace("_", " "),
            xlabel="Horizon (s)",
            ylabel="Episode mean error (cm)",
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("D0 validation, 80 training episodes, seed 0")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    segments = ("low_motion", "moving", "near_object", "far_object")
    for ax, entity in zip(axes, ("end_effector", "square_nut")):
        for offset, kind in enumerate(models):
            rows = [
                next(
                    r
                    for r in groups
                    if r["model"] == kind
                    and r["entity"] == entity
                    and r["horizon_steps"] == 10
                    and r["segment"] == s
                )
                for s in segments
            ]
            ax.bar(
                np.arange(4) + (offset - 3) * 0.11,
                [r["mean_cm"] for r in rows],
                width=0.11,
                label=kind,
            )
        ax.set(
            xticks=np.arange(4),
            xticklabels=[s.replace("_", " ") for s in segments],
            title=entity.replace("_", " "),
            ylabel="Episode mean error (cm)",
        )
        ax.grid(axis="y", alpha=0.2)
    near = next(
        r
        for r in groups
        if r["model"] == models[0]
        and r["entity"] == "end_effector"
        and r["horizon_steps"] == 10
        and r["segment"] == "near_object"
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        fontsize=8,
    )
    fig.suptitle(
        f"0.5 s prediction: near-object group has {near['episodes']} episodes / {near['windows']} windows"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output / "groups.png", dpi=180)
    fig.savefig(output / "groups.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind in graph_models:
        logs = runs[kind]["logs"]
        axes[0].plot(
            [r["epoch"] for r in logs], [r["train_mse"] for r in logs], label=kind
        )
        axes[1].plot(
            [r["epoch"] for r in logs], [r["selection_cm"] for r in logs], label=kind
        )
    axes[0].set(xlabel="Epoch", ylabel="Normalized training MSE", yscale="log")
    axes[1].set(xlabel="Epoch", ylabel="Validation selection score (cm)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
    fig.savefig(output / "learning_curves.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.analysis.parent
    output.mkdir(parents=True, exist_ok=True)
    plot_results(json.loads(args.analysis.read_text()), output)


if __name__ == "__main__":
    main()
