import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    output = args.output or args.analysis.parent
    output.mkdir(parents=True, exist_ok=True)
    splits = sorted({r["split"] for r in data["summary"]})
    variants = sorted({(r["model"], r["method"]) for r in data["summary"]})
    colors = {v: plt.get_cmap("tab10")(i) for i, v in enumerate(variants)}
    labels = {
        v: f"{v[0].replace('canonical_temporal_gnn', 'Canonical TGN').replace('temporal_gnn', 'TGN').replace('history_mlp', 'History MLP').replace('gru', 'GRU')} / {v[1]}"
        for v in variants
    }

    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, len(splits), figsize=(12, 5), squeeze=False)
    for ax, split in zip(axes[0], splits):
        for variant in variants:
            rows = sorted(
                [
                    r
                    for r in data["summary"]
                    if r["split"] == split and (r["model"], r["method"]) == variant
                ],
                key=lambda r: r["train_episodes"],
            )
            ax.errorbar(
                [r["train_episodes"] for r in rows],
                [r["mean_cm"] for r in rows],
                yerr=[r["seed_sd_cm"] or 0 for r in rows],
                marker="o",
                capsize=3,
                color=colors[variant],
                label=labels[variant],
            )
        ax.set(
            title=split.replace("_", " "),
            xlabel="Training episodes",
            ylabel="Six-error mean (cm)",
            xscale="log",
        )
        ticks = sorted({r["train_episodes"] for r in data["summary"]})
        ax.set_xticks(ticks, [str(n) for n in ticks])
        ax.grid(alpha=0.2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Data efficiency · error bars show training-seed SD")
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    save(fig, "data_efficiency")
    maximum = max(r["train_episodes"] for r in data["summary"])
    fig, axes = plt.subplots(2, len(splits), figsize=(12, 8), squeeze=False)
    for col, split in enumerate(splits):
        for row, entity in enumerate(("end_effector", "square_nut")):
            ax = axes[row, col]
            for variant in variants:
                rows = sorted(
                    [
                        r
                        for r in data["groups"]
                        if r["split"] == split
                        and r["train_episodes"] == maximum
                        and r["entity"] == entity
                        and r["segment"] == "all"
                        and (r["model"], r["method"]) == variant
                    ],
                    key=lambda r: r["horizon_steps"],
                )
                if rows:
                    ax.plot(
                        [r["horizon_steps"] / 20 for r in rows],
                        [r["mean_cm"] for r in rows],
                        marker="o",
                        color=colors[variant],
                        label=labels[variant],
                    )
            ax.set(
                title=f"{split}: {entity.replace('_', ' ')}",
                xlabel="Horizon (s)",
                ylabel="Episode mean error (cm)",
            )
            ax.grid(alpha=0.2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(
        f"Physical prediction errors · {maximum} training episodes · means across seeds"
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    save(fig, "horizons")
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant in variants:
        rows = [r for r in data["costs"] if (r["model"], r["method"]) == variant]
        counts = sorted({r["train_episodes"] for r in rows})
        costs = [
            np.mean(
                [
                    (
                        r["resources"]["initial_phase_update_seconds"]
                        + r["resources"]["finetune_phase_update_seconds"]
                    )
                    / 60
                    for r in rows
                    if r["train_episodes"] == n
                ]
            )
            for n in counts
        ]
        ax.plot(counts, costs, marker="o", color=colors[variant], label=labels[variant])
    ax.set(
        xlabel="Training episodes",
        ylabel="Mean measured update time (minutes)",
        title="Training cost · excludes validation and input preparation",
        xscale="log",
    )
    ticks = sorted({r["train_episodes"] for r in data["summary"]})
    ax.set_xticks(ticks, [str(n) for n in ticks])
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "training_cost")


if __name__ == "__main__":
    main()
