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
    methods = ["supervised30", "supervised", "state", "jepa"]
    labels = ["Supervised 30", "Supervised 40", "State 10 + FT 30", "JEPA 10 + FT 30"]
    colors = ["#8c9dad", "#496c89", "#d9a441", "#348c78"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, kind, title in zip(
        axes,
        ["temporal_gnn", "canonical_temporal_gnn"],
        ["Temporal GNN", "Canonical Temporal GNN"],
    ):
        rows = [data["runs"][f"{kind}_{method}"]["macro"] for method in methods]
        means = np.array([row["mean_cm"] for row in rows])
        intervals = np.array([row["ci95_cm"] for row in rows]).T
        ax.bar(
            np.arange(4),
            means,
            color=colors,
            yerr=np.array([means - intervals[0], intervals[1] - means]),
            capsize=4,
        )
        ax.set(
            xticks=np.arange(4),
            xticklabels=labels,
            title=title,
            ylabel="Six-error mean (cm; lower is better)",
        )
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
        for i, value in enumerate(means):
            ax.text(
                i, 0.012, f"{value:.3f}", ha="center", color="white", fontweight="bold"
            )
    fig.suptitle(
        "D0 validation · 80 training episodes · seed 0\n95% episode-bootstrap intervals; not seed uncertainty",
        fontsize=12,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"comparison.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey="row")
    for col, kind in enumerate(["temporal_gnn", "canonical_temporal_gnn"]):
        for row, entity in enumerate(["end_effector", "square_nut"]):
            ax = axes[row, col]
            for method, label, color in zip(methods, labels, colors):
                values = sorted(
                    [
                        r
                        for r in data["summary"]
                        if r["run"] == f"{kind}_{method}" and r["entity"] == entity
                    ],
                    key=lambda r: r["horizon_steps"],
                )
                ax.plot(
                    [r["horizon_steps"] / 20 for r in values],
                    [r["mean_cm"] for r in values],
                    marker="o",
                    color=color,
                    label=label,
                )
            ax.set(
                title=("Canonical " if col else "")
                + "Temporal GNN: "
                + entity.replace("_", " "),
                xlabel="Horizon (s)",
                ylabel="Episode mean error (cm)",
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("D0 validation · seed 0 · identical training episodes", fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"horizons.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
