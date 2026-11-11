import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    output = args.output or args.analysis.parent
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, entity in zip(axes, ("end_effector", "square_nut")):
        for kind in data["runs"]:
            rows = [
                r
                for r in data["summary"]
                if r["model"] == kind and r["entity"] == entity
            ]
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
    axes[0].legend(fontsize=8)
    fig.suptitle("Original D0 validation: seed 0, 80 training episodes")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"comparison.{suffix}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    checks = data["coordinate_checks"]
    ax.bar(
        range(len(checks)), [max(r["max_abs_m"] * 100, 1e-12) for r in checks.values()]
    )
    ax.axhline(
        0.002, color="red", linestyle="--", label="Canonical-model acceptance bound"
    )
    ax.set(
        xticks=range(len(checks)),
        xticklabels=[k.replace("canonical_", "canonical\n") for k in checks],
        yscale="log",
        ylabel="Max absolute future-position discrepancy (cm)",
        title="Coordinate consistency: 96 windows × 16 transforms",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"coordinate_checks.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
