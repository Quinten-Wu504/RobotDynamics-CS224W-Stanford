import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robot_dynamics.windows import anchors, predict_positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = json.loads(args.metrics.read_text())
    colors = {"hold": "#286fa0", "constant_velocity": "#c56827"}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for i, entity in enumerate(("end_effector", "square_nut")):
        for j, segment in enumerate(("all", "low_motion", "moving")):
            ax = axes[i, j]
            for model, color in colors.items():
                rows = sorted(
                    [
                        r
                        for r in result["summary"]
                        if r["entity"] == entity
                        and r["segment"] == segment
                        and r["model"] == model
                    ],
                    key=lambda r: r["horizon_steps"],
                )
                x = [r["horizon_steps"] / 20 for r in rows]
                y = np.array([r["mean_cm"] for r in rows])
                ci = np.array([r["ci95_cm"] for r in rows]).T
                ax.errorbar(
                    x,
                    y,
                    yerr=np.abs(ci - y),
                    marker="o",
                    color=color,
                    capsize=3,
                    label=model.replace("_", " "),
                )
            ax.set(
                title=f"{entity.replace('_', ' ')} / {segment.replace('_', ' ')}",
                xlabel="Horizon (s)",
                ylabel="Position error (cm)",
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "D0 validation: deterministic baselines; episode-bootstrap 95% intervals"
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "baseline_errors.png", dpi=180)
    fig.savefig(args.output_dir / "baseline_errors.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    with h5py.File(args.cache, "r") as cache:
        for j, name in enumerate(result["episode_ids"][:3]):
            for i, key in enumerate(("eef_position", "nut_position")):
                p = cache[name][key][:]
                indices = anchors(len(p), result["config"]["history_steps"], 10)
                ax = axes[i, j]
                ax.plot(
                    p[indices + 10, 0],
                    p[indices + 10, 1],
                    color="black",
                    label="Recorded future",
                )
                for model, color in colors.items():
                    pred = predict_positions(p, indices, 10, model)
                    ax.plot(
                        pred[:, 0],
                        pred[:, 1],
                        color=color,
                        alpha=0.8,
                        label=model.replace("_", " "),
                    )
                ax.set(
                    title=f"{name} / {key.removesuffix('_position')}",
                    xlabel="World x (m)",
                    ylabel="World y (m)",
                )
                ax.set_aspect("equal", adjustable="datalim")
                ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "0.5 s predictions from recorded histories; first three validation IDs"
    )
    fig.text(
        0.5,
        0.015,
        f"Source: {result['source_commit'][:12]}; direct predictions, not free rollout",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(args.output_dir / "baseline_trajectories.png", dpi=180)
    fig.savefig(args.output_dir / "baseline_trajectories.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
