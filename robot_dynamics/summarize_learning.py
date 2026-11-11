import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_dynamics.data_audit import source_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, default=Path("reports/stage3/evaluation/metrics.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(args.baseline.read_text())
    rows = [
        {**r, "model": r["model"]}
        for r in baseline["summary"]
        if r["segment"] == "all" and r["model"] != "constant_velocity_minus_hold"
    ]
    pilots = {}
    fits = {}
    reload_checks = {}
    for model in ("current_mlp", "history_mlp", "gru"):
        d = json.loads((args.runs / f"pilot_{model}/metrics.json").read_text())
        reloaded = json.loads((args.runs / f"pilot_{model}_reload.json").read_text())
        check = (
            d["prediction_sha256"] == reloaded["prediction_sha256"]
            and d["summary"] == reloaded["summary"]
        )
        if not check or d["evaluation_ids"] != baseline["episode_ids"]:
            raise ValueError("Reload or evaluation split mismatch")
        for r in d["summary"]:
            reference = next(
                x
                for x in rows
                if x["model"] == "hold"
                and x["entity"] == r["entity"]
                and x["horizon_steps"] == r["horizon_steps"]
            )
            if r["windows"] != reference["windows"]:
                raise ValueError("Baseline window count mismatch")
            rows.append({**r, "model": model})
        pilots[model] = d
        fits[model] = json.loads((args.runs / f"fit_{model}/metrics.json").read_text())
        reload_checks[model] = check
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model",
                "entity",
                "horizon_steps",
                "windows",
                "episodes",
                "mean_cm",
                "ci95_low_cm",
                "ci95_high_cm",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    k: r[k]
                    for k in (
                        "model",
                        "entity",
                        "horizon_steps",
                        "windows",
                        "episodes",
                        "mean_cm",
                    )
                }
                | {"ci95_low_cm": r["ci95_cm"][0], "ci95_high_cm": r["ci95_cm"][1]}
            )
    colors = {
        "hold": "#777777",
        "constant_velocity": "#b28b32",
        "current_mlp": "#2f78a7",
        "history_mlp": "#bb5d47",
        "gru": "#3a8b6d",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, entity in zip(axes, ("end_effector", "square_nut")):
        for model, color in colors.items():
            selected = sorted(
                [r for r in rows if r["model"] == model and r["entity"] == entity],
                key=lambda r: r["horizon_steps"],
            )
            ax.plot(
                [r["horizon_steps"] / 20 for r in selected],
                [r["mean_cm"] for r in selected],
                marker="o",
                color=color,
                label=model.replace("_", " "),
            )
        ax.set(
            title=entity.replace("_", " "),
            xlabel="Horizon (s)",
            ylabel="Episode mean error (cm)",
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("D0 validation: 80-episode learning pilots, seed 0")
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=180)
    fig.savefig(args.output / "comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model, d in pilots.items():
        logs = d["logs"]
        epochs = [r["epoch"] for r in logs]
        axes[0].plot(
            epochs, [r["train_mse"] for r in logs], label=model, color=colors[model]
        )
        axes[1].plot(
            epochs, [r["selection_cm"] for r in logs], label=model, color=colors[model]
        )
        axes[1].scatter(
            [d["best_epoch"]],
            [d["best_score_cm"]],
            color=colors[model],
            marker="x",
            s=45,
        )
    axes[0].set(xlabel="Epoch", ylabel="Normalized training MSE", yscale="log")
    axes[1].set(xlabel="Epoch", ylabel="Validation selection score (cm)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "learning_curves.png", dpi=180)
    fig.savefig(args.output / "learning_curves.pdf")
    plt.close(fig)
    summary = {
        "pilots": {
            k: {
                x: v[x]
                for x in (
                    "parameter_count",
                    "best_epoch",
                    "best_score_cm",
                    "train_windows",
                    "evaluation_windows",
                    "best_train_mse",
                    "train_summary",
                    "resources",
                    "source_commit",
                    "working_tree_dirty",
                    "best_checkpoint_sha256",
                )
            }
            for k, v in pilots.items()
        },
        "fits": {
            k: {
                x: v[x]
                for x in (
                    "best_score_cm",
                    "best_train_mse",
                    "initial_train_mse_this_invocation",
                    "resources",
                    "source_commit",
                )
            }
            for k, v in fits.items()
        },
        "standalone_reload_exact": reload_checks,
        **source_version(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
