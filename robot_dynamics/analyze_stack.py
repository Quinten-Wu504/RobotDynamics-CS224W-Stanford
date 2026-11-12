import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from robot_dynamics.analyze_formal import paired_interval
from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.formal import write_json


def analyze(runs):
    complete = json.loads((runs / "evaluation/complete.json").read_text())
    macros, groups, values = defaultdict(list), defaultdict(list), {}
    for filename in complete["results"]:
        path = runs / "evaluation" / filename
        result = json.loads(path.read_text())
        rows_path = path.with_name(path.stem + "_episodes.csv")
        raw = (
            rows_path.read_bytes()
            if rows_path.exists()
            else gzip.decompress(rows_path.with_suffix(".csv.gz").read_bytes())
        )
        if hashlib.sha256(raw).hexdigest() != result["episode_rows_sha256"]:
            raise ValueError("Episode CSV changed")
        rows = list(csv.DictReader(io.StringIO(raw.decode())))
        episode = defaultdict(list)
        for row in rows:
            if row["segment"] == "all":
                episode[row["episode"]].append(float(row["mean_error_cm"]))
        if set(episode) != set(result["episode_ids"]) or any(
            len(v) != 9 for v in episode.values()
        ):
            raise ValueError("Incomplete episode/entity/horizon results")
        ordered = sorted(episode)
        v = np.array([np.mean(episode[k]) for k in ordered])
        if not np.isclose(v.mean(), result["macro"]["mean_cm"], rtol=0, atol=1e-12):
            raise ValueError("Published macro does not match episode errors")
        key = (result["split"], result["model"])
        seed = result["seed"]
        values[(*key, seed)] = (ordered, v)
        macros[key].append((seed, float(v.mean())))
        for group in result["summary"]:
            groups[
                (*key, group["entity"], group["horizon_steps"], group["segment"])
            ].append((seed, group))
    summary = []
    for (split, model), entries in sorted(macros.items()):
        entries.sort()
        a = np.array([v for _, v in entries])
        summary.append(
            {
                "split": split,
                "model": model,
                "seeds": [s for s, _ in entries],
                "seed_means_cm": a.tolist(),
                "mean_cm": float(a.mean()),
                "seed_sd_cm": float(a.std(ddof=1)) if len(a) > 1 else None,
            }
        )
    group_rows = []
    for (split, model, entity, horizon, segment), entries in sorted(groups.items()):
        entries.sort()
        counts = {(g["episodes"], g["windows"]) for _, g in entries}
        if len(counts) != 1:
            raise ValueError("Group populations differ across seeds")
        n, windows = counts.pop()
        a = np.array([g["mean_cm"] for _, g in entries])
        group_rows.append(
            {
                "split": split,
                "model": model,
                "entity": entity,
                "horizon_steps": horizon,
                "segment": segment,
                "episodes_per_seed": n,
                "windows_per_seed": windows,
                "mean_cm": float(a.mean()),
                "seed_sd_cm": float(a.std(ddof=1)) if len(a) > 1 else None,
                "seed_means_cm": a.tolist(),
            }
        )
    contrasts = []
    for split in sorted({k[0] for k in macros}):
        for other in ("history_mlp", "gru", "temporal_no_messages"):
            differences = []
            for seed in (0, 1, 2):
                a = values[(split, "temporal_gnn", seed)]
                b = values[(split, other, seed)]
                if a[0] != b[0]:
                    raise ValueError("Unpaired episode IDs")
                differences.append(a[1] - b[1])
            contrasts.append(
                {
                    "split": split,
                    "comparison": f"temporal_gnn minus {other}",
                    **paired_interval(differences),
                }
            )
    costs = []
    plan = json.loads((runs / "matrix.json").read_text())
    for run in plan["runs"]:
        m = json.loads((runs / run["run_id"] / "metrics.json").read_text())
        costs.append(
            {
                "run_id": run["run_id"],
                "model": m["model"],
                "seed": m["config"]["seed"],
                "best_epoch": m["best_epoch"],
                "parameters": m["parameter_count"],
                "optimizer_steps": m["optimizer_steps_total"],
                "update_seconds": sum(entry["update_seconds"] for entry in m["logs"]),
                "resources": m["resources"],
            }
        )
    return {
        "summary": summary,
        "groups": group_rows,
        "paired_contrasts": contrasts,
        "costs": costs,
        "complete_sha256": file_sha256(runs / "evaluation/complete.json"),
        **source_version(),
    }


def plots(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["history_mlp", "gru", "temporal_gnn", "temporal_no_messages"]
    labels = ["MLP", "GRU", "Temporal GNN", "Messages closed"]
    colors = ["#999999", "#da8e38", "#2274a5", "#65a388"]
    splits = ["d0_test", "d1_final_test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, split in zip(axes, splits):
        for i, (model, color) in enumerate(zip(models, colors)):
            row = next(
                r
                for r in result["summary"]
                if (r["split"], r["model"]) == (split, model)
            )
            ax.bar(
                i,
                row["mean_cm"],
                yerr=row["seed_sd_cm"],
                color=color,
                alpha=0.75,
                capsize=3,
            )
            ax.scatter(
                i + np.linspace(-0.12, 0.12, 3),
                row["seed_means_cm"],
                s=16,
                color="black",
                zorder=3,
            )
        ax.set_xticks(range(4), labels, rotation=15)
        ax.set_title("Stack D0 test" if split == "d0_test" else "Stack D1")
        ax.set_ylabel("Equal-episode mean position error (cm)")
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for i, split in enumerate(splits):
        for j, entity in enumerate(("end_effector", "cubeA", "cubeB")):
            ax = axes[i, j]
            for model, label, color in zip(models, labels, colors):
                rows = sorted(
                    [
                        r
                        for r in result["groups"]
                        if (r["split"], r["model"], r["entity"], r["segment"])
                        == (split, model, entity, "all")
                    ],
                    key=lambda r: r["horizon_steps"],
                )
                ax.plot(
                    [0.05, 0.25, 0.5],
                    [r["mean_cm"] for r in rows],
                    marker="o",
                    label=label,
                    color=color,
                )
            ax.set_title(f"{'D0 test' if i == 0 else 'D1'}: {entity}")
            ax.set_xlabel("Horizon (s)")
            ax.set_ylabel("Error (cm)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.savefig(output / "horizons.png", dpi=180)
    fig.savefig(output / "horizons.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = analyze(args.runs)
    write_json(args.output / "analysis.json", result)
    for key in ("summary", "groups", "paired_contrasts", "costs"):
        rows = result[key]
        with (args.output / f"{key}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plots(result, args.output)


if __name__ == "__main__":
    main()
