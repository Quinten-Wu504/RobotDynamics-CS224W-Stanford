import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.formal import write_json


def paired_interval(values):
    values = np.asarray(values)
    rng = np.random.default_rng(224)
    seeds = rng.integers(0, len(values), (2000, len(values)))
    episodes = rng.integers(0, values.shape[1], (2000, values.shape[1]))
    sampled = values[seeds[:, :, None], episodes[:, None, :]].mean(axis=(1, 2))
    return {
        "mean_difference_cm": float(values.mean()),
        "seed_differences_cm": values.mean(axis=1).tolist(),
        "ci95_seed_episode_cm": np.quantile(sampled, [0.025, 0.975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.runs / "matrix.json").read_text())
    completed = json.loads((args.runs / "evaluation/complete.json").read_text())
    records, macros, detailed, costs, diagnostics = (
        [],
        defaultdict(list),
        defaultdict(list),
        [],
        [],
    )
    episode_values = {}
    for run in plan["runs"]:
        name, cfg = run["run_id"], run["config"]
        metrics = json.loads((args.runs / name / "metrics.json").read_text())
        costs.append(
            {
                "run_id": name,
                "model": run["model"],
                "method": run["method"],
                "train_episodes": cfg["train_episodes"],
                "seed": cfg["seed"],
                "parameter_count": metrics["parameter_count"],
                "auxiliary_parameter_count": metrics["auxiliary_parameter_count"],
                "pretrain_steps": metrics["pretrain_steps"],
                "downstream_steps": metrics["downstream_steps"],
                "best_epoch": metrics["best_epoch"],
                "train_windows": metrics["train_windows"],
                "resources": metrics["resources"],
            }
        )
        diagnostic = json.loads(
            (args.runs / "evaluation" / f"{name}_diagnostics.json").read_text()
        )
        diagnostics.append(diagnostic)
        for split in completed["splits"]:
            path = args.runs / "evaluation" / f"{name}_{split}.json"
            result = json.loads(path.read_text())
            rows_path = path.with_name(f"{name}_{split}_episodes.csv")
            if file_sha256(rows_path) != result["episode_rows_sha256"]:
                raise ValueError("Episode artifact changed")
            with rows_path.open() as stream:
                rows = list(csv.DictReader(stream))
            values = []
            for episode in result["episode_ids"]:
                selected = [
                    float(r["mean_error_cm"])
                    for r in rows
                    if r["episode"] == episode and r["segment"] == "all"
                ]
                if len(selected) != 6:
                    raise ValueError("Incomplete primary episode results")
                values.append(np.mean(selected))
            key = (split, cfg["train_episodes"], run["model"], run["method"])
            macros[key].append((cfg["seed"], result["macro"]["mean_cm"]))
            episode_values[(*key, cfg["seed"])] = np.asarray(values)
            records.append(
                {
                    "run_id": name,
                    "split": split,
                    "macro": result["macro"],
                    "evaluation_sha256": file_sha256(path),
                }
            )
            for row in result["summary"]:
                detailed[
                    (*key, row["entity"], row["horizon_steps"], row["segment"])
                ].append((cfg["seed"], row))
    summary = []
    for (split, count, model, method), values in sorted(macros.items()):
        values.sort()
        errors = np.array([v for _, v in values])
        summary.append(
            {
                "split": split,
                "train_episodes": count,
                "model": model,
                "method": method,
                "seeds": [s for s, _ in values],
                "seed_means_cm": errors.tolist(),
                "mean_cm": float(errors.mean()),
                "seed_sd_cm": float(errors.std(ddof=1)) if len(errors) > 1 else None,
                "min_cm": float(errors.min()),
                "max_cm": float(errors.max()),
            }
        )
    groups = []
    for (split, count, model, method, entity, horizon, segment), values in sorted(
        detailed.items()
    ):
        counts = {(r["episodes"], r["windows"]) for _, r in values}
        if len(counts) != 1:
            raise ValueError("Group counts vary across seeds")
        episodes, windows = counts.pop()
        errors = np.array([r["mean_cm"] for _, r in sorted(values)])
        groups.append(
            {
                "split": split,
                "train_episodes": count,
                "model": model,
                "method": method,
                "entity": entity,
                "horizon_steps": horizon,
                "segment": segment,
                "episodes_per_seed": episodes,
                "windows_per_seed": windows,
                "mean_cm": float(errors.mean()),
                "seed_sd_cm": float(errors.std(ddof=1)) if len(errors) > 1 else None,
                "seed_means_cm": errors.tolist(),
            }
        )
    pairs = [
        (("temporal_gnn", "supervised"), ("history_mlp", "supervised")),
        (("temporal_gnn", "supervised"), ("gru", "supervised")),
        (("canonical_temporal_gnn", "supervised"), ("temporal_gnn", "supervised")),
        (("temporal_gnn", "jepa"), ("temporal_gnn", "supervised")),
        (("canonical_temporal_gnn", "jepa"), ("canonical_temporal_gnn", "supervised")),
        (("canonical_temporal_gnn", "jepa"), ("temporal_gnn", "jepa")),
        (("temporal_gnn", "jepa"), ("temporal_gnn", "state")),
        (("canonical_temporal_gnn", "jepa"), ("canonical_temporal_gnn", "state")),
    ]
    contrasts = []
    for split in completed["splits"]:
        for count in sorted({r["config"]["train_episodes"] for r in plan["runs"]}):
            for left, right in pairs:
                left_key, right_key = (split, count, *left), (split, count, *right)
                if left_key not in macros or right_key not in macros:
                    continue
                seeds = sorted(s for s, _ in macros[left_key])
                delta = np.stack(
                    [
                        episode_values[(*left_key, s)] - episode_values[(*right_key, s)]
                        for s in seeds
                    ]
                )
                contrasts.append(
                    {
                        "split": split,
                        "train_episodes": count,
                        "comparison": f"{left[0]}:{left[1]} minus {right[0]}:{right[1]}",
                        "seeds": seeds,
                        **paired_interval(delta),
                    }
                )
            seeds = sorted(s for s, _ in macros[(split, count, "temporal_gnn", "jepa")])
            delta = np.stack(
                [
                    (
                        episode_values[
                            (split, count, "canonical_temporal_gnn", "jepa", s)
                        ]
                        - episode_values[
                            (split, count, "canonical_temporal_gnn", "supervised", s)
                        ]
                    )
                    - (
                        episode_values[(split, count, "temporal_gnn", "jepa", s)]
                        - episode_values[
                            (split, count, "temporal_gnn", "supervised", s)
                        ]
                    )
                    for s in seeds
                ]
            )
            contrasts.append(
                {
                    "split": split,
                    "train_episodes": count,
                    "comparison": "canonical_minus_ordinary_JEPA_effect",
                    "seeds": seeds,
                    **paired_interval(delta),
                }
            )
    flags = []
    for d in diagnostics:
        if d.get("latent_checks", {}).get("gross_collapse_flags"):
            flags.append(
                {
                    "run_id": d["run_id"],
                    "issue": "gross_representation_collapse",
                    "details": d["latent_checks"]["gross_collapse_flags"],
                }
            )
        if "coordinate_check" in d and not d["coordinate_check"]["within_bound"]:
            flags.append({"run_id": d["run_id"], "issue": "coordinate_check_failed"})
    baselines = [
        json.loads((args.runs / "evaluation" / f"{method}_{split}.json").read_text())
        for split in completed["splits"]
        for method in ("hold", "constant_velocity")
    ]
    result = {
        "status": "complete_with_diagnostic_flags" if flags else "complete",
        "matrix_sha256": file_sha256(args.runs / "matrix.json"),
        "selection_seal_sha256": file_sha256(args.runs / "selection_seal.json"),
        "summary": summary,
        "groups": groups,
        "paired_contrasts": contrasts,
        "baselines": baselines,
        "runs": records,
        "costs": costs,
        "diagnostics": diagnostics,
        "flags": flags,
        "uncertainty": "Seed SD uses all three formal seeds. Paired bootstrap jointly resamples seed labels and shared episode IDs; intervals with three seeds are approximate. Sparse proximity groups are descriptive, not contact labels.",
        **source_version(),
    }
    output = args.runs / "analysis"
    write_json(output / "analysis.json", result)
    for name, rows in [
        ("summary", summary),
        ("groups", groups),
        ("paired_contrasts", contrasts),
    ]:
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Formal experiment results",
        "",
        f"Status: **{result['status']}**. {len(plan['runs'])} trained configurations; {completed['prediction_sets']} evaluated prediction sets.",
        "",
        "All mean errors are in cm. Smaller is better. SD describes training-seed variation, not an episode confidence interval.",
        "",
        "| Split | Training episodes | Model | Method | Mean | Seed SD |",
        "| --- | ---: | --- | --- | ---: | ---: |",
    ]
    lines += [
        f"| {r['split']} | {r['train_episodes']} | {r['model']} | {r['method']} | {r['mean_cm']:.5f} | {r['seed_sd_cm']:.5f} |"
        if r["seed_sd_cm"] is not None
        else f"| {r['split']} | {r['train_episodes']} | {r['model']} | {r['method']} | {r['mean_cm']:.5f} | n/a |"
        for r in summary
    ]
    lines += [
        "",
        "![Data efficiency](data_efficiency.png)",
        "",
        "![Physical errors by horizon](horizons.png)",
        "",
        "![Measured training cost](training_cost.png)",
        "",
        "See `analysis.json`, `groups.csv`, and `paired_contrasts.csv` for all seeds, motion/proximity groups, diagnostic flags, and paired differences. Negative differences favor the first method. All failed attempts remain in the queue log; no failed seed is omitted from a completed matrix.",
        "",
        "D1 is a physical distribution shift; coordinate-transformation residuals are reported separately. Small proximity groups cannot establish contact-stage behavior. Findings from these tests are not a basis for retroactively retuning the frozen matrix.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    print(result["status"], len(summary), "summary rows", flush=True)


if __name__ == "__main__":
    main()
