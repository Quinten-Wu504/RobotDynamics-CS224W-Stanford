import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import Delaunay

from robot_dynamics.baselines import mean_interval
from robot_dynamics.data_audit import file_sha256, source_version


def initial_xy(cache, name):
    episode = cache[name]
    relative = episode["nut_position"][0] - episode["peg1/position"][:]
    return (episode["peg1/rotation"][:].T @ relative)[:2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("outputs/stage6"))
    parser.add_argument("--stage4", type=Path, default=Path("outputs/stage4"))
    parser.add_argument("--stage5", type=Path, default=Path("outputs/stage5"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(Path("reports/stage3/manifest.json").read_text())
    if file_sha256(args.cache) != manifest["datasets"]["square_d0"]["cache_sha256"]:
        raise ValueError("Cache checksum mismatch")
    train_ids = manifest["d0"]["train_subsets"]["0.1"]
    val_ids = manifest["d0"]["validation"]
    with h5py.File(args.cache, "r") as cache:
        train = np.stack([initial_xy(cache, name) for name in train_ids])
        validation = np.stack([initial_xy(cache, name) for name in val_ids])
    inside = Delaunay(train).find_simplex(validation) >= 0
    groups = {
        label: [name for name, keep in zip(val_ids, mask) if keep]
        for label, mask in (
            ("inside_train_hull", inside),
            ("outside_train_hull", ~inside),
        )
    }
    rows = []
    locations = {
        "gru": args.stage4,
        "temporal_gnn": args.stage5,
        "canonical_gru": args.runs,
        "canonical_temporal_gnn": args.runs,
    }
    for kind, parent in locations.items():
        directory = parent / f"pilot_{kind}"
        metrics = json.loads((directory / "metrics.json").read_text())
        if metrics["train_ids"] != train_ids or metrics["evaluation_ids"] != val_ids:
            raise ValueError("Episode split mismatch")
        with (directory / "episodes.csv").open(newline="") as stream:
            errors = list(csv.DictReader(stream))
        for group, ids in groups.items():
            if not ids:
                continue
            for entity in ("end_effector", "square_nut"):
                for horizon in (1, 5, 10):
                    selected = [
                        r
                        for r in errors
                        if r["episode"] in ids
                        and r["entity"] == entity
                        and int(r["horizon_steps"]) == horizon
                    ]
                    if {r["episode"] for r in selected} != set(ids):
                        raise ValueError("Missing grouped episode errors")
                    rows.append(
                        {
                            "model": kind,
                            "group": group,
                            "entity": entity,
                            "horizon_steps": horizon,
                            "episodes": len(ids),
                            "windows": sum(int(r["windows"]) for r in selected),
                            **mean_interval(
                                [float(r["mean_error_cm"]) for r in selected]
                            ),
                        }
                    )
    result = {
        "rule": "Initial nut XY in peg frame, inside/outside convex hull of the 80 training placements; no error-based threshold.",
        "interpretation": "Real D0 placement coverage diagnostic, not a separate-distribution or D1 test.",
        "train_episode_ids": train_ids,
        "train_xy_m": train.tolist(),
        "validation_episode_ids": val_ids,
        "validation_xy_m": validation.tolist(),
        "groups": groups,
        "summary": rows,
        **source_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print({key: len(value) for key, value in groups.items()})


if __name__ == "__main__":
    main()
