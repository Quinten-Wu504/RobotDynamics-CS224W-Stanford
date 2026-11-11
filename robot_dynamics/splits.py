import argparse
import csv
import json
from pathlib import Path

import numpy as np

from robot_dynamics.data_audit import source_version


def make_split(names, seed=224):
    names = sorted(names, key=lambda x: int(x.split("_")[-1]))
    if len(names) != len(set(names)):
        raise ValueError("Duplicate episode IDs")
    ordered = np.random.default_rng(seed).permutation(names).tolist()
    n_train, n_valid = int(0.8 * len(names)), int(0.1 * len(names))
    train = ordered[:n_train]
    return {
        "train": train,
        "validation": ordered[n_train : n_train + n_valid],
        "test": ordered[n_train + n_valid :],
        "train_subsets": {str(f): train[: int(f * n_train)] for f in (0.1, 0.25, 1.0)},
    }


def validate_split(split, names):
    groups = [split[k] for k in ("train", "validation", "test")]
    flat = sum(groups, [])
    if len(flat) != len(set(flat)) or set(flat) != set(names):
        raise ValueError("Splits must cover each episode exactly once")
    for fraction, subset in split["train_subsets"].items():
        if subset != split["train"][: int(float(fraction) * len(split["train"]))]:
            raise ValueError("Training subsets must be nested prefixes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=Path("reports/stage2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    datasets = {}
    for task in ("square_d0", "square_d1"):
        summary = json.loads((args.audit_dir / f"{task}.json").read_text())
        with (args.audit_dir / f"{task}_episodes.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        datasets[task] = {
            "source_sha256": summary["sha256"],
            "cache_sha256": summary["cache"]["sha256"],
            "episode_lengths": {r["episode"]: int(r["length"]) for r in rows},
        }
    split = make_split(datasets["square_d0"]["episode_lengths"])
    validate_split(split, datasets["square_d0"]["episode_lengths"])
    result = {
        "seed": 224,
        "generator": "numpy.default_rng PCG64 permutation of numeric episode order",
        "rounding": "floor for train, validation, and subset sizes; remainder to test",
        "d0": split,
        "d1_final_test": list(datasets["square_d1"]["episode_lengths"]),
        "datasets": datasets,
        **source_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
