import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from robot_dynamics.data_audit import file_sha256, source_version


def summarize(audit, cache_path):
    result = {k: v for k, v in audit.items() if k != "episodes"}
    result["field_ranges"] = {}
    for episode in audit["episodes"]:
        for key, values in episode["arrays"].items():
            if "nonfinite" not in values:
                continue
            target = result["field_ranges"].setdefault(
                key, {"min": values["min"], "max": values["max"], "nonfinite": 0}
            )
            target["min"] = min(target["min"], values["min"])
            target["max"] = max(target["max"], values["max"])
            target["nonfinite"] += values["nonfinite"]
    result["dt_min"] = min(e["dt_min"] for e in audit["episodes"])
    result["dt_max"] = max(e["dt_max"] for e in audit["episodes"])
    result["valid_one_step_transitions"] = (
        audit["sample_count"] - audit["episode_count"]
    )
    with h5py.File(cache_path, "r") as cache:
        if (
            not cache.attrs["complete"]
            or cache.attrs["source_sha256"] != audit["sha256"]
        ):
            raise ValueError("Cache does not match the audited dataset")
        result["cache"] = {
            "file": cache_path.name,
            "sha256": file_sha256(cache_path),
            "bytes": cache_path.stat().st_size,
            "source_commit": cache.attrs["source_commit"],
            "working_tree_dirty": bool(cache.attrs["working_tree_dirty"]),
        }
        result["canonical_ranges"] = {}
        for key in (
            "eef_position",
            "nut_position",
            "gripper_qpos",
            "robot_joint_qpos",
            "peg1/position",
            "table/surface_position",
        ):
            values = [np.atleast_2d(cache[name][key][:]) for name in cache]
            result["canonical_ranges"][key] = {
                "min": np.min([a.min(axis=0) for a in values], axis=0).tolist(),
                "max": np.max([a.max(axis=0) for a in values], axis=0).tolist(),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audits", nargs=2, type=Path, required=True)
    parser.add_argument("--caches", nargs=2, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audits = [json.loads(p.read_text()) for p in args.audits]
    for audit, cache in zip(audits, args.caches):
        task = Path(audit["file"]).stem
        (args.output_dir / f"{task}.json").write_text(
            json.dumps(summarize(audit, cache), indent=2, allow_nan=False) + "\n"
        )
        with (args.output_dir / f"{task}_episodes.csv").open("w", newline="") as stream:
            fields = [
                "episode",
                "length",
                "xml_sha256",
                "trajectory_sha256",
                "dt_min",
                "dt_max",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(audit["episodes"])
    overlap = set(e["trajectory_sha256"] for e in audits[0]["episodes"]) & set(
        e["trajectory_sha256"] for e in audits[1]["episodes"]
    )
    (args.output_dir / "cross_distribution.json").write_text(
        json.dumps(
            {
                "identical_state_action_trajectories": sorted(overlap),
                "limitation": "Exact byte equality only; shared source-demonstration ancestry is not present in the released files.",
                **source_version(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
