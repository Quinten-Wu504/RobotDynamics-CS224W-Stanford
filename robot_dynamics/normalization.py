import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.splits import validate_split


def fit_statistics(cache, train_ids):
    positions, speeds = [], [[], []]
    scalar = {key: [] for key in ("gripper_qpos", "robot_joint_qpos")}
    dt = float(cache.attrs["dt_seconds"])
    for name in train_ids:
        g = cache[name]
        n = len(g["time"])
        for j, key in enumerate(("eef_position", "nut_position")):
            p = g[key][:]
            positions.append(p)
            speeds[j].append(np.linalg.norm(np.diff(p, axis=0), axis=1) / dt)
        for key in ("peg1/position", "table/surface_position"):
            positions.append(np.broadcast_to(g[key][:], (n, 3)))
        for key in scalar:
            scalar[key].append(g[key][:])
    points = np.concatenate(positions)
    center = points.mean(axis=0)
    scale = float(np.sqrt(np.mean((points - center) ** 2)))
    if not scale > 0:
        raise ValueError("Training geometry has zero scale")
    scalars = {}
    for key, chunks in scalar.items():
        values = np.concatenate(chunks)
        std = values.std(axis=0)
        scalars[key] = {
            "mean": values.mean(axis=0).tolist(),
            "scale": np.where(std > 1e-12, std, 1).tolist(),
            "constant_dimensions": np.flatnonzero(std <= 1e-12).tolist(),
        }
    return {
        "train_episode_ids": train_ids,
        "position_center": center.tolist(),
        "position_scale": scale,
        "position_weighting": "four node positions per recorded training state, shared isotropic scale",
        "scalar_fields": scalars,
        "motion_threshold_m_per_s": [
            max(float(np.quantile(np.concatenate(v), 0.25)), 1e-6) for v in speeds
        ],
        "motion_rule": "past one-step speed <= training 25th percentile (floor 1e-6 m/s) is low_motion",
        "dt_seconds": dt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    expected = manifest["datasets"]["square_d0"]
    if file_sha256(args.cache) != expected["cache_sha256"]:
        raise ValueError("Cache checksum differs from audited cache")
    with h5py.File(args.cache, "r") as cache:
        if (
            not cache.attrs["complete"]
            or cache.attrs["source_sha256"] != expected["source_sha256"]
        ):
            raise ValueError("Incomplete or incorrect cache")
        validate_split(manifest["d0"], list(cache))
        stats = {
            fraction: fit_statistics(cache, ids)
            for fraction, ids in manifest["d0"]["train_subsets"].items()
        }
    result = {
        "subsets": stats,
        "manifest_sha256": file_sha256(args.manifest),
        "cache_sha256": expected["cache_sha256"],
        **source_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
