import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_version():
    return {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
    }


def numeric_summary(dataset):
    low, high, nonfinite = np.inf, -np.inf, 0
    if not np.issubdtype(dataset.dtype, np.number):
        return {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
    blocks = (
        (dataset[i : i + 256] for i in range(0, len(dataset), 256))
        if dataset.ndim
        else [dataset[()]]
    )
    for block in blocks:
        finite = np.isfinite(block)
        nonfinite += int(block.size - np.count_nonzero(finite))
        if np.any(finite):
            low = min(low, float(block[finite].min()))
            high = max(high, float(block[finite].max()))
    return {
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
        "nonfinite": nonfinite,
        "min": low if np.isfinite(low) else None,
        "max": high if np.isfinite(high) else None,
    }


def audit_file(path):
    issues = []
    episodes, schemas, xml_hashes, trajectory_hashes = (
        [],
        Counter(),
        Counter(),
        Counter(),
    )
    with h5py.File(path, "r") as source:
        data = source["data"]
        metadata = json.loads(data.attrs["env_args"])
        for name in sorted(data, key=lambda x: int(x.split("_")[-1])):
            episode = data[name]
            arrays = {}
            episode.visititems(
                lambda key, value: arrays.update({key: numeric_summary(value)})
                if isinstance(value, h5py.Dataset)
                else None
            )
            schema = json.dumps(
                {k: v["shape"][1:] for k, v in arrays.items()}, sort_keys=True
            )
            schemas[schema] += 1
            xml_hash = hashlib.sha256(episode.attrs["model_file"].encode()).hexdigest()
            xml_hashes[xml_hash] += 1
            digest = hashlib.sha256()
            for key in ("states", "actions"):
                digest.update(episode[key][:].tobytes())
            trajectory_hashes[digest.hexdigest()] += 1
            n = len(episode["actions"])
            if int(episode.attrs["num_samples"]) != n:
                issues.append(f"{name}: declared length mismatch")
            for key, summary in arrays.items():
                if not summary["shape"] or summary["shape"][0] != n:
                    issues.append(
                        f"{name}/{key}: leading dimension differs from actions"
                    )
                if summary.get("nonfinite", 0):
                    issues.append(f"{name}/{key}: nonfinite values")
            times = episode["states"][:, 0]
            delta = np.diff(times)
            if not np.allclose(
                delta, 1 / metadata["env_kwargs"]["control_freq"], atol=1e-9, rtol=0
            ):
                issues.append(f"{name}: unexpected timestep")
            episodes.append(
                {
                    "episode": name,
                    "length": n,
                    "declared_length": int(episode.attrs["num_samples"]),
                    "xml_sha256": xml_hash,
                    "trajectory_sha256": digest.hexdigest(),
                    "dt_min": float(delta.min()),
                    "dt_max": float(delta.max()),
                    "arrays": arrays,
                }
            )
            if len(episodes) % 100 == 0:
                print(f"Audited {path.name}: {len(episodes)} episodes", flush=True)
        lengths = [item["length"] for item in episodes]
        if sum(lengths) != int(data.attrs["total"]):
            issues.append("Dataset total does not match trajectory lengths")
        if len(schemas) != 1:
            issues.append("Field layout varies between trajectories")
        return {
            "issues": issues,
            "file": path.name,
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "environment": metadata,
            "declared_total": int(data.attrs["total"]),
            "episodes": episodes,
            "episode_count": len(episodes),
            "sample_count": sum(lengths),
            "length_percentiles": dict(
                zip(
                    ["min", "p25", "median", "p75", "max"],
                    np.percentile(lengths, [0, 25, 50, 75, 100]).tolist(),
                )
            ),
            "schemas": [
                {"count": v, "fields": json.loads(k)} for k, v in schemas.items()
            ],
            "unique_xml_count": len(xml_hashes),
            "duplicate_trajectory_groups": {
                k: v for k, v in trajectory_hashes.items() if v > 1
            },
            "mask_names": list(source.get("mask", {})),
            **source_version(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_file(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if result["issues"]:
        raise ValueError(
            f"Audit found {len(result['issues'])} issues; see {args.output}"
        )


if __name__ == "__main__":
    main()
