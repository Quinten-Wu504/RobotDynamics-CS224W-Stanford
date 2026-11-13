import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import h5py
import joblib
import numpy as np
from scipy.spatial.transform import Rotation

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.formal import write_json


REPOSITORY = "nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL"
REVISION = "1ef0e1c0319f79c927941a5c1828f635837a2f30"
CATEGORY = "pickup_table"


def listing(endpoint, kind):
    url = f"{endpoint}/api/datasets/{REPOSITORY}/tree/{REVISION}/data/{CATEGORY}/{kind}?limit=1000"
    rows = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            rows.extend(json.load(response))
            link = response.headers.get("Link", "")
            url = next(
                (
                    x.split("<")[1].split(">")[0]
                    for x in link.split(",")
                    if 'rel="next"' in x
                ),
                None,
            )
    return {
        Path(r["path"]).stem: r
        for r in rows
        if r["type"] == "file" and r["path"].endswith(".pkl")
    }


def download(endpoint, row, root):
    path = root / row["path"]
    path.parent.mkdir(parents=True, exist_ok=True)

    def verified():
        if not path.exists() or path.stat().st_size != row["size"]:
            return False
        if "lfs" in row:
            return file_sha256(path) == row["lfs"]["oid"]
        content = path.read_bytes()
        return (
            hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            == row["oid"]
        )

    if not verified():
        url = f"{endpoint}/datasets/{REPOSITORY}/resolve/{REVISION}/{row['path']}"
        temporary = path.with_suffix(".part")
        for attempt in range(4):
            try:
                with (
                    urllib.request.urlopen(url, timeout=90) as response,
                    temporary.open("wb") as stream,
                ):
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                temporary.replace(path)
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        if not verified():
            raise ValueError(f"Downloaded file hash differs from pinned source: {path}")
    return path


def extract(paths):
    robot, obj, meta = (joblib.load(p) for p in paths)
    if len(robot) != 1 or set(robot) != set(obj):
        raise ValueError("Robot/object source IDs do not match")
    source_id = next(iter(robot))
    robot, obj = robot[source_id], obj[source_id]
    if robot["fps"] != 25 or obj["fps"] != 25:
        raise ValueError("Unexpected trajectory rate")
    root = np.asarray(robot["root_trans_offset"])
    count = len(root)
    fields = {
        "root": (root, (count, 3)),
        "object": (obj["root_pos"], (count, 1, 3)),
        "root_quat": (robot["root_rot"], (count, 4)),
        "object_quat": (obj["root_quat"], (count, 1, 4)),
        "dof": (robot["dof"], (count, 29)),
        "hand_dof": (robot["hand_dof_pos"], (count, 14)),
        "table_pos": (meta["table_pos"], (3,)),
        "table_quat": (meta["table_quat"], (4,)),
        "table_size": (meta["table_size"], (3,)),
    }
    for name, (value, shape) in fields.items():
        if np.shape(value) != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name}: {np.shape(value)}")
    if count <= 20:
        raise ValueError("Motion too short for the frozen window")
    pos = np.stack(
        [root, obj["root_pos"][:, 0], np.broadcast_to(meta["table_pos"], root.shape)],
        axis=1,
    )
    quat = np.stack(
        [
            robot["root_rot"],
            obj["root_quat"][:, 0],
            np.broadcast_to(meta["table_quat"], (count, 4)),
        ],
        axis=1,
    )
    norms = np.linalg.norm(quat, axis=-1)
    if np.max(np.abs(norms - 1)) > 0.005:
        raise ValueError("Nonunit release quaternion")
    # The public release is xyzw; never infer conventions from a trajectory.
    rotation = Rotation.from_quat(quat.reshape(-1, 4)).as_matrix().reshape(count, 3, 9)
    joints = np.concatenate([robot["dof"], robot["hand_dof_pos"]], axis=1)
    return {
        "position": pos.astype("float32"),
        "rotation": rotation.astype("float32"),
        "joints": joints.astype("float32"),
        "table_size": np.asarray(meta["table_size"], dtype="float32"),
    }, {
        "source_id": source_id,
        "frames": count,
        "fps": 25,
        "robot_fields": sorted(robot),
        "object_fields": sorted(obj),
        "meta_fields": sorted(meta),
        "quaternion_max_norm_error": float(np.max(np.abs(norms - 1))),
    }


def prepare(root, output, endpoint, workers):
    output.mkdir(parents=True, exist_ok=True)
    inventory_path = output / "inventory.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
    else:
        inventory = {
            kind: listing(endpoint, kind) for kind in ("robot", "objects", "meta")
        }
        write_json(inventory_path, inventory)
    if len(inventory["robot"]) != 2991:
        raise ValueError("Pinned pickup_table inventory is incomplete")
    groups = {}
    for name in sorted(inventory["robot"]):
        groups.setdefault(name.split("__")[1], []).append(name)
    rng = np.random.default_rng(224)
    eligible = sorted(k for k, v in groups.items() if len(v) >= 2)
    assets = rng.permutation(eligible)[:120].tolist()
    split_assets = {
        "train": assets[:84],
        "validation": assets[84:102],
        "test": assets[102:],
    }
    splits = {
        split: sorted(
            name
            for asset in selected
            for name in rng.choice(groups[asset], 2, replace=False).tolist()
        )
        for split, selected in split_assets.items()
    }
    names = sorted(sum(splits.values(), []))
    requests = [
        inventory[kind][name] for name in names for kind in ("robot", "objects", "meta")
    ]
    print(
        json.dumps(
            {"motions": len(names), "download_bytes": sum(r["size"] for r in requests)}
        ),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(
            pool.map(lambda row: download(endpoint, row, root), requests), 1
        ):
            if i % 30 == 0:
                print(f"Verified {i}/{len(requests)} files", flush=True)
    cache_path = output / "cache.h5"
    temporary = output / "cache.tmp.h5"
    audit, digests = {}, {}
    with h5py.File(temporary, "w") as cache:
        cache.attrs.update(fps=25, category=CATEGORY, revision=REVISION, complete=False)
        for name in names:
            paths = [
                root / inventory[k][name]["path"] for k in ("robot", "objects", "meta")
            ]
            values, record = extract(paths)
            digest = hashlib.sha256(
                b"".join(values[k].tobytes() for k in sorted(values))
            ).hexdigest()
            if digest in digests:
                raise ValueError(f"Duplicate motion states: {name}, {digests[digest]}")
            digests[digest] = name
            group = cache.create_group(name)
            for key, value in values.items():
                group.create_dataset(key, data=value, compression="gzip")
            audit[name] = {
                **record,
                "asset": name.split("__")[1],
                "state_sha256": digest,
                "files": {
                    k: {"path": inventory[k][name]["path"], "sha256": file_sha256(p)}
                    for k, p in zip(("robot", "objects", "meta"), paths)
                },
            }
        cache.attrs["complete"] = True
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "category": CATEGORY,
        "selection_seed": 224,
        "splits": splits,
        "split_assets": split_assets,
        "cache_sha256": file_sha256(temporary),
        "audit": audit,
    }
    existing = output / "manifest.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        # HDF5 headers can differ; compare the audited numerical source identity.
        if {k: v for k, v in old.items() if k != "cache_sha256"} != {
            k: v for k, v in manifest.items() if k != "cache_sha256"
        }:
            raise ValueError("Existing frozen data manifest differs")
        if file_sha256(cache_path) != old["cache_sha256"]:
            raise ValueError("Existing cache was modified")
        temporary.unlink()
    else:
        temporary.replace(cache_path)
        write_json(existing, manifest)
    write_json(
        output / "preparation.json",
        {
            "passed": True,
            "motions": len(names),
            "assets": len(assets),
            "split_counts": {k: len(v) for k, v in splits.items()},
            "source": source_version(),
            "extraction_sha256": file_sha256(Path(__file__)),
            "test_predictions_computed": False,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    prepare(args.raw, args.output, args.endpoint.rstrip("/"), args.workers)


if __name__ == "__main__":
    main()
