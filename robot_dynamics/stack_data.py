import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from robot_dynamics.cache_square import check_controller
from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.formal import write_json
from robot_dynamics.splits import make_split, validate_split
from robot_dynamics.square_state import rotations_xyzw, xml_transform


ENTITIES = ("end_effector", "cubeA", "cubeB")
RELEASE = {
    "stack_d0": (
        1138107058,
        "3b687047f9d6446da762c9afe4fcb87d77a1a7c2ddf7527c56965311777868f3",
    ),
    "stack_d1": (
        1072340218,
        "edc9815a2738a583dcb3e7a9e239a71eca6057396a839e52b022935c273cfd61",
    ),
}


def geometry(xml):
    root = ET.fromstring(xml)
    table = root.find(".//body[@name='table']")
    if table is None or table.find("joint") is not None:
        raise ValueError("Expected fixed table")
    pos, rot = xml_transform(table)
    geom = table.find("geom[@name='table_collision']")
    offset, geom_rot = xml_transform(geom)
    half = np.fromstring(geom.attrib["size"], sep=" ")
    if geom.attrib.get("type") != "box" or half.shape != (3,):
        raise ValueError("Expected table box")
    # Stack's table is a direct worldbody child. Nested frames require a decoder change.
    if table not in list(root.find("worldbody")):
        raise ValueError("Nested table transform")
    center, rotation = pos + rot @ offset, rot @ geom_rot
    sizes = [np.zeros(3)]
    for name in ("cubeA", "cubeB"):
        g = root.find(f".//geom[@name='{name}_g0']")
        if g is None or g.get("type") != "box":
            raise ValueError(f"Missing box geometry for {name}")
        size = np.fromstring(g.attrib["size"], sep=" ")
        gp, gr = xml_transform(g)
        if (
            size.shape != (3,)
            or not np.all(size > 0)
            or not np.allclose(gp, 0)
            or not np.allclose(gr, np.eye(3))
        ):
            raise ValueError("Unexpected cube geometry offset")
        sizes.append(size)
    sizes.append(half)
    return center + rotation[:, 2] * half[2], rotation, np.stack(sizes)


def decode(group):
    obs = group["obs"]
    n = len(group["actions"])
    expected = {
        "robot0_eef_pos": 3,
        "robot0_eef_quat": 4,
        "robot0_gripper_qpos": 2,
        "robot0_joint_pos": 7,
        "object": 23,
    }
    for name, width in expected.items():
        if obs[name].shape != (n, width) or not np.isfinite(obs[name][:]).all():
            raise ValueError(f"Invalid {group.name}/{name}")
    states, actions = group["states"][:], group["actions"][:]
    if (
        n < 14
        or int(group.attrs["num_samples"]) != n
        or states.ndim != 2
        or len(states) != n
        or actions.shape != (n, 7)
    ):
        raise ValueError("Invalid episode dimensions")
    if (
        not np.isfinite(states).all()
        or not np.isfinite(actions).all()
        or np.abs(actions).max() > 1 + 1e-8
    ):
        raise ValueError("Invalid state/action values")
    if not np.allclose(np.diff(states[:, 0]), 0.05, atol=1e-9, rtol=0):
        raise ValueError("Expected 20 Hz timestamps")
    obj = obs["object"][:]
    positions = np.stack([obs["robot0_eef_pos"][:], obj[:, :3], obj[:, 7:10]], axis=1)
    relative = np.concatenate(
        [
            positions[:, 1] - positions[:, 0],
            positions[:, 2] - positions[:, 0],
            positions[:, 2] - positions[:, 1],
        ],
        axis=1,
    )
    if not np.allclose(relative, obj[:, 14:], atol=1e-7, rtol=0):
        raise ValueError("Cube observation ordering disagrees with relative positions")
    rotations = np.stack(
        [
            rotations_xyzw(obs["robot0_eef_quat"][:]),
            rotations_xyzw(obj[:, 3:7]),
            rotations_xyzw(obj[:, 10:14]),
        ],
        axis=1,
    )
    table, table_rotation, sizes = geometry(group.attrs["model_file"])
    return {
        "time": states[:, 0],
        "actions": actions,
        "position": positions,
        "rotation": rotations,
        "robot": np.concatenate(
            [obs["robot0_gripper_qpos"][:], obs["robot0_joint_pos"][:]], axis=1
        ),
        "table_position": table,
        "table_rotation": table_rotation,
        "half_size": sizes,
    }


def cache_file(raw, output):
    if output.exists():
        raise FileExistsError(output)
    size, sha = RELEASE[raw.stem]
    if raw.stat().st_size != size or file_sha256(raw) != sha:
        raise ValueError("Release hash mismatch")
    rows = []
    with h5py.File(raw, "r") as source, h5py.File(output, "x") as cache:
        meta = json.loads(source["data"].attrs["env_args"])
        check_controller(meta)
        if (
            meta["env_name"].lower() != raw.stem
            or meta["env_kwargs"]["control_freq"] != 20
        ):
            raise ValueError("Unexpected Stack environment")
        cache.attrs.update(
            complete=False,
            task="stack",
            source_sha256=sha,
            dt_seconds=0.05,
            environment=json.dumps(meta),
            **source_version(),
        )
        for name in sorted(source["data"], key=lambda n: int(n.split("_")[-1])):
            fields = decode(source["data"][name])
            target = cache.create_group(name)
            for key, value in fields.items():
                target.create_dataset(key, data=value)
            fingerprint = hashlib.sha256(
                source["data"][name]["states"][:].tobytes()
                + fields["actions"].tobytes()
            ).hexdigest()
            rows.append(
                {
                    "episode": name,
                    "states": len(fields["time"]),
                    "trajectory_sha256": fingerprint,
                }
            )
        if len(rows) != 1000 or len({r["trajectory_sha256"] for r in rows}) != len(
            rows
        ):
            raise ValueError("Unexpected episode count or duplicate trajectories")
        cache.attrs["complete"] = True
    return {
        "source_sha256": sha,
        "cache_sha256": file_sha256(output),
        "episodes": rows,
        "environment": meta,
    }


def fit_statistics(cache, ids):
    positions, robot, speeds = [], [], []
    for name in ids:
        g = cache[name]
        p = g["position"][:]
        positions.append(
            np.concatenate(
                [p, np.broadcast_to(g["table_position"][:], (len(p), 1, 3))], axis=1
            ).reshape(-1, 3)
        )
        robot.append(g["robot"][:])
        speeds.append(np.linalg.norm(np.diff(p, axis=0), axis=-1) / 0.05)
    points, scalars = np.concatenate(positions), np.concatenate(robot)
    center = points.mean(axis=0)
    scale = float(np.sqrt(np.mean((points - center) ** 2)))
    if not scale > 0:
        raise ValueError("Zero geometry scale")
    return {
        "train_episode_ids": ids,
        "center": center.tolist(),
        "scale": scale,
        "robot_mean": scalars.mean(axis=0).tolist(),
        "robot_std": np.maximum(scalars.std(axis=0), 1e-12).tolist(),
        "motion_threshold_m_per_s": np.maximum(
            np.quantile(np.concatenate(speeds), 0.25, axis=0), 1e-6
        ).tolist(),
    }


def episode_arrays(group, stats):
    pos, rotation, robot = group["position"][:], group["rotation"][:], group["robot"][:]
    n, scale = len(pos), stats["scale"]
    nodes = np.zeros((n, 4, 37), dtype=np.float32)
    p = np.concatenate(
        [pos, np.broadcast_to(group["table_position"][:], (n, 1, 3))], axis=1
    )
    nodes[:, :, :3] = (p - stats["center"]) / scale
    nodes[:, :3, 3:12] = rotation.reshape(n, 3, 9)
    nodes[:, 3, 3:12] = group["table_rotation"][:].ravel()
    nodes[:, :, 12:15] = group["half_size"][:] / scale
    nodes[:, 0, 15:24] = (robot - stats["robot_mean"]) / stats["robot_std"]
    nodes[:, :, 24:28] = np.eye(4)
    nodes[:, :, 28:37] = ((p[:, None, :3, :] - p[:, :, None, :]) / scale).reshape(
        n, 4, 9
    )
    states, contexts, targets, horizons, times, motion = [], [], [], [], [], []
    actions = group["actions"][:]
    for j, k in enumerate((1, 5, 10)):
        t = np.arange(3, n - k)
        states.append(nodes[t[:, None] + np.arange(-3, 1)].reshape(len(t), 4, 148))
        a = np.zeros((len(t), 10, 7), dtype=np.float32)
        a[:, :k] = actions[t[:, None] + np.arange(k)]
        contexts.append(
            np.concatenate(
                [a.reshape(len(t), 70), np.broadcast_to(np.eye(3)[j], (len(t), 3))],
                axis=1,
            )
        )
        targets.append(((pos[t + k] - pos[t]) / scale).reshape(len(t), 9))
        horizons.append(np.full(len(t), k))
        times.append(t)
        motion.append(np.linalg.norm(pos[t] - pos[t - 1], axis=-1) / 0.05)
    return {
        "state": np.concatenate(states).astype("float32"),
        "context": np.concatenate(contexts).astype("float32"),
        "target": np.concatenate(targets).astype("float32"),
        "horizon": np.concatenate(horizons),
        "anchor": np.concatenate(times),
        "speed": np.concatenate(motion),
    }


def make_arrays(cache, ids, stats):
    chunks = [episode_arrays(cache[name], stats) for name in ids]
    return {
        **{k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]},
        "episode": np.concatenate(
            [np.full(len(c["state"]), i) for i, c in enumerate(chunks)]
        ),
        "ids": ids,
    }


def error_rows(pred, arrays, scale):
    errors = (
        np.linalg.norm((pred - arrays["target"]).reshape(-1, 3, 3), axis=-1)
        * scale
        * 100
    )
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite predictions")
    return [
        {
            "episode": name,
            "entity": entity,
            "horizon_steps": k,
            "windows": int(mask.sum()),
            "mean_error_cm": float(errors[mask, j].mean()),
        }
        for i, name in enumerate(arrays["ids"])
        for k in (1, 5, 10)
        for mask in [(arrays["episode"] == i) & (arrays["horizon"] == k)]
        for j, entity in enumerate(ENTITIES)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = {
        task: cache_file(args.raw / f"{task}.hdf5", args.output / f"{task}.hdf5")
        for task in RELEASE
    }
    hashes = [{r["trajectory_sha256"] for r in v["episodes"]} for v in records.values()]
    if hashes[0] & hashes[1]:
        raise ValueError("Duplicate trajectory across distributions")
    split = make_split([r["episode"] for r in records["stack_d0"]["episodes"]])
    validate_split(split, [r["episode"] for r in records["stack_d0"]["episodes"]])
    write_json(
        args.output / "manifest.json",
        {
            "d0": split,
            "datasets": records,
            "d1_final_test": [r["episode"] for r in records["stack_d1"]["episodes"]],
            **source_version(),
        },
    )


if __name__ == "__main__":
    main()
