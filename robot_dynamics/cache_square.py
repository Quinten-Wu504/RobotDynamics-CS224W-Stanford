import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.square_state import fixture_poses, rotations_xyzw


def check_controller(metadata):
    settings = metadata["env_kwargs"]
    controller = settings["controller_configs"]
    expected = {
        "type": "OSC_POSE",
        "impedance_mode": "fixed",
        "control_delta": True,
        "input_min": -1,
        "input_max": 1,
        "interpolation": None,
        "position_limits": None,
        "orientation_limits": None,
    }
    if settings["robots"] != ["Panda"] or any(
        controller.get(k) != v for k, v in expected.items()
    ):
        raise ValueError(
            "Controller differs from the audited Panda OSC_POSE configuration"
        )
    for key, values in {
        "output_max": [0.05] * 3 + [0.5] * 3,
        "output_min": [-0.05] * 3 + [-0.5] * 3,
    }.items():
        if not np.array_equal(controller[key], values):
            raise ValueError(f"Unexpected controller scaling: {key}")


def check_episode(episode, frequency, object_width):
    n = len(episode["actions"])
    required = {
        "actions": (n, 7),
        "states": (n, 45),
        "obs/object": (n, object_width),
        "obs/robot0_eef_pos": (n, 3),
        "obs/robot0_eef_quat": (n, 4),
        "obs/robot0_gripper_qpos": (n, 2),
        "obs/robot0_joint_pos": (n, 7),
    }
    if n < 2 or int(episode.attrs["num_samples"]) != n:
        raise ValueError(f"Invalid episode length: {episode.name}")
    for key, shape in required.items():
        value = episode[key]
        if value.shape != shape or not np.isfinite(value[:]).all():
            raise ValueError(f"Invalid field: {episode.name}/{key}")
    if not np.allclose(
        np.diff(episode["states"][:, 0]), 1 / frequency, atol=1e-9, rtol=0
    ):
        raise ValueError(f"Nonuniform timestamps: {episode.name}")
    if np.abs(episode["actions"][:]).max() > 1 + 1e-8:
        raise ValueError(f"Actions outside normalized controller range: {episode.name}")
    return n


def cache_file(dataset, output):
    if output.exists():
        raise FileExistsError(output)
    source_hash = file_sha256(dataset)
    with h5py.File(dataset, "r") as source, h5py.File(output, "x") as cache:
        metadata = json.loads(source["data"].attrs["env_args"])
        check_controller(metadata)
        if metadata["env_name"] not in ("Square_D0", "Square_D1"):
            raise ValueError("Expected Square D0 or D1")
        frequency = metadata["env_kwargs"]["control_freq"]
        if frequency != 20:
            raise ValueError("This audited decoder expects 20 Hz")
        width = 14 if metadata["env_name"] == "Square_D0" else 17
        cache.attrs.update(
            {
                "complete": False,
                "schema_version": 1,
                "source_sha256": source_hash,
                "source_file": dataset.name,
                "environment": json.dumps(metadata),
                "dt_seconds": 1 / frequency,
                "position_unit": "m",
                "frame": "world",
                "rotation_convention": "body-to-world matrix; source quaternions xyzw",
                "timing": "state[t], action[t] -> state[t+1]; final action has no recorded successor",
                **source_version(),
            }
        )
        for name in sorted(source["data"], key=lambda s: int(s.split("_")[-1])):
            episode = source["data"][name]
            n = check_episode(episode, frequency, width)
            poses = fixture_poses(episode.attrs["model_file"])
            obs = episode["obs"]
            obj = obs["object"][:]
            if width == 17 and not np.allclose(
                obj[:, 14:17], poses["peg1"]["position"], atol=1e-8, rtol=0
            ):
                raise ValueError(
                    f"D1 peg observation disagrees with episode XML: {name}"
                )
            group = cache.create_group(name)
            fields = {
                "time": episode["states"][:, 0],
                "actions": episode["actions"][:],
                "eef_position": obs["robot0_eef_pos"][:],
                "eef_rotation": rotations_xyzw(obs["robot0_eef_quat"][:]),
                "nut_position": obj[:, :3],
                "nut_rotation": rotations_xyzw(obj[:, 3:7]),
                "gripper_qpos": obs["robot0_gripper_qpos"][:],
                "robot_joint_qpos": obs["robot0_joint_pos"][:],
                "sim_states": episode["states"][:],
            }
            fields["translation_command"] = fields["actions"][:, :3] * 0.05
            fields["rotation_command"] = fields["actions"][:, 3:6] * 0.5
            for key, value in fields.items():
                group.create_dataset(key, data=value, compression="gzip", shuffle=True)
            for entity, values in poses.items():
                for key, value in values.items():
                    group.create_dataset(f"{entity}/{key}", data=value)
            group.attrs["recorded_states"] = n
            group.attrs["valid_one_step_transitions"] = n - 1
        cache.attrs["complete"] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_file(args.dataset, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
