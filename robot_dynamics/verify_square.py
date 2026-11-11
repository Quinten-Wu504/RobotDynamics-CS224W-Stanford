import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.square_state import fixture_poses, rotations_xyzw


def verify(dataset, names):
    import mimicgen  # noqa: F401 -- registers the Square variants
    import robosuite

    results = []
    with h5py.File(dataset, "r") as source:
        meta = json.loads(source["data"].attrs["env_args"])
        kwargs = dict(meta["env_kwargs"])
        kwargs.update(
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False
        )
        env = robosuite.make(meta["env_name"], **kwargs)
        for name in names:
            group = source["data"][name]
            env.reset()
            env.reset_from_xml_string(env.edit_model_xml(group.attrs["model_file"]))
            env.sim.reset()
            sim = env.sim
            states, actions = group["states"][:], group["actions"][:]
            obs = group["obs"]
            fixture = fixture_poses(group.attrs["model_file"])
            errors = {}
            restored_eef, restored_nut = [], []

            def record(key, a, b):
                errors[key] = max(errors.get(key, 0), float(np.max(np.abs(a - b))))

            eef_id = env.robots[0].eef_site_id
            hand_id = sim.model.body_name2id(env.robots[0].robot_model.eef_name)
            nut_id = env.obj_body_id["SquareNut"]
            peg_id = sim.model.body_name2id("peg1")
            table_id = sim.model.geom_name2id("table_collision")
            for t, state in enumerate(states):
                sim.set_state_from_flattened(state)
                sim.forward()
                current = env._get_observations(force_update=True)
                for key in (
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                ):
                    record(key, current[key], obs[key][t])
                record(
                    "robot_joint_qpos",
                    sim.data.qpos[env.robots[0]._ref_joint_pos_indexes],
                    obs["robot0_joint_pos"][t],
                )
                restored_eef.append(sim.data.site_xpos[eef_id].tolist())
                restored_nut.append(sim.data.body_xpos[nut_id].tolist())
                record("object_vector", current["object-state"], obs["object"][t])
                record(
                    "eef_world_position",
                    sim.data.site_xpos[eef_id],
                    obs["robot0_eef_pos"][t],
                )
                record(
                    "eef_world_rotation",
                    sim.data.body_xmat[hand_id].reshape(3, 3),
                    rotations_xyzw(obs["robot0_eef_quat"][t : t + 1])[0],
                )
                record(
                    "nut_world_position",
                    sim.data.body_xpos[nut_id],
                    obs["object"][t, :3],
                )
                record(
                    "nut_world_rotation",
                    sim.data.body_xmat[nut_id].reshape(3, 3),
                    rotations_xyzw(obs["object"][t : t + 1, 3:7])[0],
                )
                record(
                    "peg_world_position",
                    sim.data.body_xpos[peg_id],
                    fixture["peg1"]["position"],
                )
                record(
                    "peg_world_rotation",
                    sim.data.body_xmat[peg_id].reshape(3, 3),
                    fixture["peg1"]["rotation"],
                )
                record(
                    "table_geom_position",
                    sim.data.geom_xpos[table_id],
                    fixture["table"]["geom_position"],
                )
                record(
                    "table_geom_rotation",
                    sim.data.geom_xmat[table_id].reshape(3, 3),
                    fixture["table"]["geom_rotation"],
                )
                record(
                    "table_half_size",
                    sim.model.geom_size[table_id],
                    fixture["table"]["half_size"],
                )
            # A restored MuJoCo state lacks solver/controller history. Report replay
            # differences rather than requiring bitwise dynamic equivalence.
            env.reset_from_xml_string(env.edit_model_xml(group.attrs["model_file"]))
            env.sim.reset()
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()
            replay = []
            for t in range(min(20, len(states) - 1)):
                env.step(actions[t])
                env.sim.forward()
                actual = env.sim.data.site_xpos[env.robots[0].eef_site_id].copy()
                replay.append(
                    {
                        "t": t,
                        "time": float(env.sim.data.time),
                        "next_error_m": float(
                            np.linalg.norm(actual - obs["robot0_eef_pos"][t + 1])
                        ),
                        "unadvanced_error_m": float(
                            np.linalg.norm(actual - obs["robot0_eef_pos"][t])
                        ),
                    }
                )
            results.append(
                {
                    "episode": name,
                    "restored_frames": len(states),
                    "max_absolute_errors": errors,
                    "replay_first_20": replay,
                    "restoration_passed": max(errors.values()) < 1e-6,
                    "restored_eef_position": restored_eef,
                    "restored_nut_position": restored_nut,
                }
            )
            print(name, len(states), errors, flush=True)
        env.close()
    return {
        "dataset": dataset.name,
        "source_sha256": file_sha256(dataset),
        "episodes": results,
        **source_version(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--episodes", nargs="+", default=["demo_0", "demo_499", "demo_999"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = verify(args.dataset, args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not all(item["restoration_passed"] for item in result["episodes"]):
        raise ValueError("Restored state disagrees with recorded observations")


if __name__ == "__main__":
    main()
