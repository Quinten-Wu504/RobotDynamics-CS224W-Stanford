import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.formal import write_json
from robot_dynamics.stack_data import decode


def verify(path):
    import mimicgen  # noqa: F401 -- registers Stack variants
    import robosuite

    results = []
    with h5py.File(path, "r") as source:
        meta = json.loads(source["data"].attrs["env_args"])
        kwargs = dict(meta["env_kwargs"])
        kwargs.update(
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False
        )
        env = robosuite.make(meta["env_name"], **kwargs)
        for name in ("demo_0", "demo_499", "demo_999"):
            group = source["data"][name]
            fields = decode(group)
            env.reset()
            env.reset_from_xml_string(env.edit_model_xml(group.attrs["model_file"]))
            env.sim.reset()
            sim = env.sim
            errors = {}

            def record(key, left, right):
                error = float(np.max(np.abs(left - right)))
                if not np.isfinite(error):
                    raise ValueError("Nonfinite restoration difference")
                errors[key] = max(errors.get(key, 0.0), error)

            hand = sim.model.body_name2id(env.robots[0].robot_model.eef_name)
            table = sim.model.geom_name2id("table_collision")
            cube_ids = [
                sim.model.body_name2id("cubeA_main"),
                sim.model.body_name2id("cubeB_main"),
            ]
            for t, state in enumerate(group["states"][:]):
                sim.set_state_from_flattened(state)
                sim.forward()
                current = env._get_observations(force_update=True)
                record("object_vector", current["object-state"], group["obs/object"][t])
                record(
                    "eef_position",
                    sim.data.site_xpos[env.robots[0].eef_site_id],
                    fields["position"][t, 0],
                )
                record(
                    "eef_rotation",
                    sim.data.body_xmat[hand].reshape(3, 3),
                    fields["rotation"][t, 0],
                )
                for j, body in enumerate(cube_ids, 1):
                    record(
                        f"cube{j}_position",
                        sim.data.body_xpos[body],
                        fields["position"][t, j],
                    )
                    record(
                        f"cube{j}_rotation",
                        sim.data.body_xmat[body].reshape(3, 3),
                        fields["rotation"][t, j],
                    )
                    geom = sim.model.geom_name2id(("cubeA", "cubeB")[j - 1] + "_g0")
                    record(
                        f"cube{j}_half_size",
                        sim.model.geom_size[geom],
                        fields["half_size"][j],
                    )
                rot = sim.data.geom_xmat[table].reshape(3, 3)
                record(
                    "table_surface",
                    sim.data.geom_xpos[table]
                    + rot[:, 2] * sim.model.geom_size[table, 2],
                    fields["table_position"],
                )
                record("table_rotation", rot, fields["table_rotation"])
                record(
                    "table_half_size",
                    sim.model.geom_size[table],
                    fields["half_size"][3],
                )
                record(
                    "robot_joints",
                    sim.data.qpos[env.robots[0]._ref_joint_pos_indexes],
                    fields["robot"][t, 2:],
                )
                record(
                    "gripper", current["robot0_gripper_qpos"], fields["robot"][t, :2]
                )
            if max(errors.values()) > 1e-6:
                raise ValueError(f"{path.name}/{name}: restoration mismatch {errors}")
            results.append(
                {
                    "episode": name,
                    "frames": len(fields["time"]),
                    "max_absolute_errors": errors,
                }
            )
        env.close()
    return {
        "source_sha256": file_sha256(path),
        "passed": True,
        "episodes": results,
        **source_version(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(
        args.output,
        {task: verify(args.raw / f"{task}.hdf5") for task in ("stack_d0", "stack_d1")},
    )


if __name__ == "__main__":
    main()
