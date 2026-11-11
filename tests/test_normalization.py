import h5py
import numpy as np

from robot_dynamics.normalization import fit_statistics


def test_statistics_ignore_heldout_episodes_and_unused_training_pool(tmp_path):
    with h5py.File(tmp_path / "cache.h5", "w") as cache:
        cache.attrs["dt_seconds"] = 0.05
        for i in range(3):
            g = cache.create_group(f"demo_{i}")
            g.create_dataset("time", data=np.arange(5) * 0.05)
            for key in ("eef_position", "nut_position"):
                g.create_dataset(key, data=np.arange(15).reshape(5, 3) * 0.01 + i)
            for key in ("peg1/position", "table/surface_position"):
                g.create_dataset(key, data=np.array([0.0, 0.0, 0.8]))
            g.create_dataset("gripper_qpos", data=np.zeros((5, 2)))
            g.create_dataset("robot_joint_qpos", data=np.zeros((5, 7)))
        before = fit_statistics(cache, ["demo_0"])
        cache["demo_1/eef_position"][:] = 1e8
        cache["demo_2/gripper_qpos"][:] = -1e8
        assert before == fit_statistics(cache, ["demo_0"])
        assert before["scalar_fields"]["gripper_qpos"]["constant_dimensions"] == [0, 1]
        assert before["train_episode_ids"] == ["demo_0"]
