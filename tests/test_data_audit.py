import h5py
import numpy as np
import pytest

from robot_dynamics.cache_square import check_episode
from robot_dynamics.data_audit import numeric_summary


@pytest.fixture
def episode(tmp_path):
    with h5py.File(tmp_path / "fixture.h5", "w") as f:
        group = f.create_group("demo_0")
        group.attrs["num_samples"] = 3
        for key, width in {
            "actions": 7,
            "states": 45,
            "obs/object": 14,
            "obs/robot0_eef_pos": 3,
            "obs/robot0_eef_quat": 4,
            "obs/robot0_gripper_qpos": 2,
            "obs/robot0_joint_pos": 7,
        }.items():
            group.create_dataset(key, data=np.zeros((3, width)))
        group["states"][:, 0] = [0, 0.05, 0.1]
        yield group


def test_nonfinite_audit_counts_values_without_hiding_them(episode):
    episode["actions"][0, 0] = np.nan
    episode["actions"][1, 0] = np.inf
    summary = numeric_summary(episode["actions"])
    assert summary["nonfinite"] == 2
    with pytest.raises(ValueError, match="Invalid field"):
        check_episode(episode, 20, 14)


def test_timestamp_gap_is_rejected(episode):
    episode["states"][2, 0] = 0.15
    with pytest.raises(ValueError, match="timestamps"):
        check_episode(episode, 20, 14)


def test_distribution_layout_must_match(episode):
    assert check_episode(episode, 20, 14) == 3
    with pytest.raises(ValueError, match="Invalid field"):
        check_episode(episode, 20, 17)
