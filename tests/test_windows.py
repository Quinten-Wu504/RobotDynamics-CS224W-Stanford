import numpy as np
import pytest

from robot_dynamics.windows import (
    anchors,
    window,
    normalize_window,
    predict_positions,
    position_error_cm,
    HISTORY_FIELDS,
    FIXTURE_FIELDS,
)
from robot_dynamics.splits import make_split, validate_split
from robot_dynamics.baselines import mean_interval, summarize


def sample_episode(n=15):
    data = {key: np.arange(n * 3, dtype=float).reshape(n, 3) for key in HISTORY_FIELDS}
    data.update({key: np.ones(3) for key in FIXTURE_FIELDS})
    data["time"] = np.arange(n) * 0.05
    data["actions"] = np.arange(n * 7).reshape(n, 7)
    return data


def test_split_reproducibility_disjointness_and_nested_subsets():
    names = [f"demo_{i}" for i in range(1000)]
    split = make_split(names)
    assert split == make_split(names[::-1])
    validate_split(split, names)
    assert [len(split[k]) for k in ("train", "validation", "test")] == [800, 100, 100]
    assert [len(v) for v in split["train_subsets"].values()] == [80, 200, 800]
    split["test"][0] = split["train"][0]
    with pytest.raises(ValueError, match="exactly once"):
        validate_split(split, names)


def test_window_time_alignment_and_episode_boundaries():
    episode = sample_episode()
    np.testing.assert_array_equal(anchors(15, 4, 5), np.arange(3, 10))
    s = window(episode, 9, 4, 5)
    np.testing.assert_array_equal(
        s["history"]["eef_position"], episode["eef_position"][6:10]
    )
    np.testing.assert_array_equal(s["actions"], episode["actions"][9:14])
    np.testing.assert_array_equal(
        s["target_delta"][0], episode["eef_position"][14] - episode["eef_position"][9]
    )
    for t in (2, 10):
        with pytest.raises(IndexError):
            window(episode, t, 4, 5)
    assert len(anchors(5, 4, 5)) == 0


def test_future_state_does_not_change_inputs_or_baseline():
    episode = sample_episode()
    before = window(episode, 5, 4, 5)
    episode["eef_position"][6:] += 10000
    after = window(episode, 5, 4, 5)
    np.testing.assert_array_equal(
        before["history"]["eef_position"], after["history"]["eef_position"]
    )
    np.testing.assert_array_equal(before["actions"], after["actions"])
    for method in ("hold", "constant_velocity"):
        p0 = predict_positions(
            before["history"]["eef_position"], np.array([3]), 5, method
        )
        p1 = predict_positions(
            after["history"]["eef_position"], np.array([3]), 5, method
        )
        np.testing.assert_array_equal(p0, p1)


def test_constant_velocity_and_cm_units():
    p = np.arange(20)[:, None] * np.array([[0.01, 0, 0]])
    indices = np.array([3, 4, 5])
    np.testing.assert_allclose(
        predict_positions(p, indices, 5, "constant_velocity"), p[indices + 5]
    )
    np.testing.assert_allclose(position_error_cm(p[indices], p[indices + 5]), 5)
    np.testing.assert_allclose(
        position_error_cm(np.array([[0, 0, 0]]), np.array([[0.03, 0.04, 0]])), [5]
    )


def test_shared_position_scale_and_target_inverse():
    s = window(sample_episode(), 5, 4, 5)
    norm = normalize_window(
        s, {"position_center": [1, 2, 3], "position_scale": 2, "scalar_fields": {}}
    )
    np.testing.assert_allclose(
        norm["history"]["eef_position"] * 2 + [1, 2, 3], s["history"]["eef_position"]
    )
    np.testing.assert_allclose(norm["target_delta"] * 2, s["target_delta"])
    np.testing.assert_array_equal(norm["actions"], s["actions"])


def test_episode_bootstrap_constant_values():
    result = mean_interval([2, 2, 2])
    assert result == {"mean_cm": 2.0, "ci95_cm": [2.0, 2.0]}


def test_aggregation_weights_episodes_not_overlapping_windows():
    rows = []
    for episode, count, error in [("demo_0", 1, 1.0), ("demo_1", 100, 9.0)]:
        for model in ("hold", "constant_velocity"):
            rows.append(
                {
                    "episode": episode,
                    "windows": count,
                    "mean_error_cm": error,
                    "model": model,
                    "entity": "end_effector",
                    "horizon_steps": 1,
                    "segment": "all",
                }
            )
    result = summarize(rows)
    assert result[0]["mean_cm"] == 5
    assert result[2]["mean_cm"] == 0
