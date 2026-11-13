import numpy as np
import pytest

from robot_dynamics.grail_results import (
    displacement,
    grouped_rows,
    asset_values,
    paired_interval,
)


def test_causal_baselines_units_and_motion_groups():
    positions = np.zeros((20, 3, 3))
    positions[:, :2, 0] = np.arange(20)[:, None] * 0.01
    arrays = {
        "anchor": np.array([8, 8, 8]),
        "horizon": np.array([1, 6, 12]),
        "raw_delta": np.array([[0.01 * k, 0, 0, 0.01 * k, 0, 0] for k in (1, 6, 12)]),
        "speed": np.array([[0, 1]] * 3),
    }
    positions = np.concatenate([positions, positions[-1:] + 0.01])
    stats = {"target_scale": [1] * 6, "motion_threshold_m_s": [0.1, 0.1]}
    delta = displacement(None, arrays, positions, stats, "constant_velocity")
    np.testing.assert_allclose(delta, arrays["raw_delta"], atol=1e-15)
    positions[9:] += 100
    np.testing.assert_array_equal(
        delta, displacement(None, arrays, positions, stats, "constant_velocity")
    )
    hold = displacement(None, arrays, positions, stats, "hold")
    rows, errors = grouped_rows(hold, arrays, "motion", "asset", stats)
    np.testing.assert_allclose(errors[:, 0], [1, 6, 12])
    assert len(rows) == 12
    assert all(
        r["entity"] == "robot_root" for r in rows if r["segment"] == "low_motion"
    )


def test_asset_pairs_and_bootstrap():
    rows = [
        {
            "episode": f"{asset}_{motion}",
            "asset": asset,
            "entity": entity,
            "horizon_steps": k,
            "segment": "all",
            "mean_error_cm": value,
        }
        for asset, value in (("a", 1), ("b", 3))
        for motion in (0, 1)
        for entity in ("robot_root", "object")
        for k in (1, 6, 12)
    ]
    names, values = asset_values(rows)
    assert names == ["a", "b"]
    np.testing.assert_array_equal(values, [1, 3])
    with pytest.raises(ValueError, match="both motions"):
        asset_values([r for r in rows if r["episode"] != "a_1"])
    interval = paired_interval(np.full((3, 18), -0.25))
    assert interval["ci95_seed_asset_cm"] == [-0.25, -0.25]
