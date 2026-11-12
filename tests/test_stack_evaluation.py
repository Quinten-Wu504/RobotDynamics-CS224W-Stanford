import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from robot_dynamics.evaluate_stack import physical_errors, grouped_rows  # noqa: E402


def test_raw_physical_targets_and_causal_baselines():
    pos = np.arange(20)[:, None, None] * np.ones((1, 3, 3)) * 0.01
    arrays = {
        "anchor": np.array([3, 3, 3]),
        "horizon": np.array([1, 5, 10]),
        "speed": np.ones((3, 3)),
    }
    cv = physical_errors(None, arrays, pos, 1, "constant_velocity")
    np.testing.assert_allclose(cv, 0, atol=1e-12)
    hold = physical_errors(None, arrays, pos, 1, "hold")
    np.testing.assert_allclose(hold[:, 0], np.sqrt(3) * np.array([1, 5, 10]))
    prediction = (
        pos[arrays["anchor"] + arrays["horizon"]] - pos[arrays["anchor"]]
    ).reshape(3, 9) / 0.2
    np.testing.assert_allclose(
        physical_errors(prediction, arrays, pos, 0.2), 0, atol=1e-12
    )
    rows = grouped_rows(hold, arrays, "episode", [0, 2, 0])
    assert len([r for r in rows if r["segment"] == "all"]) == 9
    assert {r["entity"] for r in rows if r["segment"] == "low_motion"} == {"cubeA"}
    assert all(r["windows"] == 1 for r in rows)
