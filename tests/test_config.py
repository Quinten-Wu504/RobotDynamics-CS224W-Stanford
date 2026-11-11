from pathlib import Path

import pytest
import yaml

from robot_dynamics.config import horizon_steps, load_config, unresolved_audit


def test_horizons_use_measured_frequency():
    assert horizon_steps([0.25, 0.5], 20) == [5, 10]
    assert horizon_steps([0.25, 0.5], 40) == [10, 20]
    with pytest.raises(ValueError, match="whole number"):
        horizon_steps([0.25], 30)


@pytest.mark.parametrize("frequency", [0, -20, float("nan"), float("inf")])
def test_invalid_frequency(frequency):
    with pytest.raises(ValueError):
        horizon_steps([0.25], frequency)


def test_unverified_data_cannot_be_ready():
    config = load_config("configs/square.yaml")
    config["control_frequency_hz"] = None
    config["audit"] = {key: False for key in config["audit"]}
    assert set(unresolved_audit(config)) == {
        "observation_fields",
        "controller_semantics",
        "time_alignment",
        "fixture_world_poses",
        "control_frequency_hz",
    }


def test_reject_window_level_split(tmp_path):
    config = yaml.safe_load(Path("configs/square.yaml").read_text())
    config["split"]["unit"] = "window"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="episode"):
        load_config(path)


@pytest.mark.parametrize("flags", [{}, {"observation_fields": "false"}])
def test_incomplete_audit_flags_do_not_pass(flags):
    config = load_config("configs/square.yaml")
    config["control_frequency_hz"] = 20
    config["audit"] = flags
    assert "observation_fields" in unresolved_audit(config)
