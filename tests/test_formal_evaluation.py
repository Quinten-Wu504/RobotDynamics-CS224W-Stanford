import json
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from robot_dynamics.evaluate_formal import grouped_errors, main  # noqa: E402
from robot_dynamics.formal import matrix  # noqa: E402


def test_test_access_requires_completed_selection_seal(tmp_path, monkeypatch):
    (tmp_path / "matrix.json").write_text(
        json.dumps(matrix("configs/formal_protocol.yaml"))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_formal",
            "not_opened_d0.hdf5",
            "--d1-cache",
            "not_opened_d1.hdf5",
            "--runs",
            str(tmp_path),
        ],
    )
    with pytest.raises(FileNotFoundError) as error:
        main()
    assert "selection_seal.json" in str(error.value)


def test_group_membership_uses_only_anchor_and_past():
    positions = np.zeros((10, 2, 3))
    positions[3, 0, 0] = 0.03
    positions[3, 1, 0] = 0.04
    arrays = {"anchor": np.array([3, 3, 3]), "horizon": np.array([1, 5, 10])}
    errors = np.ones((3, 2))
    before = grouped_errors(errors, "episode", arrays, positions, [0.2, 0.2], 0.05)
    positions[4:] = 999
    after = grouped_errors(errors, "episode", arrays, positions, [0.2, 0.2], 0.05)
    assert before == after
    assert all(r["mean_error_cm"] == 1 for r in before)
    assert all(r["segment"] != "far_object" for r in before)
