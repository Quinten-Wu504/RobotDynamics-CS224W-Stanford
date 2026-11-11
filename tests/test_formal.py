import json
from pathlib import Path

import pytest
import yaml

from robot_dynamics.formal import matrix, validate_completed
from robot_dynamics.data_audit import file_sha256


def test_formal_matrix_has_all_seeds_fractions_and_controls():
    plan = matrix("configs/formal_protocol.yaml")
    rows = plan["runs"]
    assert len(rows) == 60
    assert sum(r["role"] == "core" for r in rows) == 36
    assert sum(r["role"] == "non_graph_control" for r in rows) == 18
    assert sum(r["role"] == "state_control" for r in rows) == 6
    assert sum(1 + (r["method"] != "supervised") for r in rows) == 84
    for r in rows:
        assert r["config"]["seed"] in (0, 1, 2)
        assert r["config"]["horizons"] == [1, 5, 10]
        assert r["config"]["formal_protocol_sha256"] == plan["protocol_sha256"]
        if r["method"] == "state":
            assert r["config"]["train_episodes"] == 80
    for count in (80, 200, 800):
        for seed in (0, 1, 2):
            assert (
                sum(
                    r["role"] == "core"
                    and r["config"]["train_episodes"] == count
                    and r["config"]["seed"] == seed
                    for r in rows
                )
                == 4
            )


def test_duplicate_configuration_is_rejected(tmp_path):
    protocol = yaml.safe_load(Path("configs/formal_protocol.yaml").read_text())
    protocol["seeds"] = [0, 0]
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol))
    with pytest.raises(ValueError, match="Duplicate"):
        matrix(path)


def test_incomplete_or_changed_checkpoint_cannot_be_sealed(tmp_path):
    run = matrix("configs/formal_protocol.yaml")["runs"][0]
    (tmp_path / "best.pt").write_bytes(b"checkpoint identity fixture")
    (tmp_path / "normalization.json").write_text("{}")
    metrics = {
        "model": run["model"],
        "method": run["method"],
        "config": run["config"],
        "logs": [{"epoch": 39}],
        "best_checkpoint_sha256": file_sha256(tmp_path / "best.pt"),
        "identity": {
            "normalization_sha256": file_sha256(tmp_path / "normalization.json")
        },
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="Incomplete"):
        validate_completed(run, tmp_path)
    metrics["logs"][0]["epoch"] = 40
    path.write_text(json.dumps(metrics))
    validate_completed(run, tmp_path)
    (tmp_path / "best.pt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint changed"):
        validate_completed(run, tmp_path)
