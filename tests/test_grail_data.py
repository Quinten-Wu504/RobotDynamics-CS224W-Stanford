import h5py
import numpy as np
import pytest

from robot_dynamics.grail_data import (
    fit_statistics,
    make_arrays,
    error_rows,
    validate_manifest,
)


def fixture_cache(path):
    with h5py.File(path, "w") as cache:
        cache.attrs.update(complete=True, fps=25)
        for name in ("train", "validation", "test"):
            g = cache.create_group(name)
            pos = np.zeros((30, 3, 3), dtype="float32")
            pos[:, 0, 0] = np.arange(30) / 25
            pos[:, 1, 1] = np.arange(30) ** 2 / 625
            g["position"] = pos
            g["rotation"] = np.broadcast_to(np.eye(3).reshape(9), (30, 3, 9))
            g["joints"] = np.zeros((30, 43))
            g["table_size"] = [2, 0.6, 0.04]


def test_causal_history_targets_and_units(tmp_path):
    path = tmp_path / "cache.h5"
    fixture_cache(path)
    with h5py.File(path, "r+") as cache:
        stats = fit_statistics(cache, ["train"])
        arrays = make_arrays(cache, ["train"], stats)
        state = arrays["state"].copy()
        # First anchor is 8: history frames 1..8, velocities use frames 0..8.
        raw = state[0] * stats["std"] + stats["mean"]
        np.testing.assert_allclose(raw[:, 0, 0], np.arange(1, 9) / 25, atol=1e-7)
        np.testing.assert_allclose(raw[:, 0, 12], 1, atol=2e-6)
        np.testing.assert_allclose(arrays["raw_delta"][0, :3], [0.04, 0, 0], atol=1e-7)
        cache["train/position"][9:] += 3
        changed = make_arrays(cache, ["train"], stats)
        np.testing.assert_array_equal(changed["state"][0], state[0])
        assert not np.array_equal(changed["target"][0], arrays["target"][0])
        cache["test/position"][:] += 1000
        assert fit_statistics(cache, ["validation"]) == {
            **stats,
            "train_ids": ["validation"],
        }
    prediction = arrays["target"].astype("float64")
    prediction[:, 0] += 0.01 / stats["target_scale"][0]
    rows = error_rows(prediction, arrays, stats)
    assert len(rows) == 6
    assert all(
        abs(r["mean_error_cm"] - 1) < 1e-5 for r in rows if r["entity"] == "robot_root"
    )


def test_asset_leakage_is_rejected(tmp_path):
    path = tmp_path / "cache.h5"
    fixture_cache(path)
    manifest = {
        "splits": {s: [s] for s in ("train", "validation", "test")},
        "split_assets": {s: [s] for s in ("train", "validation", "test")},
        "audit": {
            s: {"asset": s, "source_id": s} for s in ("train", "validation", "test")
        },
    }
    with h5py.File(path, "r") as cache:
        validate_manifest(manifest, cache)
        manifest["split_assets"]["test"] = ["train"]
        manifest["audit"]["test"]["asset"] = "train"
        with pytest.raises(ValueError, match="crosses splits"):
            validate_manifest(manifest, cache)
