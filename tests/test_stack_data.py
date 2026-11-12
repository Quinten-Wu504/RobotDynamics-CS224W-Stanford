import h5py
import numpy as np

from robot_dynamics.stack_data import episode_arrays, fit_statistics, error_rows


def episode(cache, name, offset=0):
    g = cache.create_group(name)
    p = np.arange(20 * 3 * 3).reshape(20, 3, 3) * 0.001 + offset
    fields = {
        "position": p,
        "rotation": np.broadcast_to(np.eye(3), (20, 3, 3, 3)),
        "robot": np.arange(180).reshape(20, 9) * 0.01,
        "table_position": np.array([0.0, 0.0, 0.8]),
        "table_rotation": np.eye(3),
        "half_size": np.ones((4, 3)) * 0.02,
        "actions": np.arange(140).reshape(20, 7) * 0.001,
        "time": np.arange(20) * 0.05,
    }
    for k, v in fields.items():
        g[k] = v
    return g


def test_stack_causal_windows_and_three_targets(tmp_path):
    with h5py.File(tmp_path / "cache.h5", "w") as cache:
        g = episode(cache, "train")
        stats = fit_statistics(cache, ["train"])
        a = episode_arrays(g, stats)
        assert a["state"].shape == (16 + 12 + 7, 4, 148)
        t, k = a["anchor"][0], a["horizon"][0]
        np.testing.assert_allclose(
            a["target"][0].reshape(3, 3) * stats["scale"],
            g["position"][t + k] - g["position"][t],
            atol=1e-8,
        )
        np.testing.assert_array_equal(
            a["context"][0, :7], g["actions"][t].astype("float32")
        )
        assert not a["context"][0, 7:70].any()
        before = a["state"][0].copy()
        g["position"][t + 1 :] += 10
        after = episode_arrays(g, stats)
        np.testing.assert_array_equal(before, after["state"][0])
        assert not np.array_equal(a["target"][0], after["target"][0])
        for horizon in (1, 5, 10):
            assert a["anchor"][a["horizon"] == horizon].max() + horizon == 19


def test_stack_train_only_stats_and_cm_aggregation(tmp_path):
    with h5py.File(tmp_path / "cache.h5", "w") as cache:
        g = episode(cache, "train")
        stats = fit_statistics(cache, ["train"])
        episode(cache, "validation", offset=1000)
        assert fit_statistics(cache, ["train"]) == stats
        a = episode_arrays(g, stats)
        a.update(ids=["train"], episode=np.zeros(len(a["state"]), dtype=int))
        pred = a["target"].copy().reshape(-1, 3, 3)
        pred[:, :, 0] += 0.01 / stats["scale"]
        rows = error_rows(pred.reshape(-1, 9), a, stats["scale"])
        assert len(rows) == 9
        np.testing.assert_allclose([r["mean_error_cm"] for r in rows], 1.0, atol=1e-6)
