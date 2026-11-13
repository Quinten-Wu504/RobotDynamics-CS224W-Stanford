from collections import defaultdict

import numpy as np

from robot_dynamics.grail_data import HORIZONS


ENTITIES = ("robot_root", "object")


def displacement(prediction, arrays, positions, stats, method):
    if method == "hold":
        return np.zeros_like(arrays["raw_delta"])
    if method == "constant_velocity":
        t, k = arrays["anchor"], arrays["horizon"]
        return ((positions[t, :2] - positions[t - 1, :2]) * k[:, None, None]).reshape(
            -1, 6
        )
    return prediction.astype("float64") * stats["target_scale"]


def grouped_rows(delta, arrays, name, asset, stats):
    errors = (
        np.linalg.norm((delta - arrays["raw_delta"]).reshape(-1, 2, 3), axis=-1) * 100
    )
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite physical error")
    rows = []
    for horizon in HORIZONS:
        for j, entity in enumerate(ENTITIES):
            low = arrays["speed"][:, j] <= stats["motion_threshold_m_s"][j]
            for label, mask in (
                ("all", np.ones(len(low), bool)),
                ("low_motion", low),
                ("moving", ~low),
            ):
                mask = mask & (arrays["horizon"] == horizon)
                if mask.any():
                    rows.append(
                        {
                            "episode": name,
                            "asset": asset,
                            "entity": entity,
                            "horizon_steps": horizon,
                            "segment": label,
                            "windows": int(mask.sum()),
                            "mean_error_cm": float(errors[mask, j].mean()),
                        }
                    )
    return rows, errors


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["entity"], int(row["horizon_steps"]), row["segment"]].append(row)
    return [
        {
            "entity": entity,
            "horizon_steps": horizon,
            "segment": segment,
            "episodes": len(values),
            "assets": len({r["asset"] for r in values}),
            "windows": sum(int(r["windows"]) for r in values),
            "mean_cm": float(np.mean([float(r["mean_error_cm"]) for r in values])),
        }
        for (entity, horizon, segment), values in sorted(groups.items())
    ]


def asset_values(rows):
    episodes, membership = defaultdict(list), {}
    cells = defaultdict(set)
    for row in rows:
        if row["segment"] == "all":
            name = row["episode"]
            episodes[name].append(float(row["mean_error_cm"]))
            membership[name] = row["asset"]
            cells[name].add((row["entity"], int(row["horizon_steps"])))
    expected = {(entity, horizon) for entity in ENTITIES for horizon in HORIZONS}
    if (
        any(len(v) != 6 or cells[k] != expected for k, v in episodes.items())
        or not episodes
    ):
        raise ValueError("Missing or duplicate entity/horizon cells")
    assets = defaultdict(list)
    for name, values in episodes.items():
        assets[membership[name]].append(float(np.mean(values)))
    if any(len(v) != 2 for v in assets.values()):
        raise ValueError("Each asset must retain both motions")
    ordered = sorted(assets)
    return ordered, np.array([np.mean(assets[a]) for a in ordered])


def paired_interval(differences):
    values = np.asarray(differences)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError("Expected three seeds and paired asset columns")
    rng = np.random.default_rng(224)
    seeds = rng.integers(0, 3, (2000, 3))
    assets = rng.integers(0, values.shape[1], (2000, values.shape[1]))
    sampled = values[seeds[:, :, None], assets[:, None, :]].mean(axis=(1, 2))
    return {
        "mean_difference_cm": float(values.mean()),
        "seed_differences_cm": values.mean(1).tolist(),
        "ci95_seed_asset_cm": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "draws": 2000,
        "bootstrap_seed": 224,
    }
