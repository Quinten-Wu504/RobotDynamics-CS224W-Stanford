import numpy as np


HISTORY = 8
HORIZONS = (1, 6, 12)
FEATURES = 73


def node_features(group):
    pos = group["position"][:].astype("float64")
    n = len(pos)
    nodes = np.zeros((n, 3, FEATURES), dtype="float64")
    nodes[..., :3] = pos
    nodes[..., 3:12] = group["rotation"][:]
    # Frame zero has no predecessor and is excluded from every input window.
    nodes[1:, :, 12:15] = np.diff(pos, axis=0) * 25
    nodes[:, 0, 15:58] = group["joints"][:]
    nodes[:, 2, 58:61] = group["table_size"][:]
    nodes[..., 61:64] = np.eye(3)[None]
    nodes[..., 64:73] = (pos[:, None, :, :] - pos[:, :, None, :]).reshape(n, 3, 9)
    return nodes


def validate_manifest(manifest, cache):
    splits = manifest["splits"]
    all_ids = sum(splits.values(), [])
    if (
        set(splits) != {"train", "validation", "test"}
        or len(set(all_ids)) != len(all_ids)
        or set(all_ids) != set(cache)
    ):
        raise ValueError("Motion splits are incomplete or overlap")
    assets = []
    for split, ids in splits.items():
        actual = {manifest["audit"][x]["asset"] for x in ids}
        if actual != set(manifest["split_assets"][split]):
            raise ValueError("Asset membership mismatch")
        assets.extend(actual)
    if len(set(assets)) != len(assets):
        raise ValueError("An object asset crosses splits")
    source_splits = {}
    for split, ids in splits.items():
        for name in ids:
            source = manifest["audit"][name]["source_id"]
            if source in source_splits and source_splits[source] != split:
                raise ValueError("A source motion crosses splits")
            source_splits[source] = split
    if not cache.attrs["complete"] or cache.attrs["fps"] != 25:
        raise ValueError("Invalid cache")


def fit_statistics(cache, ids):
    features, targets, speeds = [], [], []
    for name in ids:
        group = cache[name]
        nodes = node_features(group)
        features.append(nodes[1:])
        pos = group["position"][:].astype("float64")[:, :2]
        speeds.append(np.linalg.norm(np.diff(pos, axis=0) * 25, axis=-1))
        for horizon in HORIZONS:
            anchors = np.arange(HISTORY, len(pos) - horizon)
            targets.append((pos[anchors + horizon] - pos[anchors]).reshape(-1, 6))
    flat = np.concatenate(features).reshape(-1, FEATURES)
    scale = np.maximum(np.std(np.concatenate(targets), axis=0), 1e-4)
    return {
        "train_ids": ids,
        "mean": flat.mean(0).tolist(),
        "std": np.maximum(flat.std(0), 1e-4).tolist(),
        "target_scale": scale.tolist(),
        "motion_threshold_m_s": np.maximum(
            np.quantile(np.concatenate(speeds), 0.25, axis=0), 1e-6
        ).tolist(),
    }


def make_arrays(cache, ids, stats):
    states, contexts, targets, episode, horizons, anchor_rows, raw_delta, speeds = (
        [] for _ in range(8)
    )
    for index, name in enumerate(ids):
        group = cache[name]
        nodes = node_features(group)
        normalized = ((nodes - stats["mean"]) / stats["std"]).astype("float32")
        pos = group["position"][:].astype("float64")[:, :2]
        for column, horizon in enumerate(HORIZONS):
            anchors = np.arange(HISTORY, len(pos) - horizon)
            delta = (pos[anchors + horizon] - pos[anchors]).reshape(-1, 6)
            states.append(normalized[anchors[:, None] + np.arange(1 - HISTORY, 1)])
            contexts.append(
                np.broadcast_to(np.eye(3, dtype="float32")[column], (len(anchors), 3))
            )
            targets.append((delta / stats["target_scale"]).astype("float32"))
            raw_delta.append(delta)
            speeds.append(np.linalg.norm(pos[anchors] - pos[anchors - 1], axis=-1) * 25)
            episode.append(np.full(len(anchors), index))
            horizons.append(np.full(len(anchors), horizon))
            anchor_rows.append(anchors)
    return {
        "state": np.concatenate(states),
        "context": np.concatenate(contexts),
        "target": np.concatenate(targets),
        "raw_delta": np.concatenate(raw_delta),
        "episode": np.concatenate(episode),
        "horizon": np.concatenate(horizons),
        "anchor": np.concatenate(anchor_rows),
        "speed": np.concatenate(speeds),
        "ids": ids,
    }


def error_rows(prediction, arrays, stats):
    delta = prediction.astype("float64") * stats["target_scale"]
    errors = (
        np.linalg.norm((delta - arrays["raw_delta"]).reshape(-1, 2, 3), axis=-1) * 100
    )
    rows = []
    for i, name in enumerate(arrays["ids"]):
        for horizon in HORIZONS:
            mask = (arrays["episode"] == i) & (arrays["horizon"] == horizon)
            for entity, label in enumerate(("robot_root", "object")):
                rows.append(
                    {
                        "episode": name,
                        "entity": label,
                        "horizon_steps": horizon,
                        "windows": int(mask.sum()),
                        "mean_error_cm": float(errors[mask, entity].mean()),
                    }
                )
    return rows
