import numpy as np


HISTORY_FIELDS = (
    "eef_position",
    "nut_position",
    "eef_rotation",
    "nut_rotation",
    "gripper_qpos",
    "robot_joint_qpos",
)
FIXTURE_FIELDS = (
    "peg1/position",
    "peg1/rotation",
    "peg1/geom_position",
    "peg1/geom_rotation",
    "peg1/half_size",
    "table/surface_position",
    "table/normal",
    "table/geom_position",
    "table/geom_rotation",
    "table/half_size",
)


def anchors(length, history, horizon):
    if history < 1 or horizon < 1:
        raise ValueError("History and horizon must be positive")
    return np.arange(history - 1, max(history - 1, length - horizon))


def read_episode(group):
    return {
        key: group[key][:]
        for key in HISTORY_FIELDS + FIXTURE_FIELDS + ("actions", "time")
    }


def window(episode, t, history, horizon):
    if not history - 1 <= t < len(episode["time"]) - horizon:
        raise IndexError("Window crosses episode boundary")
    past = slice(t - history + 1, t + 1)
    positions = np.stack([episode[k] for k in ("eef_position", "nut_position")], axis=1)
    return {
        "history": {k: episode[k][past].copy() for k in HISTORY_FIELDS},
        "fixtures": {k: episode[k].copy() for k in FIXTURE_FIELDS},
        "actions": episode["actions"][t : t + horizon].copy(),
        "target_delta": positions[t + horizon] - positions[t],
        "anchor": t,
        "target_index": t + horizon,
    }


def normalize_window(sample, stats):
    center, scale = np.asarray(stats["position_center"]), stats["position_scale"]
    result = {**sample, "history": {}, "fixtures": {}}
    for key, values in sample["history"].items():
        if key.endswith("_position"):
            values = (values - center) / scale
        elif key in stats["scalar_fields"]:
            s = stats["scalar_fields"][key]
            values = (values - s["mean"]) / s["scale"]
        result["history"][key] = values.copy()
    for key, values in sample["fixtures"].items():
        if key.endswith("position"):
            values = (values - center) / scale
        elif key.endswith("half_size"):
            values = values / scale
        result["fixtures"][key] = values.copy()
    result["actions"] = sample["actions"].copy()
    result["target_delta"] = sample["target_delta"] / scale
    return result


def predict_positions(positions, indices, horizon, method):
    current = positions[indices]
    if method == "hold":
        return current.copy()
    if method == "constant_velocity":
        if np.any(indices < 1):
            raise ValueError("Constant velocity needs a preceding observed frame")
        return current + horizon * (current - positions[indices - 1])
    raise ValueError(f"Unknown baseline: {method}")


def position_error_cm(predicted, target):
    return 100 * np.linalg.norm(predicted - target, axis=-1)
