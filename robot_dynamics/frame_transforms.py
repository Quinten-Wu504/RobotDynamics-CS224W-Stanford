import numpy as np


def yaw_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def transform_vectors(values, rotation):
    """Linear/angular velocities, displacement, gravity, and normals are vectors."""
    return np.asarray(values) @ rotation.T


def transform_sample(sample, rotation, translation):
    if not (
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        and np.isclose(np.linalg.det(rotation), 1)
        and np.allclose(rotation[:, 2], [0, 0, 1], atol=1e-10)
    ):
        raise ValueError("Expected a proper gravity-preserving yaw rotation")
    result = {**sample, "history": {}, "fixtures": {}}
    for section in ("history", "fixtures"):
        for key, value in sample[section].items():
            if key.endswith("position"):
                value = transform_vectors(value, rotation) + translation
            elif key.endswith("rotation"):
                value = rotation @ value
            elif key.endswith("normal"):
                value = transform_vectors(value, rotation)
            result[section][key] = np.array(value, copy=True)
    actions = sample["actions"].copy()
    actions[:, :3] = transform_vectors(actions[:, :3], rotation)
    actions[:, 3:6] = transform_vectors(actions[:, 3:6], rotation)
    result["actions"] = actions
    result["target_delta"] = transform_vectors(sample["target_delta"], rotation)
    return result
