import numpy as np
import pytest

from robot_dynamics.frame_transforms import (
    yaw_rotation,
    transform_sample,
    transform_vectors,
)
from robot_dynamics.learning_data import encode_sample
from robot_dynamics.windows import window


def sample():
    rng = np.random.default_rng(6)
    episode = {k: rng.normal(size=(20, 3)) for k in ("eef_position", "nut_position")}
    for k in ("eef_rotation", "nut_rotation"):
        episode[k] = np.stack([yaw_rotation(a) for a in rng.uniform(-3, 3, 20)])
    episode["gripper_qpos"] = rng.normal(size=(20, 2))
    episode["robot_joint_qpos"] = rng.normal(size=(20, 7))
    for k in (
        "peg1/position",
        "peg1/geom_position",
        "table/surface_position",
        "table/geom_position",
    ):
        episode[k] = rng.normal(size=3)
    for k in ("peg1/rotation", "peg1/geom_rotation", "table/geom_rotation"):
        episode[k] = yaw_rotation(rng.uniform(-3, 3))
    for k in ("peg1/half_size", "table/half_size"):
        episode[k] = rng.uniform(0.01, 0.2, 3)
    episode["table/normal"] = np.array([0.0, 0.0, 1.0])
    episode["actions"] = rng.uniform(-1, 1, (20, 7))
    episode["time"] = np.arange(20) * 0.05
    return window(episode, 5, 4, 10)


def test_frame_transform_roundtrip_and_gravity():
    raw = sample()
    r, t = yaw_rotation(1.2), np.array([0.2, -0.7, 0.3])
    changed = transform_sample(raw, r, t)
    restored = transform_sample(changed, r.T, -r.T @ t)
    for section in ("history", "fixtures"):
        for k in raw[section]:
            np.testing.assert_allclose(
                raw[section][k], restored[section][k], atol=1e-12
            )
    np.testing.assert_allclose(raw["actions"], restored["actions"], atol=1e-12)
    np.testing.assert_allclose(
        raw["target_delta"], restored["target_delta"], atol=1e-12
    )
    np.testing.assert_array_equal(transform_vectors([0, 0, -9.81], r), [0, 0, -9.81])
    velocity = np.diff(raw["history"]["eef_position"], axis=0) / 0.05
    np.testing.assert_allclose(
        np.diff(changed["history"]["eef_position"], axis=0) / 0.05,
        transform_vectors(velocity, r),
        atol=1e-12,
    )
    with pytest.raises(ValueError):
        transform_sample(raw, np.diag([-1.0, 1.0, 1.0]), t)


@pytest.mark.parametrize("kind", ["canonical_gru", "canonical_temporal_gnn"])
def test_full_normalized_model_is_equivariant(kind):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from robot_dynamics.learning_models import DynamicsPredictor

    torch.manual_seed(2)
    model = DynamicsPredictor(kind)
    raw = sample()
    stats = {
        "position_center": [0.41, -0.23, 0.82],
        "position_scale": 0.37,
        "scalar_fields": {},
    }
    s, c, _ = encode_sample(raw, stats, 10)
    pred = model(torch.tensor(s)[None], torch.tensor(c)[None])
    pred.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    for angle in (0.0, 0.7, -2.8):
        r, t = yaw_rotation(angle), np.array([0.3, -0.8, 0.1])
        transformed = transform_sample(raw, r, t)
        ss, cc, _ = encode_sample(transformed, stats, 10)
        result = (
            model(torch.tensor(ss)[None], torch.tensor(cc)[None])
            .detach()
            .numpy()
            .reshape(2, 3)
        )
        np.testing.assert_allclose(
            result,
            transform_vectors(pred.detach().numpy().reshape(2, 3), r),
            rtol=1e-4,
            atol=2e-6,
        )
