import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from robot_dynamics.jepa import (  # noqa: E402
    JointEmbedding,
    graph_backbone,
    encode,
    future_inputs,
    physical_targets,
    representation_stats,
)
from robot_dynamics.learning_models import DynamicsPredictor  # noqa: E402
from robot_dynamics.learning_data import encode_sample  # noqa: E402
from robot_dynamics.windows import window  # noqa: E402


def test_future_target_cannot_change_context_or_see_later_states():
    rng = np.random.default_rng(7)
    ep = {k: rng.normal(size=(20, 3)) for k in ("eef_position", "nut_position")}
    for k in ("eef_rotation", "nut_rotation"):
        ep[k] = np.broadcast_to(np.eye(3), (20, 3, 3)).copy()
    ep.update(
        gripper_qpos=rng.normal(size=(20, 2)),
        robot_joint_qpos=rng.normal(size=(20, 7)),
        time=np.arange(20),
        actions=rng.normal(size=(20, 7)),
    )
    for k in (
        "peg1/position",
        "peg1/geom_position",
        "peg1/half_size",
        "table/surface_position",
        "table/normal",
        "table/geom_position",
        "table/half_size",
    ):
        ep[k] = np.ones(3)
    for k in ("peg1/rotation", "peg1/geom_rotation", "table/geom_rotation"):
        ep[k] = np.eye(3)
    stats = {"position_center": [0, 0, 0], "position_scale": 1.0, "scalar_fields": {}}
    online = encode_sample(window(ep, 5, 4, 5), stats, 5)
    future = future_inputs(ep, 10, stats)
    assert np.count_nonzero(future[1][48:]) == 0
    ep["nut_position"][6:11] += 20
    modified = encode_sample(window(ep, 5, 4, 5), stats, 5)
    np.testing.assert_array_equal(online[0], modified[0])
    np.testing.assert_array_equal(online[1], modified[1])
    assert not np.array_equal(future[0], future_inputs(ep, 10, stats)[0])
    target = future_inputs(ep, 10, stats)
    ep["nut_position"][11:] += 999
    ep["actions"] += 999
    after = future_inputs(ep, 10, stats)
    np.testing.assert_array_equal(target[0], after[0])
    np.testing.assert_array_equal(target[1], after[1])


@pytest.mark.parametrize("kind", ["temporal_gnn", "canonical_temporal_gnn"])
def test_stop_gradient_ema_and_physical_target_shapes(kind):
    torch.manual_seed(0)
    model = DynamicsPredictor(kind)
    auxiliary = JointEmbedding(model)
    state = torch.randn(16, 4, 42)
    context = torch.randn(16, 121)
    context[:, 3:12] = torch.eye(3).flatten()
    future = state + torch.randn_like(state) * 0.1
    target_context = context.clone()
    target_context[:, 48:] = 0
    loss, _ = auxiliary.loss(model, state, context, future, target_context)
    loss.backward()
    assert all(p.grad is None for p in auxiliary.target.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in graph_backbone(model).encoder.parameters()
    )
    assert all(p.grad is None for p in graph_backbone(model).head.parameters())
    assert all(
        r["effective_rank"] > 1
        for r in representation_stats(encode(model, state, context))
    )
    before = {k: p.clone() for k, p in auxiliary.target.named_parameters()}
    with torch.no_grad():
        for p in graph_backbone(model).parameters():
            p.add_(0.1)
    auxiliary.update_target(model, 0.9)
    online = dict(graph_backbone(model).named_parameters())
    for name, p in auxiliary.target.named_parameters():
        torch.testing.assert_close(p, before[name] * 0.9 + online[name] * 0.1)
    targets, mask = physical_targets(model, future, target_context)
    assert targets.shape == (16, 2, 84) and mask.sum() == 132


def test_representation_screen_detects_constant_node_latents():
    collapsed = representation_stats(torch.zeros(32, 4, 64))
    assert all(r["mean_std"] == 0 and r["effective_rank"] < 2 for r in collapsed)
