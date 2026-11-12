import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from robot_dynamics.stack_models import StackPredictor  # noqa: E402


@pytest.mark.parametrize(
    "kind", ["history_mlp", "gru", "temporal_gnn", "temporal_no_messages"]
)
def test_stack_shapes_batching_and_gradients(kind):
    torch.manual_seed(4)
    state, context = torch.randn(3, 4, 148), torch.randn(3, 73)
    model = StackPredictor(kind)
    prediction = model(state, context)
    assert prediction.shape == (3, 9)
    individual = torch.cat([model(s[None], c[None]) for s, c in zip(state, context)])
    torch.testing.assert_close(prediction, individual, rtol=1e-5, atol=1e-6)
    prediction.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_stack_permutation_and_message_ablation():
    torch.manual_seed(9)
    model = StackPredictor("temporal_gnn").graph
    nodes, context = torch.randn(2, 4, 4, 37), torch.randn(2, 73)
    before = model.forward_nodes(nodes, context)
    perm = torch.tensor([3, 1, 0, 2])
    after = model.forward_nodes(
        nodes[:, :, perm], context, torch.argsort(perm)[model.edges], perm
    )
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
    changed = nodes.clone()
    changed[:, :, 2, 3:12] += 2
    assert (
        model.forward_nodes(changed, context)[:, :3] - before[:, :3]
    ).abs().max() > 1e-6
    model.messages = False
    torch.testing.assert_close(
        model.forward_nodes(nodes, context)[:, :3],
        model.forward_nodes(changed, context)[:, :3],
        rtol=0,
        atol=0,
    )
