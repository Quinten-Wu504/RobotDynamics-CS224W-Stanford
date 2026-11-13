import pytest

torch = pytest.importorskip("torch")

from robot_dynamics.grail_models import GrailPredictor, MODELS  # noqa: E402


def test_graph_permutation_batch_and_messages():
    torch.manual_seed(224)
    graph = GrailPredictor("temporal_gnn").eval()
    closed = GrailPredictor("temporal_no_messages").eval()
    closed.load_state_dict(graph.state_dict())
    state, context = torch.randn(2, 8, 3, 73), torch.eye(3)[:2]
    permutation = torch.tensor([2, 0, 1])
    expected = graph.forward_nodes(state, context)
    actual = graph.forward_nodes(state[:, :, permutation], context)[
        :, torch.argsort(permutation)
    ]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(graph(state[:1], context[:1]), graph(state, context)[:1])
    assert not torch.allclose(graph(state, context), closed(state, context))
    modified = state.clone()
    modified[:, :, 1] += 2
    torch.testing.assert_close(
        closed.forward_nodes(state, context)[:, 0],
        closed.forward_nodes(modified, context)[:, 0],
    )
    assert not torch.allclose(
        graph.forward_nodes(state, context)[:, 0],
        graph.forward_nodes(modified, context)[:, 0],
    )


def test_matched_capacity_and_gradients():
    counts = []
    for kind in MODELS:
        model = GrailPredictor(kind)
        state, context = torch.randn(2, 8, 3, 73), torch.eye(3)[:2]
        out = model(state, context)
        assert out.shape == (2, 6)
        out.square().mean().backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        counts.append(sum(p.numel() for p in model.parameters()))
    assert max(counts) / min(counts) < 1.15
