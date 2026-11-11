import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from robot_dynamics.graph_models import (  # noqa: E402
    SceneGraphPredictor,
    scene_nodes,
    scene_edges,
    batch_topology,
)


def inputs():
    torch.manual_seed(7)
    return torch.randn(3, 4, 42), torch.randn(3, 121)


def test_graph_packing_preserves_physical_fields():
    state, context = inputs()
    nodes = scene_nodes(state, context)
    torch.testing.assert_close(nodes[:, :, 0, 30:39], state[:, :, 24:33])
    torch.testing.assert_close(nodes[:, :, 1, 3:12], state[:, :, 15:24])
    torch.testing.assert_close(nodes[:, 0, 2, :27], context[:, :27])
    torch.testing.assert_close(nodes[:, 0, 3, 12:27], context[:, 33:48])
    torch.testing.assert_close(nodes[:, :, 0, 43:46], state[:, :, 33:36])
    torch.testing.assert_close(nodes[:, :, 1, 46:49], state[:, :, 36:39])
    torch.testing.assert_close(nodes[:, :, 0, 49:52], state[:, :, 39:42])


def test_pyg_batch_has_no_cross_graph_edges():
    edges = scene_edges()
    batch = batch_topology(edges, 3)
    src, dst = batch.edge_index
    assert torch.equal(batch.batch[src], batch.batch[dst])
    for i in range(3):
        assert torch.equal(batch.edge_index[:, i * 10 : (i + 1) * 10], edges + 4 * i)


@pytest.mark.parametrize(
    "kind", ["gnn", "temporal_gnn", "temporal_no_messages", "temporal_full_edges"]
)
def test_permutation_batching_and_gradients(kind):
    state, context = inputs()
    model = SceneGraphPredictor(kind)
    result = model(state, context)
    individual = torch.cat([model(s[None], c[None]) for s, c in zip(state, context)])
    torch.testing.assert_close(result, individual, rtol=1e-5, atol=1e-6)
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    shuffled = model.forward_graph(
        scene_nodes(state, context)[:, :, permutation],
        context,
        inverse[model.edges],
        permutation,
    )
    torch.testing.assert_close(result, shuffled, rtol=1e-5, atol=1e-6)
    result.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_history_and_neighbor_dependence():
    state, context = inputs()
    single = SceneGraphPredictor("gnn")
    temporal = SceneGraphPredictor("temporal_gnn")
    changed = state.clone()
    changed[:, :-1] += 2
    torch.testing.assert_close(
        single(state, context), single(changed, context), rtol=0, atol=0
    )
    assert not torch.allclose(temporal(state, context), temporal(changed, context))
    nodes = scene_nodes(state, context)
    changed_nodes = nodes.clone()
    changed_nodes[:, :, 1, 3:12] += 2
    before = temporal.forward_graph(nodes, context)[:, :3]
    after = temporal.forward_graph(changed_nodes, context)[:, :3]
    assert (before - after).abs().max() > 1e-6
    temporal.messages = False
    torch.testing.assert_close(
        temporal.forward_graph(nodes, context)[:, :3],
        temporal.forward_graph(changed_nodes, context)[:, :3],
        rtol=0,
        atol=0,
    )
