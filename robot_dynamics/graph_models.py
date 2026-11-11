import torch
from torch import nn
from torch_geometric.data import Batch, Data


GRAPH_MODELS = ("gnn", "temporal_gnn", "temporal_no_messages", "temporal_full_edges")


def scene_nodes(state, context):
    """Pack the existing normalized inputs; node order is eef, nut, peg, table."""
    b, t, _ = state.shape
    x = state.new_zeros(b, t, 4, 52)
    x[:, :, 0, :3] = state[:, :, :3]
    x[:, :, 1, :3] = state[:, :, 3:6]
    x[:, :, 0, 3:12] = state[:, :, 6:15]
    x[:, :, 1, 3:12] = state[:, :, 15:24]
    x[:, :, 0, 30:39] = state[:, :, 24:33]
    x[:, :, 0, 43:46] = state[:, :, 33:36]
    x[:, :, 0, 49:52] = state[:, :, 39:42]
    x[:, :, 1, 46:49] = state[:, :, 36:39]
    x[:, :, 2, :27] = context[:, None, :27]
    x[:, :, 3, :3] = context[:, None, 27:30]
    x[:, :, 3, 27:30] = context[:, None, 30:33]
    x[:, :, 3, 12:27] = context[:, None, 33:48]
    x[:, :, :, 39:43] = torch.eye(4, device=state.device, dtype=state.dtype)
    return x


def scene_edges(full=False):
    # Both directions; the default omits the static peg-table pair.
    return torch.tensor(
        [
            (i, j)
            for i in range(4)
            for j in range(4)
            if i != j and (full or {i, j} != {2, 3})
        ],
        dtype=torch.long,
    ).T.contiguous()


def batch_topology(edge_index, count):
    graph = Data(num_nodes=4, edge_index=edge_index)
    return Batch.from_data_list([graph] * count)


def mlp(input_dim, width, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, width), nn.ReLU(), nn.Linear(width, output_dim)
    )


class SceneGraphPredictor(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in GRAPH_MODELS:
            raise ValueError(kind)
        self.temporal = kind != "gnn"
        self.messages = kind != "temporal_no_messages"
        self.register_buffer("edges", scene_edges(kind == "temporal_full_edges"))
        self.encoder = mlp(52 + 73, 64, 64)
        self.message = mlp(64 * 2 + 3, 64, 64)
        self.update = mlp(128, 64, 64)
        if self.temporal:
            self.recurrent = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64 + 73, 192),
            nn.ReLU(),
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, 3),
        )
        self._topologies = {}

    def encode_graph(self, nodes, context, edges=None):
        if not self.temporal:
            nodes = nodes[:, -1:]
        b, t, n, _ = nodes.shape
        actions = context[:, 48:]
        encoded = self.encoder(
            torch.cat([nodes, actions[:, None, None].expand(b, t, n, -1)], dim=-1)
        )
        h = encoded.reshape(-1, 64)
        pos = nodes[..., :3].reshape(-1, 3)
        if edges is None:
            key = (b * t, str(nodes.device))
            if key not in self._topologies:
                self._topologies[key] = batch_topology(self.edges, b * t)
            topology = self._topologies[key]
        else:
            topology = batch_topology(edges, b * t)
        src, dst = topology.edge_index
        degree = torch.bincount(dst, minlength=len(h)).clamp_min(1).to(h.dtype)[:, None]
        for _ in range(2):
            message = self.message(torch.cat([h[src], h[dst], pos[src] - pos[dst]], -1))
            # Keeping this branch preserves parameter count and message compute.
            message = message * float(self.messages)
            pooled = torch.zeros_like(h).index_add(0, dst, message) / degree
            h = h + self.update(torch.cat([h, pooled], -1))
        h = h.reshape(b, t, n, 64)
        if self.temporal:
            _, last = self.recurrent(h.permute(0, 2, 1, 3).reshape(b * n, t, 64))
            h = last[-1].reshape(b, n, 64)
        else:
            h = h[:, -1]
        return h, actions

    def encode(self, state, context):
        return self.encode_graph(scene_nodes(state, context), context)[0]

    def forward_graph(self, nodes, context, edges=None, node_ids=None):
        h, actions = self.encode_graph(nodes, context, edges)
        b, n, _ = h.shape
        output = self.head(torch.cat([h, actions[:, None].expand(b, n, -1)], -1))
        if node_ids is not None:
            output = output[:, torch.argsort(node_ids)]
        return output[:, :2].reshape(b, 6)

    def forward(self, state, context):
        return self.forward_graph(scene_nodes(state, context), context)
