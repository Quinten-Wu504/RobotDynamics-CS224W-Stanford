import torch
from torch import nn

from robot_dynamics.graph_models import SceneGraphPredictor, mlp, scene_edges


class StackGraph(SceneGraphPredictor):
    def __init__(self, kind):
        super().__init__(kind)
        self.encoder = mlp(37 + 73, 64, 64)
        self.edges = scene_edges(full=True)

    def forward_nodes(self, nodes, context, edges=None, node_ids=None):
        h, actions = self.encode_graph(nodes, context, edges)
        prediction = self.head(
            torch.cat([h, actions[:, None].expand(-1, 4, -1)], dim=-1)
        )
        if node_ids is not None:
            prediction = prediction[:, torch.argsort(node_ids)]
        return prediction[:, :3].reshape(-1, 9)

    def forward(self, state, context):
        return self.forward_nodes(state.reshape(-1, 4, 4, 37), context)


class StackPredictor(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind in ("temporal_gnn", "temporal_no_messages"):
            self.graph = StackGraph(kind)
            return
        if kind == "history_mlp":
            input_dim = 4 * 148 + 73
        elif kind == "gru":
            self.recurrent = nn.GRU(148, 96, batch_first=True)
            input_dim = 96 + 73
        else:
            raise ValueError(kind)
        self.head = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.ReLU(),
            nn.Linear(160, 9),
        )

    def forward(self, state, context):
        if hasattr(self, "graph"):
            return self.graph(state, context)
        if self.kind == "history_mlp":
            h = state.flatten(1)
        else:
            _, h = self.recurrent(state)
            h = h[-1]
        return self.head(torch.cat([h, context], dim=-1))
