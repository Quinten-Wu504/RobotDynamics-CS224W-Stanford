import torch
from torch import nn


class DynamicsPredictor(nn.Module):
    def __init__(self, kind, state_dim=42, context_dim=121, history=4):
        super().__init__()
        self.kind = kind
        if kind in (
            "gnn",
            "temporal_gnn",
            "temporal_no_messages",
            "temporal_full_edges",
        ):
            from robot_dynamics.graph_models import SceneGraphPredictor

            self.graph = SceneGraphPredictor(kind)
            return
        if kind == "current_mlp":
            width, input_dim = 256, state_dim + context_dim
        elif kind == "history_mlp":
            width, input_dim = 224, history * state_dim + context_dim
        elif kind == "gru":
            self.recurrent = nn.GRU(state_dim, 96, batch_first=True)
            width, input_dim = 192, 96 + context_dim
        else:
            raise ValueError(kind)
        self.head = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 6),
        )

    def forward(self, state, context):
        if hasattr(self, "graph"):
            return self.graph(state, context)
        if self.kind == "current_mlp":
            encoded = state[:, -1]
        elif self.kind == "history_mlp":
            encoded = state.flatten(1)
        else:
            _, hidden = self.recurrent(state)
            encoded = hidden[-1]
        return self.head(torch.cat([encoded, context], dim=-1))
