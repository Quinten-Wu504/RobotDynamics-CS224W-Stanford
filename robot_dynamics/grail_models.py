import torch
from torch import nn

from robot_dynamics.grail_data import FEATURES, HISTORY


MODELS = ("history_mlp", "gru", "temporal_gnn", "temporal_no_messages")


class GrailPredictor(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in MODELS:
            raise ValueError(kind)
        self.kind = kind
        if kind == "history_mlp":
            self.head = nn.Sequential(
                nn.Linear(HISTORY * 3 * FEATURES + 3, 64),
                nn.ReLU(),
                nn.Linear(64, 96),
                nn.ReLU(),
                nn.Linear(96, 6),
            )
        elif kind == "gru":
            self.recurrent = nn.GRU(3 * FEATURES, 96, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(99, 192),
                nn.ReLU(),
                nn.Linear(192, 96),
                nn.ReLU(),
                nn.Linear(96, 6),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(FEATURES, 80), nn.ReLU(), nn.Linear(80, 80)
            )
            self.message = nn.Sequential(
                nn.Linear(163, 80), nn.ReLU(), nn.Linear(80, 80)
            )
            self.update = nn.Sequential(
                nn.Linear(160, 80), nn.ReLU(), nn.Linear(80, 80)
            )
            self.recurrent = nn.GRU(80, 80, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(83, 160),
                nn.ReLU(),
                nn.Linear(160, 96),
                nn.ReLU(),
                nn.Linear(96, 3),
            )

    def forward_nodes(self, state, context):
        b, t, n, _ = state.shape
        h = self.encoder(state)
        pos = state[..., :3]
        mask = (1 - torch.eye(n, device=state.device))[None, None, :, :, None]
        for _ in range(2):
            src = h[:, :, :, None].expand(-1, -1, -1, n, -1)
            dst = h[:, :, None, :].expand(-1, -1, n, -1, -1)
            relative = pos[:, :, :, None] - pos[:, :, None, :]
            messages = self.message(torch.cat([src, dst, relative], dim=-1))
            messages = messages * mask * float(self.kind != "temporal_no_messages")
            pooled = messages.sum(2) / (n - 1)
            h = h + self.update(torch.cat([h, pooled], dim=-1))
        _, last = self.recurrent(h.permute(0, 2, 1, 3).reshape(b * n, t, 80))
        last = last[-1].reshape(b, n, 80)
        return self.head(torch.cat([last, context[:, None].expand(-1, n, -1)], dim=-1))

    def forward(self, state, context):
        if self.kind == "history_mlp":
            return self.head(torch.cat([state.flatten(1), context], dim=-1))
        if self.kind == "gru":
            _, last = self.recurrent(state.flatten(2))
            return self.head(torch.cat([last[-1], context], dim=-1))
        return self.forward_nodes(state, context)[:, :2].reshape(-1, 6)
