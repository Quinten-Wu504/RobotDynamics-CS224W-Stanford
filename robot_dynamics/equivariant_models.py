import torch
from torch import nn


def canonical_inputs(state, context):
    """Express all geometric inputs in the observed peg body frame."""
    frame = context[:, 3:12].reshape(-1, 3, 3)
    origin = context[:, :3]
    s, c = state.clone(), context.clone()

    def vector(x):
        return torch.einsum("bij,b...i->b...j", frame, x)

    for start in (0, 3):
        s[:, :, start : start + 3] = vector(
            state[:, :, start : start + 3] - origin[:, None]
        )
    for start in (6, 15):
        rotation = state[:, :, start : start + 9].reshape(-1, state.shape[1], 3, 3)
        s[:, :, start : start + 9] = torch.matmul(
            frame.transpose(-1, -2)[:, None], rotation
        ).flatten(-2)
    for start in (33, 36, 39):
        s[:, :, start : start + 3] = vector(state[:, :, start : start + 3])
    for start in (0, 12, 27, 33):
        c[:, start : start + 3] = vector(context[:, start : start + 3] - origin)
    for start in (3, 15, 36):
        c[:, start : start + 9] = torch.matmul(
            frame.transpose(-1, -2), context[:, start : start + 9].reshape(-1, 3, 3)
        ).flatten(-2)
    c[:, 30:33] = vector(context[:, 30:33])
    actions = context[:, 48:118].reshape(-1, 10, 7)
    local_actions = actions.clone()
    local_actions[:, :, :3] = vector(actions[:, :, :3])
    local_actions[:, :, 3:6] = vector(actions[:, :, 3:6])
    c[:, 48:118] = local_actions.flatten(1)
    return s, c, frame


class CanonicalPredictor(nn.Module):
    def __init__(self, base):
        super().__init__()
        from robot_dynamics.learning_models import DynamicsPredictor

        self.base = DynamicsPredictor(base)

    def forward(self, state, context):
        s, c, frame = canonical_inputs(state, context)
        local = self.base(s, c).reshape(-1, 2, 3)
        return torch.einsum("bij,bnj->bni", frame, local).flatten(1)
