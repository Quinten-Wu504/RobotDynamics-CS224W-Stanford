import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from robot_dynamics.equivariant_models import canonical_inputs
from robot_dynamics.learning_data import encode_sample
from robot_dynamics.windows import HISTORY_FIELDS, FIXTURE_FIELDS, read_episode


def graph_backbone(model):
    return model.canonical.base.graph if hasattr(model, "canonical") else model.graph


def encoder_inputs(model, state, context):
    if hasattr(model, "canonical"):
        state, context, _ = canonical_inputs(state, context)
    return state, context


def encode(model, state, context):
    state, context = encoder_inputs(model, state, context)
    return graph_backbone(model).encode(state, context)


def future_inputs(episode, target_index, stats):
    # The target observes a history ending at t+k, never actions at or after t+k.
    sample = {
        "history": {
            k: episode[k][target_index - 3 : target_index + 1].copy()
            for k in HISTORY_FIELDS
        },
        "fixtures": {k: episode[k].copy() for k in FIXTURE_FIELDS},
        "actions": np.zeros((1, 7)),
        "target_delta": np.zeros((2, 3)),
    }
    state, context, _ = encode_sample(sample, stats, 1)
    context[48:] = 0
    return state, context


def target_arrays(cache, arrays, stats):
    states, contexts = [], []
    loaded = {}
    for episode, anchor, horizon in zip(
        arrays["episode"], arrays["anchor"], arrays["horizon"]
    ):
        name = arrays["ids"][int(episode)]
        if name not in loaded:
            loaded[name] = read_episode(cache[name])
        state, context = future_inputs(loaded[name], int(anchor + horizon), stats)
        states.append(state)
        contexts.append(context)
    return {"future_state": np.stack(states), "future_context": np.stack(contexts)}


class JointEmbedding(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.target = copy.deepcopy(graph_backbone(model))
        del self.target.head
        self.target.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)
        )

    @torch.no_grad()
    def update_target(self, model, momentum):
        online = dict(graph_backbone(model).named_parameters())
        for name, parameter in self.target.named_parameters():
            parameter.mul_(momentum).add_(online[name], alpha=1 - momentum)

    def loss(self, model, state, context, future_state, future_context):
        online = encode(model, state, context)[:, :2]
        with torch.no_grad():
            fs, fc = encoder_inputs(model, future_state, future_context)
            target = self.target.encode(fs, fc)[:, :2]
        prediction = self.predictor(online)
        predictive = F.mse_loss(
            F.layer_norm(prediction, (64,)), F.layer_norm(target, (64,))
        )
        centered = online - online.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.square().mean(dim=0) + 1e-4)
        variance = F.relu(0.2 - std).mean()
        covariance = torch.einsum("bni,bnj->nij", centered, centered) / len(online)
        off_diagonal = covariance - torch.diag_embed(
            covariance.diagonal(dim1=-2, dim2=-1)
        )
        decorrelation = off_diagonal.square().sum() / (2 * 64)
        return predictive + variance + decorrelation, {
            "predictive_loss": float(predictive.detach()),
            "variance_loss": float(variance.detach()),
            "covariance_loss": float(decorrelation.detach()),
        }


def representation_stats(latent):
    latent = latent[:, :2].detach().float().cpu()
    result = []
    for entity in range(2):
        centered = latent[:, entity] - latent[:, entity].mean(0)
        values = torch.linalg.svdvals(centered).square()
        probabilities = values / values.sum().clamp_min(1e-20)
        rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-20).log()).sum())
        result.append(
            {
                "mean_std": float(centered.std(0, unbiased=False).mean()),
                "effective_rank": float(rank),
                "fraction_std_above_0_01": float(
                    (centered.std(0, unbiased=False) > 0.01).float().mean()
                ),
            }
        )
    return result


def physical_targets(model, state, context):
    state, _ = encoder_inputs(model, state, context)
    eef = torch.cat([state[:, :, :3], state[:, :, 6:15], state[:, :, 24:33]], -1)
    nut = F.pad(torch.cat([state[:, :, 3:6], state[:, :, 15:24]], -1), (0, 9))
    target = torch.stack([eef, nut], 1).flatten(2)
    mask = torch.ones(2, 4, 21, device=state.device)
    mask[1, :, 12:] = 0
    return target, mask.flatten(1)
