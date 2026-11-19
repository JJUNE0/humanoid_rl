"""Least-squares AMP discriminator and bounded style reward (AMP Eq. 7)."""

from torch import nn


class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_dims=(512, 256)):
        super().__init__()
        layers, width = [], state_dim * 2
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.ELU()))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, normalized_pairs):
        return self.network(normalized_pairs).squeeze(-1)


class FeatureNormalizer(nn.Module):
    """Frozen train-expert statistics, shared by both sides of the discriminator."""

    def __init__(self, states, min_std=0.1):
        super().__init__()
        self.register_buffer("mean", states.mean(0))
        self.register_buffer("std", states.std(0, unbiased=False).clamp_min(min_std))

    def forward(self, pairs):
        states = pairs.reshape(*pairs.shape[:-1], 2, -1)
        return ((states - self.mean) / self.std).clamp(-10, 10).flatten(-2)


def style_reward(scores):
    return (1 - 0.25 * (scores - 1).square()).clamp(0, 1)
