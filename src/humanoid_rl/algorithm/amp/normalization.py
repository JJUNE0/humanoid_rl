"""Running normalization of auxiliary control rewards."""

import torch


class RewardEMA(torch.nn.Module):
    """Bias-corrected EMA variance; scale only, preserving negative cost signs."""
    def __init__(self, decay=0.99):
        super().__init__()
        self.decay = decay
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("square", torch.tensor(0.0))
        self.register_buffer("mass", torch.tensor(0.0))

    @torch.no_grad()
    def forward(self, rewards, update=True):
        if update:
            self.mean.lerp_(rewards.mean(), 1 - self.decay)
            self.square.lerp_(rewards.square().mean(), 1 - self.decay)
            self.mass.lerp_(torch.ones_like(self.mass), 1 - self.decay)
        mass = self.mass.clamp_min(1e-8)
        variance = (self.square / mass - (self.mean / mass).square()).clamp_min(1e-8)
        return rewards / variance.sqrt()
