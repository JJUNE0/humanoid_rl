"""ADD's zero-residual positive target and current-discriminator replay rewards.

Algorithm references:
  https://add-moo.github.io/
  https://github.com/xbpeng/MimicKit/blob/main/mimickit/learning/add_agent.py

Following the official implementation, use BCE logits, mean-absolute residual
scaling without centering, logit weight regularization, and gradient penalties
on both zero and policy inputs. FlashSAC learns through its critic; no simulator
or discriminator gradient is propagated directly into the actor.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from humanoid_rl.algorithm.amp.discriminator import Discriminator
from humanoid_rl.algorithm.amp.trainer import AmpTrainer
from humanoid_rl.algorithm.add.features import FEATURE_VERSION, FEATURE_DIM, FEATURE_GROUPS, tracking_errors


class DifferentialNormalizer(nn.Module):
    """Zero stays zero: divide by running mean |error|, never subtract a mean."""

    def __init__(self, state_dim, minimum=0.1):
        super().__init__()
        self.minimum = minimum
        self.register_buffer("mean_abs", torch.ones(state_dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, pairs):
        values = pairs.reshape(-1, len(self.mean_abs))
        count = len(values)
        total = self.count+count
        self.mean_abs.lerp_(values.abs().mean(0), count/total.float())
        self.count.copy_(total)

    def forward(self, pairs):
        values = pairs.reshape(*pairs.shape[:-1], 2, -1)
        return (values/self.mean_abs.clamp_min(self.minimum)).clamp(-10, 10).flatten(-2)


def differential_reward(logits):
    # -log(1-sigmoid(logit)), capped identically to MimicKit's probability floor.
    return F.softplus(logits).clamp_max(-math.log(1e-4))


class AddTrainer(AmpTrainer):
    checkpoint_name = "add.pt"

    def __init__(self, cfg, dataset, device):
        self.cfg, self.dataset, self.device = cfg, dataset, device
        self.feature_version, self.feature_dim, self.feature_groups = FEATURE_VERSION, FEATURE_DIM, FEATURE_GROUPS
        if dataset.train.states.shape[-1] != FEATURE_DIM:
            raise ValueError("ADD tracking feature dimensions differ")
        self.normalizer = DifferentialNormalizer(FEATURE_DIM, cfg.min_feature_std).to(device)
        self.discriminator = Discriminator(FEATURE_DIM, cfg.hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=cfg.learning_rate,
                                          weight_decay=cfg.weight_decay)
        self.update_step = 0
        self.last_reward_metrics = {}

    @torch.no_grad()
    def reward(self, pairs):
        logits = self.discriminator(self.normalizer(pairs.to(self.device)))
        return self.cfg.reward_scale*differential_reward(logits)

    def update(self, policy_pairs):
        self.discriminator.train()
        policy_pairs = policy_pairs.detach().to(self.device)
        self.normalizer.update(policy_pairs)
        policy = self.normalizer(policy_pairs).detach().requires_grad_(True)
        positive = torch.zeros_like(policy[:1], requires_grad=True)
        zero_logit, policy_logit = self.discriminator(positive), self.discriminator(policy)
        classification = .5*(F.softplus(-zero_logit).mean()+F.softplus(policy_logit).mean())
        policy_grad = torch.autograd.grad(policy_logit.sum(), policy, create_graph=True)[0]
        positive_grad = torch.autograd.grad(zero_logit.sum(), positive, create_graph=True)[0]
        penalty = .5*(policy_grad.square().sum(-1).mean()+positive_grad.square().sum(-1).mean())
        logit_reg = self.discriminator.network[-1].weight.square().sum()
        loss = classification+self.cfg.gradient_penalty*penalty+self.cfg.logit_regularization*logit_reg
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise FloatingPointError("Nonfinite ADD discriminator loss/gradient")
        self.optimizer.step()
        self.update_step += 1
        metrics = {
            "add/discriminator_loss": loss.detach().item(),
            "add/classification_loss": classification.detach().item(),
            "add/gradient_penalty": penalty.detach().item(),
            "add/gradient_norm": grad_norm.item(),
            "add/logit_regularization": logit_reg.detach().item(),
            "add/zero_logit": zero_logit.detach().mean().item(),
            "add/policy_logit": policy_logit.detach().mean().item(),
            "add/classification_accuracy": .5*((zero_logit > 0).float().mean()
                                                +(policy_logit < 0).float().mean()).detach().item(),
            "add/discriminator_updates": self.update_step,
        }
        if self.update_step == 1 or self.update_step % self.cfg.validation_every == 0:
            metrics.update(self.validate(policy_pairs))
        return metrics

    @torch.no_grad()
    def validate(self, policy_pairs):
        # These are tracking diagnostics against the selected clip, not held-out validation.
        return {f"add/{name}": value for name, value in tracking_errors(policy_pairs).items()} | {
            "add/normalized_tracking_rms": self.normalizer(policy_pairs).square().mean().sqrt().item(),
            "add/held_out_validation_enabled": 0.0,
        }

    def relabel(self, batch):
        pairs = torch.cat((batch["amp_observation"], batch["amp_next_observation"]), -1)
        batch["reward"] = self.reward(pairs)
        self.last_reward_metrics = {"add/replay_reward_mean": batch["reward"].mean().item()}
        return batch
