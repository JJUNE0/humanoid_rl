"""A separate scalar twin critic for normalized control costs; no FB networks."""

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from humanoid_rl.algorithm.flashsac.upstream import enable_flashsac
from humanoid_rl.env.control import TERM_NAMES
from humanoid_rl.algorithm.amp.normalization import RewardEMA

enable_flashsac()
from flash_rl.agents.flashSAC.layer import (  # noqa: E402
    EnsembleFlashSACBlock, EnsembleFlashSACEmbedder, EnsembleUnitRMSNorm,
)
from flash_rl.agents.utils.network import Network  # noqa: E402
from flash_rl.agents.utils.scheduler import warmup_cosine_decay_scheduler  # noqa: E402


class ScalarDoubleCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks):
        super().__init__()
        self.embedder = EnsembleFlashSACEmbedder(2, input_dim, hidden_dim)
        self.encoder = nn.ModuleList(EnsembleFlashSACBlock(2, hidden_dim) for _ in range(num_blocks))
        self.post_norm = EnsembleUnitRMSNorm(2, hidden_dim)
        # Unbounded scalar outputs: negative cost returns do not fit C51 [-5, 5].
        self.head_weight = nn.Parameter(torch.empty(2, hidden_dim))
        nn.init.normal_(self.head_weight, std=0.01)
        self.head_bias = nn.Parameter(torch.zeros(2, 1))

    def forward(self, observations, actions, training=False):
        x = torch.cat((observations, actions), -1).unsqueeze(0).expand(2, -1, -1)
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        return (self.post_norm(x) * self.head_weight[:, None]).sum(-1) + self.head_bias


def td_target(reward, next_q, terminated, discount):
    # Time limits bootstrap from terminal observations; success terminates.
    return reward + discount * (~terminated.bool()).float() * next_q


class AuxiliaryCritic:
    def __init__(self, observation_dim, action_dim, sac_cfg, cfg, device):
        self.cfg = cfg
        self.sac_cfg = sac_cfg
        self.weights = torch.tensor([getattr(cfg, name) for name in TERM_NAMES], device=device)
        self.normalizer = RewardEMA(cfg.ema_decay).to(device)
        net = ScalarDoubleCritic(observation_dim + action_dim, sac_cfg.critic_hidden_dim, sac_cfg.critic_num_blocks).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=sac_cfg.learning_rate_peak, fused=device.type == "cuda")
        schedule = warmup_cosine_decay_scheduler(
            init_value=sac_cfg.learning_rate_init, peak_value=sac_cfg.learning_rate_peak,
            end_value=sac_cfg.learning_rate_end, warmup_steps=sac_cfg.learning_rate_warmup_step,
            decay_steps=sac_cfg.learning_rate_decay_step,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule(step) / sac_cfg.learning_rate_peak)
        self.critic = Network(net, optimizer, scheduler, use_weight_normalization=True)
        self.critic.normalize_parameters()
        self.target = Network(deepcopy(net), use_weight_normalization=True, ema_source=self.critic,
                              ema_tau=sac_cfg.critic_target_update_tau)
        self.target.network.requires_grad_(False)
        self.metrics = {}

    def update(self, batch, actor):
        with torch.no_grad():
            raw_reward = batch["control_terms"] @ self.weights
            reward = self.normalizer(raw_reward)
            next_actions, _ = actor(observations=batch["next_observation"], training=False)
            # Both halves use the same batch statistics, as in FlashSAC's Q update.
            observations = torch.cat((batch["observation"], batch["next_observation"]))
            actions = torch.cat((batch["action"], next_actions))
            next_q = self.target(observations, actions, training=True).chunk(2, dim=1)[1].min(0).values
            target = td_target(reward, next_q, batch["terminated"], batch.get("discount", self.sac_cfg.gamma))
        predictions = self.critic(observations, actions, training=True).chunk(2, dim=1)[0]
        loss = (predictions - target[None]).square().mean()
        self.critic.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.network.parameters(), 10)
        self.critic.optimizer.step()
        self.critic.scheduler.step()
        self.critic.update_step += 1
        self.critic.normalize_parameters()
        self.target.ema_update_parameters()
        self.metrics = {
            "aux/loss": loss.item(), "aux/q": predictions.detach().mean().item(),
            "aux/reward_raw": raw_reward.mean().item(), "aux/reward_normalized": reward.mean().item(),
            "aux/grad_norm": grad_norm.item(),
        } | {f"control/{name}": batch["control_terms"][:, i].mean().item() for i, name in enumerate(TERM_NAMES)}

    def actor_bonus(self, q_amp, observations, actions):
        # Freeze critic parameters while retaining dQ/da for the policy gradient.
        self.critic.network.requires_grad_(False)
        q_aux = self.critic(observations, actions, training=False).min(0).values
        self.critic.network.requires_grad_(True)
        # I-BFM uses |Q_FB| as the common reference. Without FB we use |Q_AMP|.
        # A unit denominator floor avoids amplifying a near-zero auxiliary critic.
        balance = q_amp.detach().abs().mean() / q_aux.detach().abs().mean().clamp_min(1)
        scale = self.cfg.actor_weight * balance
        self.metrics.update({"aux/actor_scale": scale.item(), "aux/actor_q": q_aux.detach().mean().item()})
        return scale * q_aux

    def save(self, path):
        path = Path(path)
        self.critic.save(str(path / "aux_critic.pt"))
        self.target.save(str(path / "aux_target.pt"))
        torch.save(self.normalizer.state_dict(), path / "aux_normalizer.pt")

    def load(self, path):
        path = Path(path)
        self.critic.load(str(path / "aux_critic.pt"))
        self.target.load(str(path / "aux_target.pt"), load_optimizer=False)
        self.normalizer.load_state_dict(torch.load(path / "aux_normalizer.pt", map_location=self.weights.device, weights_only=True))
