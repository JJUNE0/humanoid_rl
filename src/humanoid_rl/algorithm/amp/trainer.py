"""LSGAN + expert gradient penalty; rescore replay with the current discriminator."""

from dataclasses import asdict

import torch

from humanoid_rl.algorithm.amp.discriminator import Discriminator, FeatureNormalizer, style_reward
from humanoid_rl.algorithm.amp.features import feature_schema


class AmpTrainer:
    def __init__(self, cfg, dataset, device):
        self.cfg, self.dataset, self.device = cfg, dataset, device
        self.feature_version, self.feature_groups = feature_schema(cfg.observe_object)
        self.feature_dim = sum(self.feature_groups.values())
        if any(split is not None and split.states.shape[-1] != self.feature_dim
               for split in (dataset.train, dataset.validation)):
            raise ValueError("AMP configuration and expert feature dimensions differ")
        self.normalizer = FeatureNormalizer(
            dataset.train.states, cfg.min_feature_std
        ).to(device)
        self.discriminator = Discriminator(self.feature_dim, cfg.hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.update_step = 0
        self.last_reward_metrics = {}

    @torch.no_grad()
    def reward(self, pairs):
        scores = self.discriminator(self.normalizer(pairs.to(self.device)))
        return self.cfg.reward_scale * style_reward(scores)

    def update(self, policy_pairs):
        self.discriminator.train()
        # The discriminator always runs in FP32, independently of SAC mixed precision.
        expert = self.dataset.train.sample_pairs(
            self.cfg.batch_size, self.device, self.cfg.phase_balanced
        )
        expert = self.normalizer(expert).detach().requires_grad_(True)
        policy = self.normalizer(policy_pairs.detach()).detach()
        expert_score = self.discriminator(expert)
        policy_score = self.discriminator(policy)
        regression = (expert_score - 1).square().mean() + (
            policy_score + 1
        ).square().mean()
        gradient = torch.autograd.grad(expert_score.sum(), expert, create_graph=True)[0]
        penalty = gradient.square().sum(-1).mean()
        loss = regression + 0.5 * self.cfg.gradient_penalty * penalty
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise FloatingPointError("Nonfinite AMP discriminator loss/gradient")
        self.optimizer.step()
        self.update_step += 1
        metrics = {
            "amp/discriminator_loss": float(loss.detach()),
            "amp/regression_loss": float(regression.detach()),
            "amp/gradient_penalty": float(penalty.detach()),
            "amp/gradient_norm": float(grad_norm),
            "amp/expert_score": float(expert_score.mean().detach()),
            "amp/policy_score": float(policy_score.mean().detach()),
            "amp/classification_accuracy": float(
                (
                    (expert_score > 0).float().mean()
                    + (policy_score < 0).float().mean()
                ).detach()
                / 2
            ),
            "amp/discriminator_updates": self.update_step,
        }
        if self.update_step == 1 or self.update_step % self.cfg.validation_every == 0:
            metrics.update(self.validate(policy_pairs))
        return metrics

    @torch.no_grad()
    def validate(self, policy_pairs):
        held_out_enabled = self.dataset.validation is not None
        split = self.dataset.validation if held_out_enabled else self.dataset.train
        prefix = "amp/validation_expert" if held_out_enabled else "amp/train_reference"
        distance_key = "amp/validation_feature_distance" if held_out_enabled else "amp/train_reference_feature_distance"
        held_out = split.sample_pairs(
            self.cfg.batch_size, self.device, self.cfg.phase_balanced
        )
        score = self.discriminator(self.normalizer(held_out))
        # A fixed-feature distance complements the nonstationary learned score.
        distance = torch.cdist(
            self.normalizer(policy_pairs[:128]), self.normalizer(held_out[:128])
        )
        return {
            "amp/held_out_validation_enabled": float(held_out_enabled),
            f"{prefix}_score": float(score.mean()),
            f"{prefix}_accuracy": float((score > 0).float().mean()),
            f"{prefix}_reward": float(style_reward(score).mean()),
            distance_key: float(
                distance.min(-1).values.mean() / (2 * self.feature_dim) ** 0.5
            ),
        }

    def relabel(self, batch):
        pairs = torch.cat((batch["amp_observation"], batch["amp_next_observation"]), -1)
        batch["reward"] = self.reward(pairs)
        self.last_reward_metrics = {
            "amp/replay_reward_mean": float(batch["reward"].mean())
        }
        return batch

    def save(self, path):
        torch.save(
            {
                "feature_version": self.feature_version,
                "feature_dim": self.feature_dim,
                "feature_groups": self.feature_groups,
                "config": asdict(self.cfg),
                "joint_names": self.dataset.joint_names,
                "train_fingerprint": self.dataset.train.fingerprint,
                "validation_fingerprint": self.dataset.validation.fingerprint if self.dataset.validation is not None else None,
                "normalizer": self.normalizer.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_step": self.update_step,
            },
            path,
        )

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        expected = (
            self.feature_version,
            self.dataset.joint_names,
            self.dataset.train.fingerprint,
            self.dataset.validation.fingerprint if self.dataset.validation is not None else None,
        )
        actual = (
            state["feature_version"],
            tuple(state["joint_names"]),
            state["train_fingerprint"],
            state["validation_fingerprint"],
        )
        if actual != expected:
            raise ValueError("AMP checkpoint feature schema or expert dataset differs")
        self.normalizer.load_state_dict(state["normalizer"])
        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update_step = state["update_step"]
