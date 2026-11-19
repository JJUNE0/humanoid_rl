"""Reuse FlashSAC updates; inject only current-discriminator replay rewards."""

from humanoid_rl.algorithm.flashsac.upstream import enable_flashsac

enable_flashsac()
from flash_rl.agents.flashSAC.agent import FlashSACAgent  # noqa: E402

from humanoid_rl.algorithm.amp.replay import AmpReplay  # noqa: E402


class AmpFlashSACAgent(FlashSACAgent):
    def __init__(self, observation_space, action_space, env_info, cfg, amp):
        if cfg.n_step != 1:
            raise ValueError("AMP-only training currently requires --n-step 1")
        super().__init__(observation_space, action_space, env_info, cfg)
        self.amp = amp
        self.auxiliary = None
        if amp.cfg.control is not None:
            from humanoid_rl.algorithm.amp.auxiliary import AuxiliaryCritic
            from humanoid_rl.env.control import TERM_NAMES

            self.auxiliary = AuxiliaryCritic(
                self._critic_observation_dim, self._action_dim, cfg, amp.cfg.control, self._device
            )
        self._replay_buffer = AmpReplay(
            self._replay_buffer, amp.feature_dim, len(TERM_NAMES) if self.auxiliary else 0
        )

    def _relabel(self, batch):
        if self.auxiliary is not None:
            self.auxiliary.update(batch, self._actor)
        return self.amp.relabel(batch)

    def update(self):
        pairs = self._replay_buffer.sample_pairs(self.amp.cfg.batch_size, self._device)
        metrics = self.amp.update(pairs)
        metrics.update(super().update(
            batch_transform=self._relabel,
            actor_q_bonus=self.auxiliary.actor_bonus if self.auxiliary else None,
        ))
        if self.auxiliary is not None:
            metrics.update(self.auxiliary.metrics)
        metrics.update(self.amp.last_reward_metrics)
        return metrics

    def save(self, path):
        from pathlib import Path

        super().save(path)
        self.amp.save(Path(path) / getattr(self.amp, "checkpoint_name", "amp.pt"))
        if self.auxiliary is not None:
            self.auxiliary.save(path)

    def load(self, path):
        from pathlib import Path

        self.amp.load(Path(path) / getattr(self.amp, "checkpoint_name", "amp.pt"))
        super().load(path)
        if self.auxiliary is not None:
            self.auxiliary.load(path)
