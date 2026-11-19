"""Deterministic actor playback with resampled latent commands for BFM-Zero."""

import torch

from humanoid_rl.algorithm.bfm_zero.upstream import enable_bfm_zero
from humanoid_rl.env.bfm_adapter import BfmMjlabAdapter


def make_playback_env(env):
    class PlaybackEnv(BfmMjlabAdapter):
        @property
        def unwrapped(self):
            return self.env

        @property
        def cfg(self):
            return self.env.cfg

        def get_observations(self):
            externally_reset = self.env.episode_length_buf[:, None] < self.step_count
            if externally_reset.any():
                ids = externally_reset[:, 0].nonzero(as_tuple=False).flatten()
                self.history.reset(ids)
                self.last_action[ids], self.step_count[ids] = 0, 0
            self.observations = self.observe()
            return self.observations

        def random_actions(self):
            return torch.empty(self.num_envs, self.action_dim, device=self.device).uniform_(-1,1)

        def reset(self, *, env_ids=None):
            _, extras = self.env.reset(env_ids=env_ids)
            ids = slice(None) if env_ids is None else env_ids
            self.history.reset(ids)
            self.last_action[ids], self.step_count[ids] = 0, 0
            self.observations = self.observe()
            return self.observations, extras

        def step(self, actions):
            transition, metrics = super().step(actions)
            transition["next_observation"] = transition["next"]["observation"]["state"]
            return transition, metrics

    return PlaybackEnv(env, dataset=None)


def make_policy(args, env, metadata):
    enable_bfm_zero()
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModel

    dimensions = {key:value.shape[-1] for key,value in env.get_observations().items()}
    if dimensions != metadata["observation_dims"] or env.action_dim != metadata["action_dim"]:
        raise ValueError("BFM checkpoint observation/action dimensions differ from the task")
    model = FBcprAuxModel.load(str(args.checkpoint/"model"), device=torch.device(env.device).type, strict=False)
    z = model.sample_z(env.num_envs, device=env.device)
    period = metadata["train_config"]["agent"]["update_z_every_step"]

    @torch.no_grad()
    def policy(observations):
        nonlocal z
        resample = env.step_count % period == 0
        if resample.any():
            z = torch.where(resample, model.sample_z(env.num_envs, device=env.device), z)
        return model.act(observations, z, mean=True)

    return policy
