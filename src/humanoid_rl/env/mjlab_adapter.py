"""Preserve terminal observations and GPU tensors across the off-policy boundary."""

import gymnasium as gym
import numpy as np
import torch


class MjlabAdapter:
    def __init__(self, env, feature_fn=None, control_fn=None):
        if env.cfg.auto_reset:
            raise ValueError("FlashSAC adapter requires auto_reset=False")
        self.env = env
        self.feature_fn = feature_fn
        self.control_fn = control_fn
        self.num_envs = env.num_envs
        observations, _ = env.reset()
        self.observations = observations["actor"].clone()
        self.observation_dim = self.observations.shape[-1]
        self.action_dim = env.action_manager.total_action_dim
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            (self.num_envs, self.observation_dim),
            np.float32,
        )
        self.action_space = gym.spaces.Box(
            -1,
            1,
            (self.num_envs, self.action_dim),
            np.float32,
        )

    def random_actions(self):
        return torch.empty(
            self.num_envs, self.action_dim, device=self.env.device
        ).uniform_(-1, 1)

    def step(self, actions):
        amp_before = (
            self.feature_fn(self.env).clone() if self.feature_fn is not None else None
        )
        actions = actions.detach().clamp(-1, 1)
        previous_action = self.env.action_manager.action.clone() if self.control_fn else None
        obs, reward, terminated, truncated, extras = self.env.step(actions)
        transition = {
            "observation": self.observations,
            "action": actions,
            "reward": reward.clone(),
            "terminated": terminated.clone(),
            "truncated": truncated.clone(),
            "next_observation": obs["actor"].clone(),
        }
        if self.feature_fn is not None:
            transition["amp_observation"] = amp_before
            transition["amp_next_observation"] = self.feature_fn(self.env).clone()
        if self.control_fn is not None:
            transition["control_terms"] = self.control_fn(self.env, actions, previous_action)
        done = transition["terminated"] | transition["truncated"]
        reasons = {
            name: self.env.termination_manager.get_term(name).clone()
            for name in self.env.termination_manager.active_terms
        }
        if done.any():
            reset_obs, _ = self.env.reset(
                env_ids=done.nonzero(as_tuple=False).flatten()
            )
            self.observations = reset_obs["actor"].clone()
        else:
            self.observations = transition["next_observation"]
        return transition, reasons

    def close(self):
        self.env.close()
