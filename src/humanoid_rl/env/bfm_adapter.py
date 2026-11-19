"""Preserve dictionary observations, histories and terminal transitions for FB-CPR-Aux."""

import gymnasium as gym
import numpy as np
import torch

from humanoid_rl.env.bfm_features import ObservationHistory, from_environment
from humanoid_rl.tasks.motion_prior.mdp import AuxiliaryRewards, fallen


class BfmMjlabAdapter:
    def __init__(self, env, dataset):
        if env.cfg.auto_reset:
            raise ValueError("BFM adapter requires auto_reset=False")
        self.env, self.num_envs, self.device = env, env.num_envs, env.device
        if dataset is not None:
            env.motion_dataset = dataset
        self.action_dim = env.action_manager.total_action_dim
        self.history = ObservationHistory(env.num_envs, env.cfg.motion_prior.history_length, self.action_dim, env.device)
        self.last_action = torch.zeros(env.num_envs, self.action_dim, device=env.device)
        self.step_count = torch.zeros(env.num_envs, 1, dtype=torch.long, device=env.device)
        self.costs = AuxiliaryRewards(env)
        self.profile = env.cfg.motion_prior_agent
        self.action_observation_scale = (float(env.cfg.robot_control["normalize_action_to"])
                                         if self.profile.action_observation_units == "controller" else 1.0)
        env.reset()
        self.observations = self.observe()
        self.observation_space = gym.spaces.Dict({
            key: gym.spaces.Box(-np.inf, np.inf, (value.shape[-1],), np.float32)
            for key, value in self.observations.items()})

    def observe(self):
        return from_environment(self.env) | {
            "last_action": self.last_action.clone()*self.action_observation_scale,
            "history_actor": self.history.observation().clone()}

    @torch.no_grad()
    def step(self, action):
        action = action.detach().clamp(-1, 1)
        previous = self.last_action.clone()
        observation = self.observations
        _, _, terminated, truncated, _ = self.env.step(action)
        history_action = previous if self.profile.history_uses_previous_action else action
        self.history.push(observation["state"], history_action*self.action_observation_scale)
        self.last_action = action.clone()
        self.step_count += 1
        next_obs = self.observe()
        if any(not torch.isfinite(value).all() for value in next_obs.values()):
            raise FloatingPointError("Nonfinite humanoid simulation observation")
        terminated, truncated = terminated.clone(), truncated.clone()
        transition = {"observation": observation, "action": action,
                      "next": {"observation": next_obs, "terminated": terminated[:, None], "truncated": truncated[:, None]},
                      "aux_rewards": self.costs(self.env, action, previous)}
        metrics = {"env/fallen_fraction": fallen(self.env).float().mean().item(),
                   "env/root_height": (self.env.scene["robot"].data.root_link_pos_w[:, 2]-self.env.scene.env_origins[:, 2]).mean().item(),
                   "env/terminated_fraction": terminated.float().mean().item(),
                   "env/timeout_fraction": truncated.float().mean().item()}
        done = terminated | truncated
        if done.any():
            ids = done.nonzero(as_tuple=False).flatten()
            self.env.reset(env_ids=ids)
            self.history.reset(ids)
            self.last_action[ids], self.step_count[ids] = 0, 0
            self.observations = self.observe()
        else:
            self.observations = next_obs
        return transition, metrics
