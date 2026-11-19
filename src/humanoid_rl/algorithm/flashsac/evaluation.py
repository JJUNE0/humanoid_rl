"""Seeded evaluation from ordinary spawns or the explicit fixed-scene start."""

import random
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import torch


@contextmanager
def preserve_random_state(device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class PolicyEvaluator:
    def __init__(self, training_cfg, num_envs, steps, seed, device):
        from mjlab.envs import ManagerBasedRlEnv

        cfg = deepcopy(training_cfg)
        cfg.scene.num_envs = num_envs
        cfg.seed = seed
        cfg.events.pop("amp_reference_pose", None)
        cfg.terminations = {"time_out": cfg.terminations["time_out"]}
        cfg.episode_length_s = (steps + 2) * cfg.sim.mujoco.timestep * cfg.decimation
        with preserve_random_state(device):
            self.env = ManagerBasedRlEnv(cfg, device=device)
        self.steps, self.seed, self.device = steps, seed, device
        self.tracking_features = None
        if getattr(cfg, "add_tracking", False):
            from humanoid_rl.algorithm.add.features import EnvironmentAddFeatures
            self.tracking_features = EnvironmentAddFeatures(self.env, cfg.add_reference)
            # Fixed reference-only scale: evaluation remains comparable as D changes.
            self.tracking_scale = cfg.add_reference.std(0, unbiased=False).clamp_min(.1).to(device)

    @torch.no_grad()
    def evaluate(self, agent):
        from humanoid_rl.tasks.carry_box.mdp.terminations import fallen, placed

        with preserve_random_state(self.device):
            # Seeded ordinary starts, or the fixed experiment's exact frame 0.
            # Random-time RSI is removed; fixed_scene reset remains enabled.
            self.env.seed(self.seed)
            observations, _ = self.env.reset()
            first_fall = torch.full((self.env.num_envs,), self.steps + 1, device=self.device)
            box_start = self.env.scene["box"].data.root_link_pos_w.clone()
            contact_sensor = self.env.scene["hand_box_contact"]
            groups = [[i for i, name in enumerate(contact_sensor.primary_names) if f"{side}_hand" in name]
                      for side in ("left", "right")]
            if not all(groups):
                raise ValueError("Carry evaluation requires left/right hand contacts")
            ever_contact = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
            ever_bilateral = torch.zeros_like(ever_contact)
            ever_held_lift = torch.zeros_like(ever_contact)
            ever_placed = torch.zeros_like(ever_contact)
            max_rise = torch.zeros(self.env.num_envs, device=self.device)
            min_goal_error = (box_start - self.env.carry_goal_w).norm(dim=-1)
            tracking_squared_error = torch.zeros((), device=self.device)
            tracking_metrics = {}
            tracking_steps = 0
            for step in range(1, self.steps + 1):
                mean, _ = agent._actor.apply("get_mean_and_std", observations["actor"], training=False)
                observations, _, _, _, _ = self.env.step(mean.tanh())
                first_fall = torch.where(fallen(self.env) & (first_fall > self.steps), step, first_fall)
                found = contact_sensor.data.found.flatten(1)
                held = torch.stack([found[:, group].any(-1) for group in groups], -1)
                box = self.env.scene["box"].data.root_link_pos_w
                rise = box[:, 2] - box_start[:, 2]
                ever_contact |= held.any(-1)
                ever_bilateral |= held.all(-1)
                ever_held_lift |= held.all(-1) & (rise > .05)
                ever_placed |= placed(self.env)
                max_rise = torch.maximum(max_rise, rise)
                min_goal_error = torch.minimum(min_goal_error, (box - self.env.carry_goal_w).norm(dim=-1))
                if self.tracking_features is not None and step < len(self.tracking_features.reference):
                    from humanoid_rl.algorithm.add.features import tracking_errors
                    residual = self.tracking_features(self.env)
                    tracking_squared_error += (residual/self.tracking_scale).square().mean()
                    tracking_steps += 1
                    for key, value in tracking_errors(residual).items():
                        tracking_metrics[key] = tracking_metrics.get(key, 0.0)+value
            dt = self.env.step_dt
            horizon = self.steps * dt
            metrics = {
                "eval/upright_time_s": (first_fall.clamp_max(self.steps).float().mean() * dt).item(),
                "eval/survival_horizon": (first_fall > self.steps).float().mean().item(),
                "eval/horizon_s": horizon,
                "eval/hand_box_contact_fraction": ever_contact.float().mean().item(),
                "eval/bilateral_hand_box_contact_fraction": ever_bilateral.float().mean().item(),
                "eval/bilateral_box_lift_5cm_fraction": ever_held_lift.float().mean().item(),
                "eval/placement_condition_fraction": ever_placed.float().mean().item(),
                "eval/max_box_rise_m": max_rise.mean().item(),
                "eval/min_box_goal_error_m": min_goal_error.mean().item(),
            }
            for seconds in (1, 5, 10):
                if seconds <= horizon + 1e-6:
                    metrics[f"eval/survival_{seconds}s"] = (first_fall * dt > seconds).float().mean().item()
            if self.tracking_features is not None:
                metrics["eval/reference_tracking_rms"] = (tracking_squared_error/tracking_steps).sqrt().item()
                metrics["eval/reference_tracking_horizon_s"] = tracking_steps*dt
                metrics.update({f"eval/{key}": value/tracking_steps for key, value in tracking_metrics.items()})
            return metrics

    def close(self):
        self.env.close()
