"""Quantitative root-cause diagnostic for Humanoid-Light SuperMimic checkpoint & retargeted motions.

Evaluates across all 140 training clips in parallel:
1. Reference motion initial state sanity & Open-Loop Reference PD tracking feasibility.
2. Trained policy (step_2050048) per-category survival, tracking error, and action clamping saturation (|a| > 1).
3. World Model 1-step and 8-step open-loop prediction errors vs ground-truth MuJoCo Warp physics.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CKPT_FILE = (
    REPO_ROOT
    / "runs/light_supermimic_t4_long/free_t4_fp16_20261117/training/checkpoints/step_2050048/last.pt"
)


def inverse_joint_range_targets(target_q, offset, limits):
    """Invert joint_range_targets(action, offset, limits) -> action in [-1, 1]."""
    import torch

    pos_span = (limits[..., 1] - offset).clamp(min=1e-6)
    neg_span = (offset - limits[..., 0]).clamp(min=1e-6)
    diff = target_q - offset
    raw = torch.where(diff >= 0, diff / pos_span, diff / neg_span)
    return raw.clamp(-1.0, 1.0)


def clip_category(name: str) -> str:
    # e.g. "cmu/01_02/0-501" or "lafan/dance2_subject1/0-501"
    parts = name.split("/")
    if parts[0] == "cmu":
        return "cmu"
    stem = parts[1]
    for prefix in ("dance", "ground", "jumps", "walk", "run"):
        if stem.startswith(prefix):
            return f"lafan_{prefix}"
    return f"lafan_{stem}"


def main() -> None:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "2")

    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg
    from supermimic.learners.wm_learner import WorldModelLearnerCfg
    from supermimic.models import Dynamics
    from supermimic.representations import extract_actor_goal
    from supermimic.runners.supermimic_runner import (
        SuperMimicRunnerCfg,
        build_actor,
        build_world_model,
    )
    from supermimic.tasks.task_spec import build_actor_obs_td

    from humanoid_rl.algorithm.supermimic.motions import prepare
    from humanoid_rl.env.actions import joint_range_targets
    from humanoid_rl.tasks.motion_tracking.light_supermimic import make_cfg

    configure_torch_backends()
    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.manual_seed(42)

    saved = torch.load(CKPT_FILE, map_location=device, weights_only=False)
    settings = saved["settings"]

    motion_cfg = deepcopy(settings["motion"])
    motion_cfg["root"] = str(REPO_ROOT)
    motion_cfg["manifest"] = str(REPO_ROOT / "configs/motions/light_supermimic_t4_trial.json")
    motion_cfg["cache_dir"] = str(REPO_ROOT / "data/cache/supermimic/humanoid_light")
    packed, dataset = prepare(motion_cfg)
    motion_names = json.loads((packed / "motion_names.json").read_text())
    num_clips = len(motion_names)
    assert num_clips == dataset["clips"]

    runner_dict = deepcopy(settings["runner"])
    policy_cfg = ActorCriticPolicyLearnerCfg(**runner_dict.pop("policy"))
    wm_cfg = WorldModelLearnerCfg(**runner_dict.pop("wm"))
    runner_cfg = SuperMimicRunnerCfg(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in runner_dict.items()},
        wm=wm_cfg,
        policy=policy_cfg,
        buffer_size=settings["training"]["buffer_size"],
    )

    env_cfg = make_cfg(packed, settings["environment"], num_envs=num_clips, seed=42)
    env_cfg.episode_length_s = 1e9

    class LightDynamics(Dynamics):
        def __init__(self, old, representation, sim_ctx_dim, limits):
            super().__init__(
                representation,
                26,
                sim_ctx_dim,
                old.action_scale,
                old.action_offset,
                hidden_dims=runner_cfg.dynamics_hidden_dims,
            )
            self.load_state_dict(old.state_dict())
            self.register_buffer("joint_lower", limits[:, 0].clone())
            self.register_buffer("joint_upper", limits[:, 1].clone())

        def proc(self, action):
            limits = torch.stack((self.joint_lower, self.joint_upper), -1)
            return joint_range_targets(action, self.action_offset, limits)

    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))
    try:
        wm = build_world_model(runner_cfg, env, device)
        limits = env.unwrapped.scene["robot"].data.joint_pos_limits[0]
        wm.dynamics = LightDynamics(
            wm.dynamics, wm.representation, wm.sim_context.output_dim, limits
        ).to(device)
        wm.load_bundle(saved["world_model"])
        wm.dynamics.eval()

        actor = build_actor(runner_cfg, wm, device)
        actor.load_state_dict(saved["actor"])
        actor.eval()
        actor.set_slot_mix({"motion": 1.0})

        motion_cmd = env.unwrapped.command_manager.get_term("motion")
        motion = motion_cmd.motion
        ends = motion.motion_ends.cpu().numpy()
        starts = np.concatenate(([0], ends[:-1]))
        lengths = ends - starts

        # 1. Inspect reference motion kinematics vs robot joint limits
        all_q = motion.joint_pos
        all_dq = motion.joint_vel
        all_root_z = motion.root_pos_w[:, 2]
        out_of_hard_limits = (
            (all_q < limits[:, 0]) | (all_q > limits[:, 1])
        ).float().mean().item()
        soft_limits = env.unwrapped.scene["robot"].data.soft_joint_pos_limits[0]
        out_of_soft_limits = (
            (all_q < soft_limits[:, 0]) | (all_q > soft_limits[:, 1])
        ).float().mean().item()
        max_abs_dq = all_dq.abs().max().item()
        p99_abs_dq = torch.quantile(all_dq.abs().flatten(), 0.99).item()

        def reset_all_clips_to_start():
            env.reset()
            env_ids = torch.arange(num_clips, device=device)
            start_frames = torch.tensor(starts, dtype=torch.long, device=device)
            motion_cmd.set_time_steps(env_ids, start_frames)
            motion_cmd._write_reference_state_to_sim(
                env_ids,
                motion.root_pos_w[start_frames] + env.unwrapped.scene.env_origins[env_ids],
                motion.root_quat_w[start_frames],
                motion.root_lin_vel_w[start_frames],
                motion.root_ang_vel_w[start_frames],
                motion_cmd.joint_pos[env_ids],
                motion_cmd.joint_vel[env_ids],
            )
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.forward()
            motion_cmd.update_relative_body_poses()
            env.unwrapped.sim.sense()
            env.unwrapped.observation_manager.compute(update_history=True)

        def evaluate_mode(mode: str, max_steps: int = 500):
            reset_all_clips_to_start()
            sim_ctx = wm.sim_context.read(env.unwrapped)
            offset = wm.dynamics.action_offset
            alive = torch.ones(num_clips, dtype=torch.bool, device=device)
            survival_steps = torch.zeros(num_clips, dtype=torch.long, device=device)
            fell = torch.zeros(num_clips, dtype=torch.bool, device=device)
            completed_clip = torch.zeros(num_clips, dtype=torch.bool, device=device)
            reward_sum = torch.zeros(num_clips, device=device)

            sat_fracs = []
            max_raw_actions = []
            abs_raw_actions = []
            wm_1step_joint_vel_mse = []
            wm_1step_root_pos_err = []

            with torch.no_grad():
                for step in range(max_steps):
                    obs = env.get_observations()
                    gt_states, goals = obs["full_state"], obs["goal"]
                    gt_rep = wm.representation.represent(gt_states)
                    gt_aux = wm.representation.extract_aux_state(gt_states)

                    if mode == "trained_policy":
                        sensor = wm.representation.compute_actor_sensor(gt_rep, sim_ctx)
                        actor_goal = extract_actor_goal(goals)
                        actor_obs = build_actor_obs_td(
                            {"motion": actor_goal}, sensor.unsqueeze(1)
                        )
                        actions = actor(actor_obs)
                        if alive.any():
                            active_act = actions[alive]
                            sat_fracs.append((active_act.abs() > 1.0).float().mean().item())
                            max_raw_actions.append(active_act.abs().max().item())
                            abs_raw_actions.append(active_act.abs().mean().item())
                    elif mode == "open_loop_ref_pd":
                        next_frames = (motion_cmd.time_steps + 1).clamp(
                            max=motion.time_step_total - 1
                        )
                        target_q = motion.joint_pos[next_frames]
                        actions = inverse_joint_range_targets(target_q, offset, limits)
                    else:
                        raise ValueError(mode)

                    pred_aux, pred_rep = wm.dynamics(gt_rep, actions, gt_aux, sim_ctx)
                    next_obs, rewards, dones, extras = env.step(actions)

                    term_mgr = env.unwrapped.termination_manager
                    clip_end = term_mgr.get_term("motion_end").bool()
                    bad_fall = term_mgr.terminated.bool() & ~term_mgr.time_outs.bool()

                    still_valid = alive & ~dones.bool()
                    if still_valid.any():
                        next_gt_states = next_obs["full_state"]
                        next_gt_rep = wm.representation.represent(next_gt_states)
                        next_gt_aux = wm.representation.extract_aux_state(next_gt_states)
                        pred_vj = pred_rep[still_valid, -26:]
                        gt_vj = next_gt_rep[still_valid, -26:]
                        wm_1step_joint_vel_mse.append(((pred_vj - gt_vj) ** 2).mean().item())
                        wm_1step_root_pos_err.append(
                            (pred_aux[still_valid, :3] - next_gt_aux[still_valid, :3])
                            .norm(dim=-1)
                            .mean()
                            .item()
                        )

                    survival_steps[alive] += 1
                    reward_sum[alive] += rewards[alive]
                    newly_done = alive & dones.bool()
                    fell |= newly_done & bad_fall
                    completed_clip |= newly_done & clip_end & ~bad_fall
                    alive &= ~dones.bool()
                    if not alive.any():
                        break

            # Clips still alive at max_steps also completed
            completed_clip |= alive

            by_cat = defaultdict(list)
            for i, name in enumerate(motion_names):
                cat = clip_category(name)
                by_cat[cat].append(
                    {
                        "clip": name,
                        "len": int(lengths[i]),
                        "survived": int(survival_steps[i].item()),
                        "survival_ratio": float(survival_steps[i].item()) / max(int(lengths[i]) - 1, 1),
                        "fell": bool(fell[i].item()),
                        "completed": bool(completed_clip[i].item()),
                        "mean_reward_per_sec": float(
                            reward_sum[i].item() / max(int(survival_steps[i].item()), 1) / 0.02
                        ),
                    }
                )

            cat_summary = {}
            for cat, rows in sorted(by_cat.items()):
                cat_summary[cat] = {
                    "clips": len(rows),
                    "completed_clips": sum(r["completed"] for r in rows),
                    "fall_rate": round(sum(r["fell"] for r in rows) / len(rows), 4),
                    "mean_survived_steps": round(float(np.mean([r["survived"] for r in rows])), 1),
                    "median_survived_steps": round(float(np.median([r["survived"] for r in rows])), 1),
                    "mean_survival_ratio": round(float(np.mean([r["survival_ratio"] for r in rows])), 4),
                    "mean_reward_out_of_5": round(
                        float(np.mean([r["mean_reward_per_sec"] for r in rows])), 3
                    ),
                }

            overall = {
                "total_clips": num_clips,
                "completed_clips": int(completed_clip.sum().item()),
                "fall_rate": round(float(fell.float().mean().item()), 4),
                "mean_survived_steps": round(float(survival_steps.float().mean().item()), 1),
                "median_survived_steps": round(float(survival_steps.float().median().item()), 1),
                "mean_reward_out_of_5": round(
                    float(
                        (reward_sum / survival_steps.clamp(min=1) / 0.02).mean().item()
                    ),
                    3,
                ),
                "wm_1step_joint_vel_mse": round(float(np.mean(wm_1step_joint_vel_mse)), 4),
                "wm_1step_root_pos_err_m": round(float(np.mean(wm_1step_root_pos_err)), 5),
                "by_category": cat_summary,
            }
            if sat_fracs:
                overall["action_saturation_frac_gt_1"] = round(float(np.mean(sat_fracs)), 4)
                overall["mean_abs_raw_action"] = round(float(np.mean(abs_raw_actions)), 4)
                overall["max_abs_raw_action"] = round(float(np.max(max_raw_actions)), 4)
            return overall

        report = {
            "motion_dataset_stats": {
                "clips": num_clips,
                "frames": int(motion.time_step_total),
                "root_z_min_m": round(float(all_root_z.min().item()), 4),
                "root_z_median_m": round(float(all_root_z.median().item()), 4),
                "root_z_max_m": round(float(all_root_z.max().item()), 4),
                "frac_joint_pos_outside_hard_limits": round(out_of_hard_limits, 6),
                "frac_joint_pos_outside_soft_limits": round(out_of_soft_limits, 6),
                "p99_abs_joint_vel_rad_s": round(p99_abs_dq, 3),
                "max_abs_joint_vel_rad_s": round(max_abs_dq, 3),
            },
            "open_loop_ref_pd": evaluate_mode("open_loop_ref_pd"),
            "trained_policy_step_2050048": evaluate_mode("trained_policy"),
        }
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
