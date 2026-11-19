"""Dump exact MuJoCo (mjlab) step 0..80 trace on Clip 0 for side-by-side comparison with Isaac Sim."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TEST_DIR = REPO_ROOT / "test"
for p in (SRC_DIR, TEST_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from humanoid_rl.tasks.motion_tracking.light_supermimic import make_cfg
from sim2sim_supermimic_adapter import (
    JOINT_NAMES,
    PackedMotionLibrary,
    build_motion_actor_goal,
    build_rep_state,
    build_standalone_actor_from_saved,
    compute_coupled_action_scale_tensor,
    load_checkpoint_compat,
    pack_manifest_pure_numpy,
    parse_mjcf_joint_limits,
    raw_action_to_joint_targets_mj,
)

DEFAULT_CHECKPOINT = (
    REPO_ROOT / "runs/light_supermimic_walk1_t4/t4_walk1_fixed_20261118/training"
)
DEFAULT_MANIFEST = REPO_ROOT / "configs/motions/light_supermimic_walk1.json"


def main() -> None:
    device = "cpu"
    saved, _, _ = load_checkpoint_compat(DEFAULT_CHECKPOINT, device=device)
    env_settings = saved["settings"]["environment"]
    packed_dir = pack_manifest_pure_numpy(
        DEFAULT_MANIFEST, repo_root=REPO_ROOT, split="train", chunk_frames=501
    )
    motion = PackedMotionLibrary(packed_dir, device=device)

    env_cfg = make_cfg(packed_dir, env_settings, num_envs=1, seed=42)
    env_cfg.episode_length_s = 1e9
    env_cfg.commands["motion"].sampling_mode = "start"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()

    # Force Clip 0 start frame 0 deterministically
    cmd = env.command_manager.get_term("motion")
    cmd.time_steps[:] = 0
    cmd._resample_command(torch.tensor([0], dtype=torch.long, device=device))
    cmd.time_steps[:] = 0
    robot = env.scene["robot"]
    # Write exact frame 0 state matching validate_light_supermimic_isaac.py
    root_pose = torch.cat(
        [motion.root_pos_w[0:1] + env.scene.env_origins, motion.root_quat_w[0:1]], dim=-1
    )
    root_vel = torch.cat(
        [motion.root_lin_vel_w[0:1], motion.root_ang_vel_w[0:1]], dim=-1
    )
    robot.write_root_link_pose_to_sim(root_pose)
    robot.write_root_link_velocity_to_sim(root_vel)
    robot.write_joint_state_to_sim(motion.joint_pos[0:1], motion.joint_vel[0:1])
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env_settings["physics_dt"])

    body_names = [env.sim.mj_model.body(i).name for i in range(1, env.sim.mj_model.nbody)]
    body_pos_0 = (robot.data.body_link_pos_w[0] - env.scene.env_origins[0]).cpu().tolist()
    body_quat_0 = robot.data.body_link_quat_w[0].cpu().tolist()

    actor = build_standalone_actor_from_saved(saved, device=device)
    scale_tensor = compute_coupled_action_scale_tensor(
        float(env_settings.get("action_scale", 0.25)), device=device
    )
    limits_tensor = parse_mjcf_joint_limits(device=device)

    trace = []
    motion_times = torch.tensor([0], dtype=torch.long, device=device)
    with torch.no_grad():
        for step_idx in range(80):
            root_pos_local = robot.data.root_link_pos_w - env.scene.env_origins
            root_quat = robot.data.root_link_quat_w
            root_lin_vel = robot.data.root_link_lin_vel_w
            root_ang_vel = robot.data.root_link_ang_vel_w
            qj_mj = robot.data.joint_pos
            vj_mj = robot.data.joint_vel

            rep_state = build_rep_state(
                root_pos_local, root_quat, qj_mj, root_lin_vel, root_ang_vel, vj_mj
            )
            actor_goal = build_motion_actor_goal(
                motion, motion_times, root_pos_local, root_quat
            )
            raw_action = actor(rep_state, actor_goal)
            q_target_mj = raw_action_to_joint_targets_mj(
                raw_action,
                env_settings,
                scale_tensor=scale_tensor,
                limits_tensor=limits_tensor,
            )

            # Step mjlab env with raw_action!
            env.step(raw_action)
            substep0_torques = robot.data.actuator_force[0].cpu().tolist()

            post_pos = (robot.data.root_link_pos_w[0] - env.scene.env_origins[0]).cpu().tolist()
            post_quat = robot.data.root_link_quat_w[0].cpu().tolist()
            ref_pos = motion.root_pos_w[motion_times[0]].cpu().tolist()
            post_qj = robot.data.joint_pos[0].cpu().tolist()
            ref_qj = motion.joint_pos[motion_times[0]].cpu().tolist()
            max_j_err_idx = int(np.argmax(np.abs(np.array(post_qj) - np.array(ref_qj))))

            trace.append(
                {
                    "step": step_idx,
                    "root_pos": [round(x, 4) for x in post_pos],
                    "ref_pos": [round(x, 4) for x in ref_pos],
                    "root_quat": [round(x, 4) for x in post_quat],
                    "max_joint_err_name": JOINT_NAMES[max_j_err_idx],
                    "max_joint_err_rad": round(abs(post_qj[max_j_err_idx] - ref_qj[max_j_err_idx]), 4),
                }
            )
            if step_idx == 0:
                trace[-1]["step0_full"] = {
                    "qj_mj_init": [round(x, 5) for x in qj_mj[0].cpu().tolist()],
                    "vj_mj_init": [round(x, 5) for x in vj_mj[0].cpu().tolist()],
                    "raw_action": [round(x, 5) for x in raw_action[0].cpu().tolist()],
                    "q_target_mj": [round(x, 5) for x in q_target_mj[0].cpu().tolist()],
                    "last_substep_torques_all": dict(
                        zip(JOINT_NAMES, [round(x, 4) for x in substep0_torques])
                    ),
                }
            motion_times = motion_times + 1

    out_dir = REPO_ROOT / "outputs/sim2sim/debug_step0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "trace_mujoco.json"
    out_json.write_text(
        json.dumps(
            {
                "frame0_body_pos_local": dict(zip(body_names, body_pos_0)),
                "frame0_body_quat_w": dict(zip(body_names, body_quat_0)),
                "trace": trace,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[DONE] Saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
