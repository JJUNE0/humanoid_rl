"""Profile whole-body ground contact forces (N & xBW) across LAFAN1 tasks in Isaac Sim 5.1 USD.

Measures both 400 Hz substep transient impact peaks and 50 Hz step-averaged forces across
all anatomical body groups (feet, knees, hips/pelvis, hands/wrists, elbows, shoulders, torso, head),
strictly separating normal tracking steps (alive) from post-fall crash steps.

Usage:
  /home/cocel/isaac/env_isaaclab/bin/python src/humanoid_rl/sim2sim/profile_light_supermimic_contact_forces.py \
      --headless \
      --checkpoint runs/light_supermimic_lafan_all/rtx3090_lafan_all_stage2_20261118/training \
      --manifest src/humanoid_retarget/configs/motions/light_supermimic_lafan_all.json \
      --split validation \
      --report-json outputs/sim2sim/isaac_lafan_all_stage2_contact_forces_val.json
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

COCELO_ISAACLAB_ROOT = Path("/home/cocel/Desktop/Research/humanoid/cocelo-rl-isaaclab")
USD_CANDIDATES = (
    COCELO_ISAACLAB_ROOT
    / "lab/cocelo/assets/data/Robots/Flamingo/humanoid_light_rev_1_0_12_45/assets (2)/humanoid_light_rev_1_0_12_45_merge_joints.usd",
    COCELO_ISAACLAB_ROOT
    / "lab/cocelo/assets/data/Robots/Flamingo/humanoid_light_rev_1_0_12_45/assets/humanoid_light_rev_1_0_12_45_merge_joints.usd",
)

ROBOT_MASS_KG = 36.140
ROBOT_WEIGHT_N = ROBOT_MASS_KG * 9.81  # 354.53 N

BODY_GROUPS: dict[str, list[str]] = {
    "feet": [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_ankle_pitch_link",
        "right_ankle_pitch_link",
    ],
    "knees": ["left_knee_link", "right_knee_link"],
    "hips_pelvis": [
        "pelvis_link",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
    ],
    "hands_wrists": ["left_wrist_link", "right_wrist_link"],
    "elbows": ["left_elbow_link", "right_elbow_link"],
    "shoulders": [
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
    ],
    "torso": ["torso_yaw_link", "torso_pitch_link", "torso_roll_link"],
    "head": ["head_link", "head_cam_link"],
}


def resolve_usd_path() -> Path:
    for candidate in USD_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find Humanoid-Light USD in {USD_CANDIDATES}")


def build_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument(
        "--sequence-filter",
        type=str,
        default="",
        help="Optional comma-separated regex/substring filter on motion names (e.g. 'sprint1,fallAndGetUp2,ground1,walk1_subject1').",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--sim-dt", type=float, default=0.0025)
    parser.add_argument("--decimation", type=int, default=8)
    parser.add_argument("--report-json", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    if str(COCELO_ISAACLAB_ROOT) not in sys.path:
        sys.path.insert(0, str(COCELO_ISAACLAB_ROOT))

    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass

    from lab.cocelo.assets.flamingo.humanoid_light_v1 import HUMANOID_LIGHT_CFG
    from humanoid_rl.sim2sim.sim2sim_supermimic_adapter import (
        JOINT_NAMES,
        PackedMotionLibrary,
        SensorHistoryTracker,
        build_motion_actor_goal,
        build_standalone_actor_from_saved,
        compute_coupled_action_scale_tensor,
        joint_targets_mj_to_isaac_actuator_slots,
        load_checkpoint_compat,
        pack_manifest_pure_numpy,
        parse_mjcf_joint_limits,
        quat_apply_inverse,
        raw_action_to_joint_targets_mj,
    )

    try:
        device = args.device
        saved, ckpt_file, ckpt_sha = load_checkpoint_compat(args.checkpoint, device=device)
        env_settings = saved["settings"]["environment"]
        episode_length_s = float(env_settings.get("episode_length_s", 10.0))
        chunk_frames = max(1, round(episode_length_s * 50))

        packed_dir = pack_manifest_pure_numpy(
            args.manifest.expanduser().resolve(),
            repo_root=REPO_ROOT,
            split=args.split,
            chunk_frames=chunk_frames,
        )
        motion = PackedMotionLibrary(packed_dir, device=device)

        if args.sequence_filter.strip():
            tokens = [t.strip() for t in args.sequence_filter.split(",") if t.strip()]
            clip_indices = [
                c
                for c in range(motion.num_clips)
                if any(tok in motion.motion_names[c] for tok in tokens)
            ]
        else:
            clip_indices = list(range(motion.num_clips))

        if not clip_indices:
            raise RuntimeError(f"No clips matched --sequence-filter={args.sequence_filter!r}")

        num_envs = len(clip_indices)
        usd_path = resolve_usd_path()
        robot_cfg: ArticulationCfg = copy.deepcopy(HUMANOID_LIGHT_CFG).replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        robot_cfg.spawn.usd_path = str(usd_path)

        @configclass
        class ForceProfileSceneCfg(InteractiveSceneCfg):
            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                terrain_generator=None,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
                debug_vis=False,
            )
            robot: ArticulationCfg = robot_cfg
            contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                history_length=args.decimation,
                track_air_time=False,
            )
            light = AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DistantLightCfg(color=(0.8, 0.8, 0.8), intensity=3000.0),
            )

        sim_cfg = sim_utils.SimulationCfg(
            dt=args.sim_dt,
            render_interval=args.decimation,
            device=device,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )
        sim_cfg.physx.gpu_max_rigid_patch_count = 10 * 2**15
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(ForceProfileSceneCfg(num_envs=num_envs, env_spacing=3.0))
        sim.reset()

        robot = scene["robot"]
        contacts = scene["contact_forces"]

        isaac_joint_ids_list, matched_names = robot.find_joints(
            list(JOINT_NAMES), preserve_order=True
        )
        if tuple(matched_names) != JOINT_NAMES:
            raise RuntimeError(f"Joint mapping mismatch: {matched_names}")
        isaac_joint_ids = torch.tensor(isaac_joint_ids_list, dtype=torch.long, device=device)

        group_sensor_ids: dict[str, torch.Tensor] = {}
        group_robot_body_ids: dict[str, torch.Tensor] = {}
        for gname, link_names in BODY_GROUPS.items():
            found_sids: list[int] = []
            found_bids: list[int] = []
            for ln in link_names:
                sids, _ = contacts.find_bodies([ln], preserve_order=True)
                bids, _ = robot.find_bodies([ln], preserve_order=True)
                if sids and bids:
                    found_sids.extend(sids)
                    found_bids.extend(bids)
            group_sensor_ids[gname] = torch.tensor(found_sids, dtype=torch.long, device=device)
            group_robot_body_ids[gname] = torch.tensor(found_bids, dtype=torch.long, device=device)

        l_foot_id = contacts.find_bodies(["left_ankle_roll_link"], preserve_order=True)[0][0]
        r_foot_id = contacts.find_bodies(["right_ankle_roll_link"], preserve_order=True)[0][0]
        l_foot_bid = robot.find_bodies(["left_ankle_roll_link"], preserve_order=True)[0][0]
        r_foot_bid = robot.find_bodies(["right_ankle_roll_link"], preserve_order=True)[0][0]

        fall_body_ids, _ = robot.find_bodies(
            ["torso_roll_link", "head_link", "left_knee_link", "right_knee_link"],
            preserve_order=True,
        )
        torso_id, head_id, l_knee_id, r_knee_id = fall_body_ids

        actor = build_standalone_actor_from_saved(saved, device=device)
        scale_tensor = compute_coupled_action_scale_tensor(
            float(env_settings.get("action_scale", 0.25)), device=device
        )
        limits_tensor = parse_mjcf_joint_limits(device=device)

        clip_starts = torch.tensor(
            [motion.clip_bounds(c)[0] for c in clip_indices], dtype=torch.long, device=device
        )
        clip_ends = torch.tensor(
            [motion.clip_bounds(c)[1] for c in clip_indices], dtype=torch.long, device=device
        )

        def reset_envs(env_ids: torch.Tensor) -> None:
            frames = clip_starts[env_ids]
            root_pos = motion.root_pos_w[frames] + scene.env_origins[env_ids]
            root_quat = motion.root_quat_w[frames]
            robot.write_root_link_pose_to_sim(
                torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids
            )
            robot.write_root_link_velocity_to_sim(
                torch.cat(
                    [motion.root_lin_vel_w[frames], motion.root_ang_vel_w[frames]], dim=-1
                ),
                env_ids=env_ids,
            )
            q_isaac = robot.data.default_joint_pos[env_ids].clone()
            dq_isaac = robot.data.default_joint_vel[env_ids].clone()
            q_isaac[:, isaac_joint_ids] = motion.joint_pos[frames]
            dq_isaac[:, isaac_joint_ids] = motion.joint_vel[frames]
            robot.write_joint_state_to_sim(q_isaac, dq_isaac, env_ids=env_ids)
            robot.reset(env_ids=env_ids)
            contacts.reset(env_ids=env_ids)

        all_env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
        motion_times = clip_starts.clone()
        reset_envs(all_env_ids)
        scene.write_data_to_sim()
        sim.forward()
        scene.update(dt=args.sim_dt)

        last_raw_action = torch.zeros((num_envs, len(JOINT_NAMES)), device=device)
        alive = torch.ones(num_envs, dtype=torch.bool, device=device)
        world_down = torch.tensor([0.0, 0.0, -1.0], device=device).expand(num_envs, 3)

        # Accumulate per-env, per-step alive contact metrics on CPU lists grouped by sequence prefix
        seq_names = [motion.motion_names[c].split("/")[1] for c in clip_indices]
        cat_names = [re.sub(r"_subject\d+$", "", s) for s in seq_names]

        # Store step-level records per category:
        # For each body group: list of 400Hz peak forces (N) across alive steps, and 50Hz mean forces (N)
        cat_peak400: dict[str, dict[str, list[float]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        cat_mean50: dict[str, dict[str, list[float]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        # Single-foot strike / stance forces (max(F_L, F_R)) and vertical foot touchdown speed
        cat_single_foot_peak400: dict[str, list[float]] = collections.defaultdict(list)
        cat_single_foot_mean50: dict[str, list[float]] = collections.defaultdict(list)
        cat_loading_rate_ns: dict[str, list[float]] = collections.defaultdict(list)
        cat_alive_steps: dict[str, int] = collections.defaultdict(int)
        cat_fall_crash_peak400: dict[str, list[float]] = collections.defaultdict(list)

        sensor_tracker = SensorHistoryTracker(saved, num_envs=num_envs, device=device)
        with torch.no_grad():
            for _step_idx in range(args.steps):
                root_pos_local = robot.data.root_link_pos_w - scene.env_origins
                root_quat = robot.data.root_link_quat_w
                root_lin_vel = robot.data.root_link_lin_vel_w
                root_ang_vel = robot.data.root_link_ang_vel_w
                qj_mj = robot.data.joint_pos[:, isaac_joint_ids]
                vj_mj = robot.data.joint_vel[:, isaac_joint_ids]

                sensor_obs = sensor_tracker.observe(
                    root_pos_local, root_quat, qj_mj, root_lin_vel, root_ang_vel, vj_mj
                )
                actor_goal = build_motion_actor_goal(
                    motion,
                    motion_times,
                    root_pos_local,
                    root_quat,
                    goal_root_twist=sensor_tracker.goal_root_twist,
                )
                raw_action = actor(sensor_obs, actor_goal)
                sensor_tracker.record_action(raw_action)
                q_target_mj = raw_action_to_joint_targets_mj(
                    raw_action,
                    env_settings,
                    scale_tensor=scale_tensor,
                    limits_tensor=limits_tensor,
                )
                q_target_slots = joint_targets_mj_to_isaac_actuator_slots(q_target_mj)
                q_target_isaac = robot.data.default_joint_pos.clone()
                q_target_isaac[:, isaac_joint_ids] = q_target_slots
                robot.set_joint_position_target(q_target_isaac)

                for _ in range(args.decimation):
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(args.sim_dt)

                last_raw_action = raw_action

                post_root_pos = robot.data.root_link_pos_w - scene.env_origins
                post_root_quat = robot.data.root_link_quat_w
                body_z = robot.data.body_link_pos_w[:, :, 2]

                ref_root_pos = motion.root_pos_w[motion_times]
                ref_root_quat = motion.root_quat_w[motion_times]

                height_err = (post_root_pos[:, 2] - ref_root_pos[:, 2]).abs()
                g_rob = quat_apply_inverse(post_root_quat, world_down)
                g_ref = quat_apply_inverse(ref_root_quat, world_down)
                grav_err = torch.linalg.vector_norm(g_rob - g_ref, dim=-1)

                ref_upright = (ref_root_pos[:, 2] > 0.58) & (g_ref[:, 2] < -0.75)
                low_body = ref_upright & (
                    (post_root_pos[:, 2] < 0.45)
                    | (body_z[:, torso_id] < 0.50)
                    | (body_z[:, head_id] < 0.75)
                    | (body_z[:, l_knee_id] < 0.12)
                    | (body_z[:, r_knee_id] < 0.12)
                )
                bad_fall = (height_err > 0.25) | (grav_err > 0.70) | low_body

                # Force history: (E, 8, num_bodies, 3)
                # Use upward vertical GRF (Fz > 0) & 3D norm where Fz > 1.0 N so pure aerial/internal self-rub is excluded
                f_hist = contacts.data.net_forces_w_history  # (E, 8, B, 3)
                fz_hist = f_hist[..., 2].clamp(min=0.0)  # upward support/impact component
                fnorm_hist = torch.where(
                    fz_hist > 1.0, f_hist.norm(dim=-1), torch.zeros_like(fz_hist)
                )

                # Single-foot forces (max of left or right sole, gated near ground z < 0.22m)
                lf_near = (body_z[:, l_foot_bid] < 0.22).unsqueeze(1)
                rf_near = (body_z[:, r_foot_bid] < 0.22).unsqueeze(1)
                single_foot_hist = torch.maximum(
                    torch.where(lf_near, fnorm_hist[:, :, l_foot_id], 0.0),
                    torch.where(rf_near, fnorm_hist[:, :, r_foot_id], 0.0),
                )  # (E, 8)
                sf_peak400 = single_foot_hist.max(dim=1).values.cpu().numpy()
                sf_mean50 = single_foot_hist.mean(dim=1).cpu().numpy()

                # Loading rate dF/dt across 400Hz substeps for feet
                df_dt = (
                    (single_foot_hist[:, 1:] - single_foot_hist[:, :-1]).clamp(min=0.0).max(dim=1).values
                    / args.sim_dt
                ).cpu().numpy()

                group_z_max = {
                    "feet": 0.13,
                    "hands_wrists": 0.10,
                    "elbows": 0.14,
                    "knees": 0.16,
                    "hips_pelvis": 0.18,
                    "shoulders": 0.18,
                    "head": 0.18,
                    "torso": 0.24,
                }
                group_peak400_np: dict[str, np.ndarray] = {}
                group_mean50_np: dict[str, np.ndarray] = {}
                for gname, gids in group_sensor_ids.items():
                    bids = group_robot_body_ids[gname]
                    z_thresh = group_z_max[gname]
                    near_ground = (body_z[:, bids] < z_thresh)[:, None, :, None]  # (E, 1, G, 1)
                    g_vec = torch.where(near_ground, f_hist[:, :, gids, :], 0.0).sum(dim=2)  # (E, 8, 3)
                    g_fz = g_vec[..., 2].clamp(min=0.0)
                    g_mag = torch.where(
                        g_fz > 1.0, g_vec.norm(dim=-1), torch.zeros_like(g_fz)
                    )  # (E, 8)
                    group_peak400_np[gname] = g_mag.max(dim=1).values.cpu().numpy()
                    group_mean50_np[gname] = g_mag.mean(dim=1).cpu().numpy()

                # Total non-foot body crash force when falling
                all_non_foot_peak = np.zeros(num_envs, dtype=np.float32)
                for gname in ("knees", "hips_pelvis", "hands_wrists", "elbows", "shoulders", "torso", "head"):
                    all_non_foot_peak = np.maximum(all_non_foot_peak, group_peak400_np[gname])

                alive_np = alive.cpu().numpy()
                bad_fall_np = bad_fall.cpu().numpy()

                for i in range(num_envs):
                    if not alive_np[i]:
                        continue
                    cat = cat_names[i]
                    if bad_fall_np[i]:
                        cat_fall_crash_peak400[cat].append(float(all_non_foot_peak[i]))
                        continue
                    cat_alive_steps[cat] += 1
                    cat_single_foot_peak400[cat].append(float(sf_peak400[i]))
                    cat_single_foot_mean50[cat].append(float(sf_mean50[i]))
                    cat_loading_rate_ns[cat].append(float(df_dt[i]))
                    for gname in BODY_GROUPS:
                        cat_peak400[cat][gname].append(float(group_peak400_np[gname][i]))
                        cat_mean50[cat][gname].append(float(group_mean50_np[gname][i]))

                next_times = motion_times + 1
                at_clip_end = next_times >= clip_ends
                alive &= ~(bad_fall | at_clip_end)
                motion_times = torch.where(at_clip_end, clip_ends - 1, next_times)
                if not alive.any():
                    break

        def summarize_arr(vals: list[float], contact_thresh: float = 5.0) -> dict[str, Any]:
            if not vals:
                return {
                    "contact_ratio": 0.0,
                    "active_mean_n": 0.0,
                    "p95_n": 0.0,
                    "p99_n": 0.0,
                    "max_n": 0.0,
                    "p99_bw": 0.0,
                    "max_bw": 0.0,
                }
            arr = np.asarray(vals, dtype=np.float64)
            active = arr[arr >= contact_thresh]
            contact_ratio = float(len(active) / len(arr))
            if len(active) == 0:
                return {
                    "contact_ratio": 0.0,
                    "active_mean_n": 0.0,
                    "p95_n": 0.0,
                    "p99_n": 0.0,
                    "max_n": round(float(arr.max()), 1),
                    "p99_bw": 0.0,
                    "max_bw": round(float(arr.max() / ROBOT_WEIGHT_N), 2),
                }
            p99 = float(np.percentile(active, 99))
            mx = float(active.max())
            return {
                "contact_ratio": round(contact_ratio, 4),
                "active_mean_n": round(float(active.mean()), 1),
                "p95_n": round(float(np.percentile(active, 95)), 1),
                "p99_n": round(p99, 1),
                "max_n": round(mx, 1),
                "p99_bw": round(p99 / ROBOT_WEIGHT_N, 2),
                "max_bw": round(mx / ROBOT_WEIGHT_N, 2),
            }

        categories_report: dict[str, Any] = {}
        for cat in sorted(cat_alive_steps.keys()):
            categories_report[cat] = {
                "alive_steps_sampled": cat_alive_steps[cat],
                "single_foot_400hz_peak": summarize_arr(cat_single_foot_peak400[cat]),
                "single_foot_50hz_mean": summarize_arr(cat_single_foot_mean50[cat]),
                "foot_loading_rate_kns_p99": round(
                    float(np.percentile(cat_loading_rate_ns[cat], 99)) / 1000.0, 2
                )
                if cat_loading_rate_ns[cat]
                else 0.0,
                "body_groups_400hz_peak": {
                    gname: summarize_arr(cat_peak400[cat][gname]) for gname in BODY_GROUPS
                },
                "body_groups_50hz_mean": {
                    gname: summarize_arr(cat_mean50[cat][gname]) for gname in BODY_GROUPS
                },
                "fall_step_nonfoot_max_n": round(
                    float(max(cat_fall_crash_peak400[cat])), 1
                )
                if cat_fall_crash_peak400[cat]
                else 0.0,
            }

        report = {
            "simulator": "Isaac Sim 5.1 / IsaacLab v2.3.0 PhysX",
            "checkpoint": str(ckpt_file),
            "checkpoint_sha256": ckpt_sha,
            "robot_mass_kg": ROBOT_MASS_KG,
            "robot_weight_n": round(ROBOT_WEIGHT_N, 2),
            "split": args.split,
            "sequence_filter": args.sequence_filter,
            "num_clips": num_envs,
            "categories": categories_report,
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[INFO] Saved contact force profile to {args.report_json}", flush=True)
        for cat, info in categories_report.items():
            sf400 = info["single_foot_400hz_peak"]
            sf50 = info["single_foot_50hz_mean"]
            non_foot_active = [
                f"{g}:{info['body_groups_400hz_peak'][g]['contact_ratio']*100:.1f}% (P99={info['body_groups_400hz_peak'][g]['p99_n']:.0f}N, max={info['body_groups_400hz_peak'][g]['max_n']:.0f}N)"
                for g in ("knees", "hips_pelvis", "hands_wrists", "elbows", "shoulders", "torso", "head")
                if info["body_groups_400hz_peak"][g]["contact_ratio"] >= 0.005
            ]
            nf_str = ", ".join(non_foot_active) if non_foot_active else "none (<0.5%)"
            print(
                f"[{cat:15s}] steps={info['alive_steps_sampled']:6d} | "
                f"1-Foot 50Hz: mean={sf50['active_mean_n']:5.0f}N P99={sf50['p99_n']:6.0f}N | "
                f"1-Foot 400Hz: P95={sf400['p95_n']:6.0f}N P99={sf400['p99_n']:6.0f}N ({sf400['p99_bw']:.2f}xBW) max={sf400['max_n']:6.0f}N | "
                f"Non-foot contacts: {nf_str}",
                flush=True,
            )
    finally:
        if args.headless:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        simulation_app.close()


if __name__ == "__main__":
    main()
