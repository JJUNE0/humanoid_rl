"""Isaac Sim / IsaacLab USD Sim-to-Sim validation for Humanoid-Light SuperMimic policies.

Loads the original USD (`humanoid_light_rev_1_0_12_45_merge_joints.usd`) and `HUMANOID_LIGHT_CFG`
(`DelayedPDActuatorCfg` + `CoupledDelayedPDActuatorCfg`) in `/home/cocel/isaac/env_isaaclab`,
maps MuJoCo SuperMimic observations/actions & coupled-motor targets, and evaluates walking.

Tip: For browser-based Viser 3D visualization of Isaac Sim PhysX (lighter & faster than Omniverse GUI),
use `src/humanoid_rl/sim2sim/play_light_supermimic_isaac_viser.py`!

Examples:
  # 1) Interactive Isaac Sim GUI viewer (loops continuously until closed):
  /home/cocel/isaac/env_isaaclab/bin/python src/humanoid_rl/sim2sim/validate_light_supermimic_isaac.py

  # 2) Quantitative headless Sim-to-Sim validation across all walk1 clips (train or held-out validation):
  /home/cocel/isaac/env_isaaclab/bin/python src/humanoid_rl/sim2sim/validate_light_supermimic_isaac.py \
      --headless --all-clips --steps 500 --split train \
      --report-json outputs/sim2sim/isaac_walk1_train_report.json

  # 3) Record MP4 video headlessly in Isaac Sim with tracking camera:
  /home/cocel/isaac/env_isaaclab/bin/python src/humanoid_rl/sim2sim/validate_light_supermimic_isaac.py \
      --headless --enable_cameras --steps 500 \
      --video outputs/sim2sim/isaac_walk1_clip0.mp4
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

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
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "runs/light_supermimic_walk1_t4/t4_walk1_fixed_20261118/training"
)
DEFAULT_MANIFEST = REPO_ROOT / "src/humanoid_retarget/configs/motions/light_supermimic_walk1.json"


def build_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="SuperMimic checkpoint dir, training dir (with latest.json), or last.pt.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional motion manifest JSON override (defaults to checkpoint's manifest mapped locally).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="train",
        help="Evaluate on train (walk1_subject1) or held-out validation (walk1_subject5) clips.",
    )
    parser.add_argument(
        "--clip-index",
        type=int,
        default=0,
        help="Clip index when running a single environment (default: 0).",
    )
    parser.add_argument(
        "--all-clips",
        action="store_true",
        help="Spawn one parallel Isaac env per clip in the selected split for full quantitative evaluation.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of 50 Hz policy steps (default: 0 = infinite loop in GUI, 500 = 10.0s in --headless).",
    )
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=0.0025,
        help="Isaac PhysX simulation timestep in seconds (default: 0.0025s = 400Hz; cocelo default is 0.005s).",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=8,
        help="Physics substeps per 50 Hz policy step (default: 8 for sim-dt=0.0025; use 4 for sim-dt=0.005).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional output MP4 path (automatically enables RTX camera rendering).",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional JSON file path to save quantitative Sim-to-Sim validation metrics.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def resolve_usd_path() -> Path:
    for candidate in USD_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find Humanoid-Light USD in {USD_CANDIDATES}")


def resolve_local_manifest(saved_settings: dict, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    orig_root = Path(saved_settings["motion"]["root"])
    orig_manifest = Path(saved_settings["motion"]["manifest"])
    if orig_manifest.is_file():
        return orig_manifest
    try:
        rel = orig_manifest.relative_to(orig_root)
    except ValueError:
        rel = Path("src/humanoid_retarget/configs/motions/light_supermimic_walk1.json")
    for candidate in (REPO_ROOT / rel, REPO_ROOT / "src/humanoid_retarget" / rel):
        if candidate.is_file():
            return candidate
    return DEFAULT_MANIFEST


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 500 if (args.headless or args.video is not None) else 0
    if abs(args.sim_dt * args.decimation - 0.02) > 1e-6:
        raise SystemExit(
            f"sim_dt ({args.sim_dt}) * decimation ({args.decimation}) must equal 0.02s (50 Hz)"
        )
    if args.video is not None:
        args.enable_cameras = True

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
    from isaaclab.sensors import CameraCfg
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

    device = args.device if getattr(args, "device", None) else "cuda:0"
    saved, ckpt_file, ckpt_sha = load_checkpoint_compat(args.checkpoint, device=device)
    env_settings = saved["settings"]["environment"]
    manifest_path = resolve_local_manifest(saved["settings"], args.manifest)
    chunk_frames = int(saved["settings"]["motion"].get("chunk_frames", 501))

    packed_dir = pack_manifest_pure_numpy(
        manifest_path, repo_root=REPO_ROOT, split=args.split, chunk_frames=chunk_frames
    )
    motion = PackedMotionLibrary(packed_dir, device=device)
    num_envs = motion.num_clips if args.all_clips else 1
    usd_path = resolve_usd_path()

    print(f"[INFO] Loaded checkpoint: {ckpt_file} (sha256={ckpt_sha})", flush=True)
    print(f"[INFO] Loaded USD: {usd_path}", flush=True)
    print(
        f"[INFO] Split={args.split} clips={motion.num_clips} frames={motion.time_step_total} num_envs={num_envs}",
        flush=True,
    )

    robot_cfg: ArticulationCfg = deepcopy(HUMANOID_LIGHT_CFG).replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    robot_cfg.spawn.usd_path = str(usd_path)

    @configclass
    class Sim2SimSceneCfg(InteractiveSceneCfg):
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
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.8, 0.8, 0.8), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.35, 0.40, 0.48), intensity=1200.0),
        )

    if args.video is not None:
        Sim2SimSceneCfg.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.02,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(2.3, -2.3, 1.3),
                rot=(0.8804762, 0.1159169, 0.2798481, 0.3647052),
                convention="world",
            ),
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
    if not args.headless:
        sim.set_camera_view(eye=[2.5, -2.5, 1.35], target=[0.0, 0.0, 0.75])

    scene_cfg = Sim2SimSceneCfg(num_envs=num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    isaac_joint_ids_list, matched_names = robot.find_joints(
        list(JOINT_NAMES), preserve_order=True
    )
    if tuple(matched_names) != JOINT_NAMES:
        raise RuntimeError(f"Joint mapping mismatch: {matched_names} != {JOINT_NAMES}")
    isaac_joint_ids = torch.tensor(isaac_joint_ids_list, dtype=torch.long, device=device)

    fall_check_bodies = ["torso_roll_link", "head_link", "left_knee_link", "right_knee_link"]
    fall_body_ids_list, _ = robot.find_bodies(fall_check_bodies, preserve_order=True)
    torso_id, head_id, l_knee_id, r_knee_id = fall_body_ids_list

    actor = build_standalone_actor_from_saved(saved, device=device)
    scale_tensor = compute_coupled_action_scale_tensor(
        float(env_settings.get("action_scale", 0.25)), device=device
    )
    limits_tensor = parse_mjcf_joint_limits(device=device)

    if args.all_clips:
        clip_indices = list(range(motion.num_clips))
    else:
        if not (0 <= args.clip_index < motion.num_clips):
            raise ValueError(f"--clip-index {args.clip_index} out of [0, {motion.num_clips})")
        clip_indices = [args.clip_index]

    clip_starts = torch.tensor(
        [motion.clip_bounds(c)[0] for c in clip_indices], dtype=torch.long, device=device
    )
    clip_ends = torch.tensor(
        [motion.clip_bounds(c)[1] for c in clip_indices], dtype=torch.long, device=device
    )
    clip_lengths = clip_ends - clip_starts

    def reset_envs(env_ids: torch.Tensor) -> None:
        frames = clip_starts[env_ids]
        root_pos = motion.root_pos_w[frames] + scene.env_origins[env_ids]
        root_quat = motion.root_quat_w[frames]
        root_pose = torch.cat([root_pos, root_quat], dim=-1)
        root_vel = torch.cat(
            [motion.root_lin_vel_w[frames], motion.root_ang_vel_w[frames]], dim=-1
        )
        robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        robot.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

        q_isaac = robot.data.default_joint_pos[env_ids].clone()
        dq_isaac = robot.data.default_joint_vel[env_ids].clone()
        q_isaac[:, isaac_joint_ids] = motion.joint_pos[frames]
        dq_isaac[:, isaac_joint_ids] = motion.joint_vel[frames]
        robot.write_joint_state_to_sim(q_isaac, dq_isaac, env_ids=env_ids)
        robot.reset(env_ids=env_ids)

    all_env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    motion_times = clip_starts.clone()
    reset_envs(all_env_ids)
    scene.write_data_to_sim()
    sim.forward()
    is_rendering = sim.has_gui() or sim.has_rtx_sensors()
    if is_rendering:
        sim.render()
    scene.update(args.sim_dt)

    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    survived_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
    completed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    joint_sq_err_sum = torch.zeros(num_envs, device=device)
    leg_sq_err_sum = torch.zeros(num_envs, device=device)
    root_xy_err_sum = torch.zeros(num_envs, device=device)
    root_z_err_sum = torch.zeros(num_envs, device=device)
    frames_rgb: list[np.ndarray] = []

    world_down = torch.tensor([[0.0, 0.0, -1.0]], device=device).expand(num_envs, 3)
    cam_offset = torch.tensor([[2.2, -2.2, 0.55]], device=device).expand(num_envs, 3)

    sensor_tracker = SensorHistoryTracker(saved, num_envs=num_envs, device=device)
    step_idx = 0
    infinite = args.steps == 0 and not args.headless
    try:
        with torch.no_grad():
            while simulation_app.is_running() and (infinite or step_idx < args.steps):
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
                isaac_slot_targets_mj = joint_targets_mj_to_isaac_actuator_slots(
                    q_target_mj
                )

                isaac_targets = robot.data.default_joint_pos.clone()
                isaac_targets[:, isaac_joint_ids] = isaac_slot_targets_mj
                robot.set_joint_position_target(isaac_targets)

                # CRITICAL: Always call sim.step(render=False) per physics substep, and
                # call sim.render() once after the final substep (matching ManagerBasedRLEnv).
                # Calling sim.step(render=True) steps an entire rendering_dt (0.02s) at once!
                for substep in range(args.decimation):
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    if substep == args.decimation - 1 and is_rendering:
                        if not args.headless and num_envs == 1:
                            pelvis_w = robot.data.root_link_pos_w[0].detach().cpu().numpy()
                            sim.set_camera_view(
                                eye=[float(pelvis_w[0] + 2.3), float(pelvis_w[1] - 2.3), 1.3],
                                target=[float(pelvis_w[0]), float(pelvis_w[1]), 0.75],
                            )
                        if args.video is not None and "camera" in scene.sensors:
                            pelvis_all = robot.data.root_link_pos_w
                            targets = torch.stack(
                                [pelvis_all[:, 0], pelvis_all[:, 1], torch.full_like(pelvis_all[:, 2], 0.75)],
                                dim=-1,
                            )
                            eyes = targets + cam_offset
                            scene["camera"].set_world_poses_from_view(eyes=eyes, targets=targets)
                        sim.render()
                    scene.update(args.sim_dt)

                if args.video is not None and "camera" in scene.sensors:
                    rgb = scene["camera"].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                    if rgb.dtype != np.uint8:
                        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                    frames_rgb.append(rgb)

                # Evaluate post-step tracking & strict 3D fall detection
                post_root_pos = robot.data.root_link_pos_w - scene.env_origins
                post_root_quat = robot.data.root_link_quat_w
                post_qj = robot.data.joint_pos[:, isaac_joint_ids]
                body_z = robot.data.body_link_pos_w[..., 2]

                ref_root_pos = motion.root_pos_w[motion_times]
                ref_root_quat = motion.root_quat_w[motion_times]
                ref_qj = motion.joint_pos[motion_times]

                height_err = (post_root_pos[:, 2] - ref_root_pos[:, 2]).abs()
                xy_err = torch.linalg.vector_norm(
                    post_root_pos[:, :2] - ref_root_pos[:, :2], dim=-1
                )
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

                next_times = motion_times + 1
                at_clip_end = next_times >= clip_ends

                if infinite:
                    motion_times = torch.where(at_clip_end, clip_starts, next_times)
                    reset_mask = bad_fall | at_clip_end
                    if reset_mask.any():
                        r_ids = reset_mask.nonzero(as_tuple=False).reshape(-1)
                        motion_times[r_ids] = clip_starts[r_ids]
                        reset_envs(r_ids)
                        sensor_tracker.reset(r_ids)
                else:
                    survived_steps[alive] += 1
                    sq_diff = (post_qj[alive] - ref_qj[alive]) ** 2
                    joint_sq_err_sum[alive] += sq_diff.mean(dim=-1)
                    leg_sq_err_sum[alive] += sq_diff[:, :12].mean(dim=-1)
                    root_xy_err_sum[alive] += xy_err[alive]
                    root_z_err_sum[alive] += height_err[alive]

                    newly_fell = alive & bad_fall
                    newly_done_clean = alive & ~bad_fall & at_clip_end
                    fell |= newly_fell
                    completed |= newly_done_clean
                    alive &= ~(bad_fall | at_clip_end)

                    motion_times = torch.where(at_clip_end, clip_ends - 1, next_times)
                    if not alive.any():
                        step_idx += 1
                        break

                step_idx += 1

        completed |= alive

        if args.video is not None and frames_rgb:
            import imageio.v2 as imageio

            args.video.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(args.video), frames_rgb, fps=50)
            print(f"[INFO] Saved Isaac Sim rollout video: {args.video}", flush=True)

        if not infinite:
            denom = survived_steps.clamp(min=1).float()
            joint_rmse_rad = torch.sqrt(joint_sq_err_sum / denom)
            leg_rmse_rad = torch.sqrt(leg_sq_err_sum / denom)
            mean_root_xy_err = root_xy_err_sum / denom
            mean_root_z_err = root_z_err_sum / denom
            max_possible_steps = torch.minimum(
                clip_lengths.clamp(min=1),
                torch.full_like(clip_lengths, max(args.steps, 1)),
            ).float()
            summary = {
                "simulator": "Isaac Sim 5.1 / IsaacLab v2.3.0 PhysX",
                "usd_path": str(usd_path),
                "checkpoint": str(ckpt_file),
                "checkpoint_sha256": ckpt_sha,
                "split": args.split,
                "sim_dt": args.sim_dt,
                "decimation": args.decimation,
                "num_clips_evaluated": num_envs,
                "completed_clips": int(completed.sum().item()),
                "completion_rate": round(float(completed.float().mean().item()), 4),
                "fall_rate": round(float(fell.float().mean().item()), 4),
                "mean_survived_steps": round(float(survived_steps.float().mean().item()), 2),
                "mean_survival_ratio": round(
                    float((survived_steps.float() / max_possible_steps).mean().item()),
                    4,
                ),
                "mean_leg_joint_rmse_deg": round(
                    float(leg_rmse_rad.mean().item() * 180.0 / np.pi), 2
                ),
                "mean_joint_pos_rmse_rad": round(float(joint_rmse_rad.mean().item()), 4),
                "mean_joint_pos_rmse_deg": round(
                    float(joint_rmse_rad.mean().item() * 180.0 / np.pi), 2
                ),
                "mean_root_xy_err_m": round(float(mean_root_xy_err.mean().item()), 4),
                "mean_root_height_err_m": round(float(mean_root_z_err.mean().item()), 4),
                "per_clip": [
                    {
                        "clip": motion.motion_names[clip_indices[i]],
                        "clip_frames": int(clip_lengths[i].item()),
                        "survived_steps": int(survived_steps[i].item()),
                        "completed": bool(completed[i].item()),
                        "fell": bool(fell[i].item()),
                        "leg_joint_rmse_deg": round(
                            float(leg_rmse_rad[i].item() * 180.0 / np.pi), 2
                        ),
                        "joint_rmse_deg": round(
                            float(joint_rmse_rad[i].item() * 180.0 / np.pi), 2
                        ),
                        "root_xy_err_m": round(float(mean_root_xy_err[i].item()), 4),
                        "root_z_err_m": round(float(mean_root_z_err[i].item()), 4),
                    }
                    for i in range(num_envs)
                ],
            }
            print(json.dumps(summary, indent=2), flush=True)
            if args.report_json is not None:
                args.report_json.parent.mkdir(parents=True, exist_ok=True)
                args.report_json.write_text(json.dumps(summary, indent=2) + "\n")
                print(f"[INFO] Saved report JSON: {args.report_json}", flush=True)
    finally:
        if args.headless:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        simulation_app.close()


if __name__ == "__main__":
    main()
