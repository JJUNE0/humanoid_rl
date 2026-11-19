"""Step-by-step diagnostic comparing Isaac Sim USD vs reference motion & MuJoCo torques on Clip 0.

Saves step-by-step trace JSON and tracking-camera PNG keyframes to pinpoint why Clip 0 falls.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

COCELO_ISAACLAB_ROOT = Path("/home/cocel/Desktop/Research/humanoid/cocelo-rl-isaaclab")
USD_PATH = (
    COCELO_ISAACLAB_ROOT
    / "lab/cocelo/assets/data/Robots/Flamingo/humanoid_light_rev_1_0_12_45/assets (2)/humanoid_light_rev_1_0_12_45_merge_joints.usd"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "runs/light_supermimic_walk1_t4/t4_walk1_fixed_20261118/training"
)
DEFAULT_MANIFEST = REPO_ROOT / "configs/motions/light_supermimic_walk1.json"


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--replicate-physics", type=int, default=1)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--sim-dt", type=float, default=0.0025)
    parser.add_argument("--decimation", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/sim2sim/debug_step0")
    parser.add_argument("--save-pngs", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.save_pngs:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app  # noqa: F841

    if str(COCELO_ISAACLAB_ROOT) not in sys.path:
        sys.path.insert(0, str(COCELO_ISAACLAB_ROOT))

    import imageio.v2 as imageio
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg, ContactSensorCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass

    from lab.cocelo.assets.flamingo.humanoid_light_v1 import HUMANOID_LIGHT_CFG
    from sim2sim_supermimic_adapter import (
        JOINT_NAMES,
        PackedMotionLibrary,
        build_motion_actor_goal,
        build_rep_state,
        build_standalone_actor_from_saved,
        compute_coupled_action_scale_tensor,
        joint_targets_mj_to_isaac_actuator_slots,
        load_checkpoint_compat,
        pack_manifest_pure_numpy,
        parse_mjcf_joint_limits,
        raw_action_to_joint_targets_mj,
    )

    device = "cuda:0"
    saved, ckpt_file, _ = load_checkpoint_compat(DEFAULT_CHECKPOINT, device=device)
    env_settings = saved["settings"]["environment"]
    packed_dir = pack_manifest_pure_numpy(
        DEFAULT_MANIFEST, repo_root=REPO_ROOT, split="train", chunk_frames=501
    )
    motion = PackedMotionLibrary(packed_dir, device=device)

    robot_cfg: ArticulationCfg = deepcopy(HUMANOID_LIGHT_CFG).replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    robot_cfg.spawn.usd_path = str(USD_PATH)

    init_root = motion.root_pos_w[0].detach().cpu().numpy()

    @configclass
    class DebugSceneCfg(InteractiveSceneCfg):
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
            history_length=1,
            track_air_time=False,
        )
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.8, 0.8, 0.8), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.4, 0.45, 0.5), intensity=1200.0),
        )

    if args.save_pngs:
        DebugSceneCfg.camera = CameraCfg(
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
                pos=(float(init_root[0] + 2.2), float(init_root[1] - 2.2), 1.25),
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

    scene_cfg = DebugSceneCfg(
        num_envs=args.num_envs,
        env_spacing=3.0,
        replicate_physics=bool(args.replicate_physics),
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    contacts = scene["contact_forces"]
    isaac_joint_ids_list, _ = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
    isaac_joint_ids = torch.tensor(isaac_joint_ids_list, dtype=torch.long, device=device)

    # Inspect PhysX properties on env 0
    pv = robot.root_physx_view
    props = {
        "num_envs": args.num_envs,
        "replicate_physics": bool(args.replicate_physics),
        "is_fixed_base": bool(robot.is_fixed_base),
        "dof_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "dof_stiffnesses": pv.get_dof_stiffnesses()[0].cpu().tolist(),
        "dof_dampings": pv.get_dof_dampings()[0].cpu().tolist(),
        "dof_armatures": pv.get_dof_armatures()[0].cpu().tolist(),
        "dof_friction_coeffs": robot.data.joint_friction_coeff[0].cpu().tolist(),
        "dof_max_forces": pv.get_dof_max_forces()[0].cpu().tolist(),
        "dof_max_velocities": pv.get_dof_max_velocities()[0].cpu().tolist(),
        "material_properties_shape": list(pv.get_material_properties().shape),
        "material_properties_sample": pv.get_material_properties()[0, :5].cpu().tolist(),
        "init_root_pos_w_motion0": init_root.tolist(),
        "env_origins": scene.env_origins.cpu().tolist(),
    }

    actor = build_standalone_actor_from_saved(saved, device=device)
    scale_tensor = compute_coupled_action_scale_tensor(
        float(env_settings.get("action_scale", 0.25)), device=device
    )
    limits_tensor = parse_mjcf_joint_limits(device=device)

    all_env_ids = torch.arange(args.num_envs, dtype=torch.long, device=device)
    # Put Clip 0 in env 0, and Clip 1..N-1 in other envs (same as --all-clips when num_envs=29)
    clip_starts = torch.tensor(
        [motion.clip_bounds(min(c, motion.num_clips - 1))[0] for c in range(args.num_envs)],
        dtype=torch.long,
        device=device,
    )
    motion_times = clip_starts.clone()

    root_pos = motion.root_pos_w[clip_starts] + scene.env_origins
    root_quat = motion.root_quat_w[clip_starts]
    root_pose = torch.cat([root_pos, root_quat], dim=-1)
    root_vel = torch.cat(
        [motion.root_lin_vel_w[clip_starts], motion.root_ang_vel_w[clip_starts]], dim=-1
    )
    robot.write_root_link_pose_to_sim(root_pose, env_ids=all_env_ids)
    robot.write_root_link_velocity_to_sim(root_vel, env_ids=all_env_ids)

    q_isaac = robot.data.default_joint_pos.clone()
    dq_isaac = robot.data.default_joint_vel.clone()
    q_isaac[:, isaac_joint_ids] = motion.joint_pos[clip_starts]
    dq_isaac[:, isaac_joint_ids] = motion.joint_vel[clip_starts]
    robot.write_joint_state_to_sim(q_isaac, dq_isaac, env_ids=all_env_ids)
    robot.reset(env_ids=all_env_ids)

    # IMPORTANT: Check what happens during scene.write_data_to_sim() BEFORE first action!
    scene.write_data_to_sim()
    sim.forward()
    scene.update(args.sim_dt)

    # Check FK of all bodies at frame 0 before any step!
    body_pos_0 = (robot.data.body_link_pos_w[0] - scene.env_origins[0]).cpu().tolist()
    body_quat_0 = robot.data.body_link_quat_w[0].cpu().tolist()
    props["frame0_body_pos_local"] = dict(zip(robot.body_names, body_pos_0))
    props["frame0_body_quat_w"] = dict(zip(robot.body_names, body_quat_0))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"n{args.num_envs}_rep{args.replicate_physics}"
    trace = []

    with torch.no_grad():
        for step_idx in range(args.steps):
            root_pos_local = robot.data.root_link_pos_w - scene.env_origins
            root_quat = robot.data.root_link_quat_w
            root_lin_vel = robot.data.root_link_lin_vel_w
            root_ang_vel = robot.data.root_link_ang_vel_w
            qj_mj = robot.data.joint_pos[:, isaac_joint_ids]
            vj_mj = robot.data.joint_vel[:, isaac_joint_ids]

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
            isaac_slot_targets_mj = joint_targets_mj_to_isaac_actuator_slots(q_target_mj)
            isaac_targets = robot.data.default_joint_pos.clone()
            isaac_targets[:, isaac_joint_ids] = isaac_slot_targets_mj
            robot.set_joint_position_target(isaac_targets)

            substep_torques = []
            for substep in range(args.decimation):
                scene.write_data_to_sim()
                if substep == 0:
                    substep_torques = (
                        robot.data.applied_torque[0, isaac_joint_ids].detach().cpu().tolist()
                    )
                should_render = args.save_pngs and (substep == args.decimation - 1)
                sim.step(render=should_render)
                scene.update(args.sim_dt)

            if args.save_pngs and step_idx in (0, 10, 20, 30, 40, 50, 70):
                rgb = scene["camera"].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                imageio.imwrite(str(args.out_dir / f"{tag}_step_{step_idx:03d}.png"), rgb)

            net_forces = contacts.data.net_forces_w[0].norm(dim=-1).detach().cpu().tolist()
            active_contacts = {
                b: round(f, 2)
                for b, f in zip(contacts.body_names, net_forces)
                if f > 0.5
            }

            post_pos = (robot.data.root_link_pos_w[0] - scene.env_origins[0]).cpu().tolist()
            post_quat = robot.data.root_link_quat_w[0].cpu().tolist()
            ref_pos = motion.root_pos_w[motion_times[0]].cpu().tolist()
            ref_quat = motion.root_quat_w[motion_times[0]].cpu().tolist()
            post_qj = robot.data.joint_pos[0, isaac_joint_ids].cpu().tolist()
            ref_qj = motion.joint_pos[motion_times[0]].cpu().tolist()
            max_j_err_idx = int(np.argmax(np.abs(np.array(post_qj) - np.array(ref_qj))))

            trace.append(
                {
                    "step": step_idx,
                    "root_pos": [round(x, 4) for x in post_pos],
                    "ref_pos": [round(x, 4) for x in ref_pos],
                    "root_quat": [round(x, 4) for x in post_quat],
                    "ref_quat": [round(x, 4) for x in ref_quat],
                    "max_joint_err_name": JOINT_NAMES[max_j_err_idx],
                    "max_joint_err_rad": round(abs(post_qj[max_j_err_idx] - ref_qj[max_j_err_idx]), 4),
                    "substep0_torque_sample": {
                        JOINT_NAMES[k]: round(substep_torques[k], 3)
                        for k in (0, 1, 3, 4, 5, 6, 9, 10, 11, 13, 14)
                    },
                    "active_contacts": active_contacts,
                }
            )
            if step_idx == 0:
                trace[-1]["step0_full"] = {
                    "qj_mj_init": [round(x, 5) for x in qj_mj[0].cpu().tolist()],
                    "vj_mj_init": [round(x, 5) for x in vj_mj[0].cpu().tolist()],
                    "raw_action": [round(x, 5) for x in raw_action[0].cpu().tolist()],
                    "q_target_mj": [round(x, 5) for x in q_target_mj[0].cpu().tolist()],
                    "substep0_torques_all": dict(
                        zip(JOINT_NAMES, [round(x, 4) for x in substep_torques])
                    ),
                }

            motion_times = motion_times + 1

    out_json = args.out_dir / f"trace_{tag}.json"
    out_json.write_text(json.dumps({"props": props, "trace": trace}, indent=2) + "\n")
    print(f"[DONE] Saved {out_json}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
