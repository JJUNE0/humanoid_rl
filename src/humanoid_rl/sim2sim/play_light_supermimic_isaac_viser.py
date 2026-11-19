"""Live browser-based Viser 3D viewer for Isaac Sim 5.1 / IsaacLab USD Sim-to-Sim rollouts.

Runs Isaac Sim PhysX + `humanoid_light_rev_1_0_12_45_merge_joints.usd` + `HUMANOID_LIGHT_CFG`
(`DelayedPDActuator` + `CoupledDelayedPDActuator`) headlessly (no heavy Omniverse GUI window!)
and streams live 50 Hz USD state + contact forces into a lightweight Viser web server.

Works with EITHER Python interpreter:
  /home/cocel/isaac/env_isaaclab/bin/python src/humanoid_rl/sim2sim/play_light_supermimic_isaac_viser.py --port 8081
  .venv_supermimic_light/bin/python src/humanoid_rl/sim2sim/play_light_supermimic_isaac_viser.py --port 8081
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"

MJLAB_PYTHON = REPO_ROOT / ".venv_supermimic_light/bin/python"
ISAAC_PYTHON = Path("/home/cocel/isaac/env_isaaclab/bin/python")
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


def parse_common_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
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
        help="Optional motion manifest JSON override.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="train",
        help="Initial split: train (walk1_subject1) or held-out validation (walk1_subject5).",
    )
    parser.add_argument(
        "--clip-index",
        type=int,
        default=0,
        help="Initial motion clip index (default: 0).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Viser server bind host.")
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Viser server bind port (default: 8081, so 8080 MuJoCo Viser can run alongside).",
    )
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=0.0025,
        help="Isaac PhysX simulation timestep in seconds (default: 0.0025s = 400 Hz).",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=8,
        help="Physics substeps per 50 Hz policy step (default: 8).",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Start Viser playback paused on frame 0.",
    )
    parser.add_argument(
        "--no-ref-ghost",
        action="store_true",
        help="Disable green reference-motion ghost overlay initially.",
    )
    parser.add_argument(
        "--playlist",
        action="store_true",
        help="Enable sequential auto-advance across clips in manifest order.",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=None,
        help="Run N live Isaac->Viser steps and exit cleanly (for automated verification).",
    )
    parser.add_argument("--isaac-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--socket-port", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_known_args(argv)


def resolve_usd_path() -> Path:
    for candidate in USD_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find Humanoid-Light USD in {USD_CANDIDATES}")


def resolve_local_manifest(saved_settings: dict[str, Any], override: Path | None) -> Path:
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


# ===========================================================================
# Child Process: Headless Isaac Sim 5.1 / IsaacLab PhysX USD Worker
# ===========================================================================


def run_isaac_worker(args: argparse.Namespace, extra_argv: list[str]) -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args = launcher_parser.parse_args(["--headless", *extra_argv])
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    if str(COCELO_ISAACLAB_ROOT) not in sys.path:
        sys.path.insert(0, str(COCELO_ISAACLAB_ROOT))

    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
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

    device = "cuda:0"
    saved, ckpt_file, ckpt_sha = load_checkpoint_compat(args.checkpoint, device=device)
    env_settings = saved["settings"]["environment"]
    manifest_path = resolve_local_manifest(saved["settings"], args.manifest)
    chunk_frames = int(saved["settings"]["motion"].get("chunk_frames", 501))

    motions: dict[str, PackedMotionLibrary] = {}
    for sp in ("train", "validation"):
        packed_dir = pack_manifest_pure_numpy(
            manifest_path, repo_root=REPO_ROOT, split=sp, chunk_frames=chunk_frames
        )
        motions[sp] = PackedMotionLibrary(packed_dir, device=device)

    raw_manifest = json.loads(manifest_path.read_text())
    playlists: dict[str, list[tuple[str, int]]] = {}
    has_any_showcase = False
    for sp in ("train", "validation"):
        m_names = list(motions[sp].motion_names)
        raw_recs = raw_manifest.get("splits", {}).get(sp, [])
        has_showcase = any("showcase_label" in r for r in raw_recs)
        has_any_showcase = has_any_showcase or has_showcase
        name_to_indices: dict[str, list[int]] = {}
        for p_idx, m_name in enumerate(m_names):
            name_to_indices.setdefault(m_name, []).append(p_idx)
        items: list[tuple[str, int]] = []
        if has_showcase:
            for rec in raw_recs:
                stem = Path(rec["path"]).stem
                start, stop = int(rec["start_frame"]), int(rec["stop_frame"])
                label = rec.get("showcase_label", "")
                for first in range(start, stop, chunk_frames):
                    last = min(first + chunk_frames, stop)
                    if last - first >= 2:
                        key = f"{rec['source']}/{stem}/{first}-{last}"
                        if key in name_to_indices and name_to_indices[key]:
                            p_idx = name_to_indices[key].pop(0)
                            pos = len(items)
                            title = f"{pos:02d}: {label} [{key}]" if label else f"{pos:02d}: {key}"
                            items.append((title, p_idx))
        if not items:
            items = [(f"{i}: {n}", i) for i, n in enumerate(m_names)]
        playlists[sp] = items

    usd_path = resolve_usd_path()
    robot_cfg: ArticulationCfg = copy.deepcopy(HUMANOID_LIGHT_CFG).replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    robot_cfg.spawn.usd_path = str(usd_path)

    @configclass
    class WorkerSceneCfg(InteractiveSceneCfg):
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
    scene = InteractiveScene(WorkerSceneCfg(num_envs=1, env_spacing=3.0))
    sim.reset()

    robot = scene["robot"]
    contacts = scene["contact_forces"]
    isaac_joint_ids_list, matched_names = robot.find_joints(
        list(JOINT_NAMES), preserve_order=True
    )
    if tuple(matched_names) != JOINT_NAMES:
        raise RuntimeError(f"Joint mismatch: {matched_names}")
    isaac_joint_ids = torch.tensor(isaac_joint_ids_list, dtype=torch.long, device=device)

    l_foot_ids, _ = contacts.find_bodies(["left_ankle_roll_link"], preserve_order=True)
    r_foot_ids, _ = contacts.find_bodies(["right_ankle_roll_link"], preserve_order=True)
    l_foot_id, r_foot_id = l_foot_ids[0], r_foot_ids[0]

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
    env0 = torch.tensor([0], dtype=torch.long, device=device)
    world_down = torch.tensor([[0.0, 0.0, -1.0]], device=device)

    sensor_tracker = SensorHistoryTracker(saved, num_envs=1, device=device)
    cur_split = args.split
    if args.clip_index == 0 and playlists[cur_split]:
        cur_clip = playlists[cur_split][0][1]
    else:
        cur_clip = max(0, min(args.clip_index, motions[cur_split].num_clips - 1))
    motion_time = torch.tensor([0], dtype=torch.long, device=device)
    step_in_ep = 0

    def reset_to_clip(split_name: str, clip_idx: int) -> dict[str, Any]:
        nonlocal cur_split, cur_clip, step_in_ep
        cur_split = split_name
        mot = motions[cur_split]
        cur_clip = max(0, min(int(clip_idx), mot.num_clips - 1))
        c_start, _ = mot.clip_bounds(cur_clip)
        motion_time[0] = c_start
        step_in_ep = 0
        sensor_tracker.reset()

        root_pos = mot.root_pos_w[motion_time] + scene.env_origins
        root_quat = mot.root_quat_w[motion_time]
        root_pose = torch.cat([root_pos, root_quat], dim=-1)
        root_vel = torch.cat(
            [mot.root_lin_vel_w[motion_time], mot.root_ang_vel_w[motion_time]], dim=-1
        )
        robot.write_root_link_pose_to_sim(root_pose, env_ids=env0)
        robot.write_root_link_velocity_to_sim(root_vel, env_ids=env0)

        q_isaac = robot.data.default_joint_pos.clone()
        dq_isaac = robot.data.default_joint_vel.clone()
        q_isaac[:, isaac_joint_ids] = mot.joint_pos[motion_time]
        dq_isaac[:, isaac_joint_ids] = mot.joint_vel[motion_time]
        robot.write_joint_state_to_sim(q_isaac, dq_isaac, env_ids=env0)
        robot.reset(env_ids=env0)

        scene.write_data_to_sim()
        sim.forward()
        scene.update(args.sim_dt)
        return pack_frame_payload(fell_flag=False)

    def pack_frame_payload(fell_flag: bool) -> dict[str, Any]:
        mot = motions[cur_split]
        c_start, c_end = mot.clip_bounds(cur_clip)
        r_pos = (robot.data.root_link_pos_w[0] - scene.env_origins[0]).detach().cpu().tolist()
        r_quat = robot.data.root_link_quat_w[0].detach().cpu().tolist()
        qj = robot.data.joint_pos[0, isaac_joint_ids].detach().cpu().tolist()

        ref_pos = mot.root_pos_w[motion_time[0]].detach().cpu().tolist()
        ref_quat = mot.root_quat_w[motion_time[0]].detach().cpu().tolist()
        ref_qj = mot.joint_pos[motion_time[0]].detach().cpu().tolist()

        leg_rmse_deg = float(
            np.sqrt(np.mean((np.array(qj[:12]) - np.array(ref_qj[:12])) ** 2))
            * 180.0
            / np.pi
        )
        f_norm = contacts.data.net_forces_w[0].norm(dim=-1)
        f_l = float(f_norm[l_foot_id].item())
        f_r = float(f_norm[r_foot_id].item())

        return {
            "type": "state",
            "split": cur_split,
            "clip_index": cur_clip,
            "clip_name": mot.motion_names[cur_clip],
            "frame_in_clip": int(motion_time[0].item() - c_start),
            "clip_frames": int(c_end - c_start),
            "step_in_ep": step_in_ep,
            "qpos_isaac": r_pos + r_quat + qj,
            "qpos_ref": ref_pos + ref_quat + ref_qj,
            "z_isaac": round(r_pos[2], 4),
            "z_ref": round(ref_pos[2], 4),
            "leg_rmse_deg": round(leg_rmse_deg, 2),
            "contact_l_n": round(f_l, 1),
            "contact_r_n": round(f_r, 1),
            "fell": fell_flag,
        }

    def step_once(auto_reset: bool = True, next_clip_on_end: int | None = None) -> dict[str, Any]:
        nonlocal step_in_ep
        mot = motions[cur_split]
        c_start, c_end = mot.clip_bounds(cur_clip)

        with torch.no_grad():
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
                mot,
                motion_time,
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
            isaac_slot_targets_mj = joint_targets_mj_to_isaac_actuator_slots(q_target_mj)
            isaac_targets = robot.data.default_joint_pos.clone()
            isaac_targets[:, isaac_joint_ids] = isaac_slot_targets_mj
            robot.set_joint_position_target(isaac_targets)

            for _ in range(args.decimation):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(args.sim_dt)

            step_in_ep += 1
            post_pos = robot.data.root_link_pos_w - scene.env_origins
            post_quat = robot.data.root_link_quat_w
            body_z = robot.data.body_link_pos_w[..., 2]
            ref_pos = mot.root_pos_w[motion_time]
            ref_quat = mot.root_quat_w[motion_time]

            height_err = (post_pos[:, 2] - ref_pos[:, 2]).abs()
            g_rob = quat_apply_inverse(post_quat, world_down)
            g_ref = quat_apply_inverse(ref_quat, world_down)
            grav_err = torch.linalg.vector_norm(g_rob - g_ref, dim=-1)
            ref_upright = (ref_pos[:, 2] > 0.58) & (g_ref[:, 2] < -0.75)
            low_body = ref_upright & (
                (post_pos[:, 2] < 0.45)
                | (body_z[:, torso_id] < 0.50)
                | (body_z[:, head_id] < 0.75)
                | (body_z[:, l_knee_id] < 0.12)
                | (body_z[:, r_knee_id] < 0.12)
            )
            fell_flag = bool(((height_err > 0.25) | (grav_err > 0.70) | low_body)[0].item())

            next_time = int(motion_time[0].item()) + 1
            at_end = next_time >= c_end
            if auto_reset and (fell_flag or at_end):
                target_clip = cur_clip if next_clip_on_end is None else int(next_clip_on_end)
                res = reset_to_clip(cur_split, target_clip)
                res["advanced_playlist"] = next_clip_on_end is not None
                return res
            motion_time[0] = min(next_time, c_end - 1)
            return pack_frame_payload(fell_flag=fell_flag)

    sock = socket.create_connection(("127.0.0.1", args.socket_port), timeout=30.0)
    rfile = sock.makefile("r", encoding="utf-8")
    wfile = sock.makefile("w", encoding="utf-8")

    def send_msg(payload: dict[str, Any]) -> None:
        wfile.write(json.dumps(payload) + "\n")
        wfile.flush()

    init_state = reset_to_clip(cur_split, cur_clip)
    send_msg(
        {
            "type": "hello",
            "checkpoint": str(ckpt_file),
            "checkpoint_sha256": ckpt_sha,
            "usd_path": str(usd_path),
            "clips": {
                sp: list(motions[sp].motion_names) for sp in ("train", "validation")
            },
            "playlists": {
                sp: [[title, idx] for title, idx in playlists[sp]]
                for sp in ("train", "validation")
            },
            "has_showcase_labels": has_any_showcase,
            "state": init_state,
        }
    )

    try:
        while simulation_app.is_running():
            line = rfile.readline()
            if not line:
                break
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "step":
                send_msg(
                    step_once(
                        auto_reset=bool(req.get("auto_reset", True)),
                        next_clip_on_end=req.get("next_clip_on_end"),
                    )
                )
            elif cmd == "reset":
                send_msg(
                    reset_to_clip(
                        req.get("split", cur_split),
                        int(req.get("clip_index", cur_clip)),
                    )
                )
            elif cmd == "quit":
                break
    finally:
        try:
            sock.close()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


# ===========================================================================
# Parent Process: Lightweight Viser 3D Web Server (.venv_supermimic_light)
# ===========================================================================


def run_viser_frontend(args: argparse.Namespace, extra_argv: list[str]) -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    import mujoco
    import numpy as np
    import viser
    from mjlab.viewer.viser.scene import MjlabViserScene
    from robots.humanoid.humanoid_light.mjlab import get_robot_spec, load_control

    if not ISAAC_PYTHON.is_file():
        raise FileNotFoundError(f"IsaacLab python not found at {ISAAC_PYTHON}")

    # 1. Start local TCP bridge & launch headless Isaac Sim PhysX worker
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind(("127.0.0.1", 0))
    listen_sock.listen(1)
    bridge_port = listen_sock.getsockname()[1]

    worker_cmd = [
        str(ISAAC_PYTHON),
        str(Path(__file__).resolve()),
        "--isaac-worker",
        "--socket-port",
        str(bridge_port),
        "--checkpoint",
        str(args.checkpoint),
        "--split",
        args.split,
        "--clip-index",
        str(args.clip_index),
        "--sim-dt",
        str(args.sim_dt),
        "--decimation",
        str(args.decimation),
    ]
    if args.manifest is not None:
        worker_cmd += ["--manifest", str(args.manifest)]
    if args.playlist:
        worker_cmd += ["--playlist"]
    worker_cmd += extra_argv

    print(
        "[INFO] Starting headless Isaac Sim 5.1 PhysX USD worker (no Omniverse GUI overhead)...",
        flush=True,
    )
    worker_proc = subprocess.Popen(worker_cmd, cwd=str(REPO_ROOT))

    def cleanup_worker(*_sig_args: Any) -> None:
        if worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                worker_proc.kill()

    signal.signal(signal.SIGTERM, lambda s, f: (cleanup_worker(), sys.exit(0)))

    try:
        listen_sock.settimeout(90.0)
        conn, _ = listen_sock.accept()
        listen_sock.close()
        rfile = conn.makefile("r", encoding="utf-8")
        wfile = conn.makefile("w", encoding="utf-8")
        io_lock = threading.Lock()

        def rpc(req: dict[str, Any]) -> dict[str, Any]:
            with io_lock:
                wfile.write(json.dumps(req) + "\n")
                wfile.flush()
                line = rfile.readline()
                if not line:
                    raise RuntimeError("Isaac Sim PhysX worker closed connection unexpectedly.")
                return json.loads(line)

        hello_line = rfile.readline()
        if not hello_line:
            raise RuntimeError("Isaac Sim PhysX worker failed before handshake.")
        hello = json.loads(hello_line)
        clips_by_split: dict[str, list[str]] = hello["clips"]
        raw_playlists = hello.get("playlists") or {
            sp: [[f"{i}: {n}", i] for i, n in enumerate(clips_by_split[sp])]
            for sp in ("train", "validation")
        }
        playlists_by_split: dict[str, list[tuple[str, int]]] = {
            sp: [(str(title), int(idx)) for title, idx in items]
            for sp, items in raw_playlists.items()
        }
        has_showcase_labels = bool(hello.get("has_showcase_labels", False))
        cur_state: dict[str, Any] = hello["state"]

        print(f"[INFO] Connected to Isaac Sim PhysX USD: {hello['usd_path']}", flush=True)
        print(
            f"[INFO] Checkpoint: {hello['checkpoint']} (sha256={hello['checkpoint_sha256']})",
            flush=True,
        )

        # 2. Build CPU-only MuJoCo visual model + green reference ghost for Viser
        spec = get_robot_spec(load_control())
        spec.worldbody.add_geom(
            name="terrain",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0.0, 0.0, 0.05],
            rgba=[0.20, 0.22, 0.25, 1.0],
        )
        mj_model = spec.compile()
        mj_data = mujoco.MjData(mj_model)

        ref_ghost_model = copy.deepcopy(mj_model)
        ref_ghost_model.geom_rgba[:] = [0.20, 0.85, 0.38, 0.42]
        terrain_gid = mujoco.mj_name2id(
            ref_ghost_model, mujoco.mjtObj.mjOBJ_GEOM, "terrain"
        )
        if terrain_gid >= 0:
            ref_ghost_model.geom_rgba[terrain_gid, 3] = 0.0

        # 3. Start Viser server & UI
        server = viser.ViserServer(
            host=args.host,
            port=args.port,
            label="Isaac Sim 5.1 USD Sim2Sim",
        )
        scene = MjlabViserScene(server=server, mj_model=mj_model, num_envs=1)
        scene.debug_visualization_enabled = True

        is_paused = bool(args.paused)
        show_ghost = not args.no_ref_ghost
        auto_reset = True
        auto_playlist = bool(args.playlist or has_showcase_labels)
        suppress_clip_cb = False
        pending_step = False
        pending_reset: tuple[str, int] | None = None

        init_list = playlists_by_split[args.split]
        playlist_pos = 0
        for p_pos, (_, p_idx) in enumerate(init_list):
            if p_idx == int(cur_state["clip_index"]):
                playlist_pos = p_pos
                break

        with server.gui.add_folder("Isaac Sim 5.1 USD Status"):
            status_md = server.gui.add_markdown("")

        with server.gui.add_folder("Playback & Clip Controls"):
            pause_btn = server.gui.add_button(
                "Play" if is_paused else "Pause",
                icon=viser.Icon.PLAYER_PLAY if is_paused else viser.Icon.PLAYER_PAUSE,
            )
            step_btn = server.gui.add_button("Step Once", icon=viser.Icon.PLAYER_TRACK_NEXT)
            reset_btn = server.gui.add_button("Reset Clip")
            prev_btn = server.gui.add_button("Prev Clip")
            next_btn = server.gui.add_button("Next Clip")
            speed_slider = server.gui.add_slider(
                "Playback Speed", min=0.2, max=2.0, step=0.1, initial_value=1.0
            )
            ghost_cb = server.gui.add_checkbox(
                "Green Reference Ghost", initial_value=show_ghost
            )
            autoreset_cb = server.gui.add_checkbox(
                "Auto-Reset on Clip End / Fall", initial_value=auto_reset
            )
            playlist_cb = server.gui.add_checkbox(
                "Auto-Advance Playlist", initial_value=auto_playlist
            )

            split_dd = server.gui.add_dropdown(
                "Split",
                options=["train", "validation"],
                initial_value=args.split,
            )

            init_opts = [t for t, _ in init_list]
            clip_dd = server.gui.add_dropdown(
                "Showcase Clip" if has_showcase_labels else "Motion Clip",
                options=init_opts,
                initial_value=init_opts[playlist_pos],
            )

        scene.create_scene_gui(camera_distance=2.8, camera_azimuth=135.0, camera_elevation=18.0)

        def current_playlist() -> list[tuple[str, int]]:
            return playlists_by_split[split_dd.value]

        def step_playlist(delta: int) -> None:
            nonlocal playlist_pos, pending_reset, suppress_clip_cb
            cur_list = current_playlist()
            playlist_pos = (playlist_pos + delta) % len(cur_list)
            suppress_clip_cb = True
            clip_dd.value = cur_list[playlist_pos][0]
            suppress_clip_cb = False
            pending_reset = (split_dd.value, cur_list[playlist_pos][1])

        @pause_btn.on_click
        def _(_) -> None:
            nonlocal is_paused
            is_paused = not is_paused
            pause_btn.label = "Play" if is_paused else "Pause"

        @step_btn.on_click
        def _(_) -> None:
            nonlocal pending_step
            pending_step = True

        @reset_btn.on_click
        def _(_) -> None:
            nonlocal pending_reset
            cur_list = current_playlist()
            pending_reset = (split_dd.value, cur_list[playlist_pos][1])

        @prev_btn.on_click
        def _(_) -> None:
            step_playlist(-1)

        @next_btn.on_click
        def _(_) -> None:
            step_playlist(1)

        @ghost_cb.on_update
        def _(_) -> None:
            nonlocal show_ghost
            show_ghost = bool(ghost_cb.value)
            if not show_ghost:
                scene.clear_debug_all()

        @autoreset_cb.on_update
        def _(_) -> None:
            nonlocal auto_reset
            auto_reset = bool(autoreset_cb.value)

        @playlist_cb.on_update
        def _(_) -> None:
            nonlocal auto_playlist
            auto_playlist = bool(playlist_cb.value)

        @split_dd.on_update
        def _(_) -> None:
            nonlocal playlist_pos, pending_reset, suppress_clip_cb
            cur_list = current_playlist()
            playlist_pos = 0
            suppress_clip_cb = True
            clip_dd.options = [t for t, _ in cur_list]
            clip_dd.value = cur_list[0][0]
            suppress_clip_cb = False
            pending_reset = (split_dd.value, cur_list[0][1])

        @clip_dd.on_update
        def _(_) -> None:
            nonlocal playlist_pos, pending_reset
            if suppress_clip_cb:
                return
            cur_list = current_playlist()
            titles = [t for t, _ in cur_list]
            if clip_dd.value in titles:
                playlist_pos = titles.index(clip_dd.value)
                pending_reset = (split_dd.value, cur_list[playlist_pos][1])

        def push_state_to_viser(st: dict[str, Any]) -> None:
            mj_data.qpos[:] = np.asarray(st["qpos_isaac"], dtype=np.float64)
            mujoco.mj_forward(mj_model, mj_data)
            scene.clear()
            if show_ghost:
                scene.add_ghost_entity(
                    np.asarray(st["qpos_ref"], dtype=np.float64),
                    model=ref_ghost_model,
                    alpha=0.42,
                    label="ref_motion",
                )
            scene.update_from_mjdata(mj_data)
            scene._sync_debug_visualizations(np.zeros(3, dtype=np.float32))
            cur_list = current_playlist()
            cur_title = cur_list[playlist_pos][0] if 0 <= playlist_pos < len(cur_list) else st["clip_name"]
            status_md.content = (
                f"**Engine**: `Isaac Sim 5.1 PhysX (USD)`  \n"
                f"**Showcase**: `{cur_title}`  \n"
                f"**Clip**: `{st['clip_name']}` (`{st['frame_in_clip']}/{st['clip_frames']}`)  \n"
                f"**Pelvis Z**: `{st['z_isaac']:.3f} m` (ref `{st['z_ref']:.3f} m`)  \n"
                f"**12-Leg RMSE**: `{st['leg_rmse_deg']:.2f}°`  \n"
                f"**Foot Force (L / R)**: `{st['contact_l_n']:.0f} N` / `{st['contact_r_n']:.0f} N`"
            )

        push_state_to_viser(cur_state)
        print(
            f"[READY] Isaac Sim USD Viser live at http://localhost:{args.port}",
            flush=True,
        )

        executed_steps = 0
        while True:
            t0 = time.perf_counter()
            if worker_proc.poll() is not None:
                raise RuntimeError(
                    f"Isaac Sim worker exited with code {worker_proc.returncode}"
                )

            if pending_reset is not None:
                sp_req, c_req = pending_reset
                pending_reset = None
                cur_state = rpc({"cmd": "reset", "split": sp_req, "clip_index": c_req})
                push_state_to_viser(cur_state)
            elif not is_paused or pending_step:
                pending_step = False
                cur_list = current_playlist()
                next_pos = (playlist_pos + 1) % len(cur_list)
                next_clip_req = (
                    cur_list[next_pos][1]
                    if (auto_reset and auto_playlist and len(cur_list) > 1)
                    else None
                )
                cur_state = rpc(
                    {
                        "cmd": "step",
                        "auto_reset": auto_reset,
                        "next_clip_on_end": next_clip_req,
                    }
                )
                if cur_state.get("advanced_playlist"):
                    playlist_pos = next_pos
                    suppress_clip_cb = True
                    clip_dd.value = cur_list[playlist_pos][0]
                    suppress_clip_cb = False
                push_state_to_viser(cur_state)
                executed_steps += 1
                if args.smoke_steps is not None and executed_steps >= args.smoke_steps:
                    print(
                        json.dumps(
                            {
                                "status": "smoke_ok",
                                "executed_steps": executed_steps,
                                "last_state": {
                                    k: cur_state[k]
                                    for k in (
                                        "clip_name",
                                        "frame_in_clip",
                                        "z_isaac",
                                        "z_ref",
                                        "leg_rmse_deg",
                                        "contact_l_n",
                                        "contact_r_n",
                                        "fell",
                                    )
                                },
                            },
                            indent=2,
                        ),
                        flush=True,
                    )
                    with io_lock:
                        wfile.write(json.dumps({"cmd": "quit"}) + "\n")
                        wfile.flush()
                    server.stop()
                    break

            spd = max(float(speed_slider.value), 0.1)
            target_dt = 0.02 / spd
            sleep_s = target_dt - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        cleanup_worker()


def main() -> None:
    args, extra = parse_common_args()
    if args.isaac_worker:
        run_isaac_worker(args, extra)
        return

    # If invoked from env_isaaclab (no viser installed), transparently re-exec into .venv_supermimic_light
    try:
        import viser as _viser  # noqa: F401
    except ImportError:
        if not MJLAB_PYTHON.is_file():
            raise SystemExit(
                f"viser not installed in {sys.executable} and {MJLAB_PYTHON} not found."
            )
        os.execv(str(MJLAB_PYTHON), [str(MJLAB_PYTHON), *sys.argv])

    run_viser_frontend(args, extra)


if __name__ == "__main__":
    main()
