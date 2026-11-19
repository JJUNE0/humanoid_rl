"""Viser Video-to-Robot Dance Studio bridge: YouTube/MP4 -> GVHMR -> SMPL-X -> Humanoid-Light 50Hz + 1:1 Video Sync.

Pipeline order:
  1. Video acquisition & trim (`yt-dlp` for YouTube URLs + `/usr/bin/ffmpeg` segment trim & RGB frame decode)
  2. Monocular 3D human motion recovery (`GVHMR` -> `hmr4d_results.pt` containing `smpl_params_global` in SMPL-X)
     or direct load of pre-extracted `.pt` (`GVHMR`) / `.npz` (`SMPL-X` AMASS / FineDance)
  3. Kinematic retargeting (`SMPLX_NEUTRAL.npz` exact FK -> calibrated Humanoid-Light offsets -> 50 Hz Mink/DAQP IK)
  4. Live Viser hot-swap (`motion_cmd.motion` tensor append + 3D virtual studio screen & side-panel 1:1 frame sync)
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import traceback
from typing import Callable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (SRC_DIR, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_SMPLX_NPZ = REPO_ROOT / "data/body_models/smplx/SMPLX_NEUTRAL.npz"
STUDIO_OUTPUT_DIR = REPO_ROOT / "outputs/video_dance_studio"
RETARGET_DEPS_DIR = REPO_ROOT / "outputs/humanoid_light_retarget/python_deps"

# First 22 SMPL-X body joints in kinematic order (indices 0..21)
SMPLX_22_JOINTS: tuple[str, ...] = (
    "pelvis",          # 0
    "left_hip",        # 1
    "right_hip",       # 2
    "spine1",          # 3
    "left_knee",       # 4
    "right_knee",      # 5
    "spine2",          # 6
    "left_ankle",      # 7
    "right_ankle",     # 8
    "spine3",          # 9
    "left_foot",       # 10 (ball/toe joint in SMPL-X)
    "right_foot",      # 11 (ball/toe joint in SMPL-X)
    "neck",            # 12
    "left_collar",     # 13
    "right_collar",    # 14
    "head",            # 15
    "left_shoulder",   # 16 (glenohumeral joint)
    "right_shoulder",  # 17 (glenohumeral joint)
    "left_elbow",      # 18
    "right_elbow",     # 19
    "left_wrist",      # 20
    "right_wrist",     # 21
)

# 1-to-1 correspondence from canonical 22 LAFAN bone names (expected by retarget_humanoid_light_lafan)
# to SMPL-X 22 body joint names.
LAFAN_TO_SMPLX_22: tuple[tuple[str, str], ...] = (
    ("Hips", "pelvis"),
    ("LeftUpLeg", "left_hip"),
    ("LeftLeg", "left_knee"),
    ("LeftFoot", "left_ankle"),
    ("LeftToe", "left_foot"),
    ("RightUpLeg", "right_hip"),
    ("RightLeg", "right_knee"),
    ("RightFoot", "right_ankle"),
    ("RightToe", "right_foot"),
    ("Spine", "spine1"),
    ("Spine1", "spine2"),
    ("Spine2", "spine3"),
    ("Neck", "neck"),
    ("Head", "head"),
    ("LeftShoulder", "left_collar"),
    ("LeftArm", "left_shoulder"),
    ("LeftForeArm", "left_elbow"),
    ("LeftHand", "left_wrist"),
    ("RightShoulder", "right_collar"),
    ("RightArm", "right_shoulder"),
    ("RightForeArm", "right_elbow"),
    ("RightHand", "right_wrist"),
)

LAFAN_22_NAMES: list[str] = [lafan_name for lafan_name, _ in LAFAN_TO_SMPLX_22]
YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _rotate_quat_vec(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vectors `v` (N, 3) by scalar-first quaternions `q_wxyz` (N, 4)."""
    uv = 2.0 * np.cross(q_wxyz[:, 1:], v)
    return v + q_wxyz[:, :1] * uv + np.cross(q_wxyz[:, 1:], uv)


def smplx_fk_from_npz(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    betas: np.ndarray | None = None,
    *,
    smplx_npz_path: Path = DEFAULT_SMPLX_NPZ,
    input_yup: bool = True,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact 22-joint Z-up world positions & rotations from SMPL-X parameters using SMPLX_NEUTRAL.npz.

    Returns:
      names: list of 22 canonical LAFAN bone names
      positions_zup: (T, 22, 3) float64 in meters, Z-up
      rotations_zup: (T, 22, 3, 3) float64 world rotation matrices, Z-up
      j_rest_zup: (22, 3) float64 rest-pose joint centers in Z-up (facing +X)
    """
    if not smplx_npz_path.is_file():
        raise FileNotFoundError(
            f"SMPL-X body model not found at {smplx_npz_path}. "
            "Extract models_smplx_v1_1.zip into data/body_models/smplx/."
        )
    model_data = np.load(smplx_npz_path, allow_pickle=True)
    v_template = np.asarray(model_data["v_template"], dtype=np.float64)          # (10475, 3)
    shapedirs = np.asarray(model_data["shapedirs"], dtype=np.float64)            # (10475, 3, 400)
    j_regressor = np.asarray(model_data["J_regressor"][:22], dtype=np.float64)   # (22, 10475)
    parents = np.asarray(model_data["kintree_table"][0, :22], dtype=np.int64)    # (22,)

    global_orient = np.asarray(global_orient, dtype=np.float64).reshape(-1, 3)
    num_frames = global_orient.shape[0]
    body_pose = np.asarray(body_pose, dtype=np.float64).reshape(num_frames, 21, 3)
    transl = np.asarray(transl, dtype=np.float64).reshape(num_frames, 3)

    if betas is None:
        v_shaped = v_template
    else:
        b_vec = np.asarray(betas, dtype=np.float64)
        if b_vec.ndim == 2:
            b_vec = b_vec[0]
        b_dim = min(b_vec.shape[0], shapedirs.shape[2])
        v_shaped = v_template + np.einsum("vcb,b->vc", shapedirs[:, :, :b_dim], b_vec[:b_dim])

    j_rest = j_regressor @ v_shaped  # (22, 3) in native SMPL-X Y-up (+Y up, +Z forward, +X left)

    rot_world = np.empty((num_frames, 22, 3, 3), dtype=np.float64)
    pos_world = np.empty((num_frames, 22, 3), dtype=np.float64)

    rot_world[:, 0] = R.from_rotvec(global_orient).as_matrix()
    pos_world[:, 0] = j_rest[0][None, :] + transl

    for j in range(1, 22):
        p = int(parents[j])
        local_r = R.from_rotvec(body_pose[:, j - 1]).as_matrix()
        rot_world[:, j] = rot_world[:, p] @ local_r
        offset = j_rest[j] - j_rest[p]
        pos_world[:, j] = pos_world[:, p] + np.einsum("tij,j->ti", rot_world[:, p], offset)

    if input_yup:
        pos_zup_smplx = pos_world @ YUP_TO_ZUP.T
        rot_zup_smplx = np.einsum("ij,tnjk->tnik", YUP_TO_ZUP, rot_world)
    else:
        pos_zup_smplx = pos_world
        rot_zup_smplx = rot_world

    # Reorder from SMPLX_22_JOINTS into LAFAN_22_NAMES
    smplx_idx = [SMPLX_22_JOINTS.index(smplx_name) for _, smplx_name in LAFAN_TO_SMPLX_22]
    positions_zup = pos_zup_smplx[:, smplx_idx, :]
    rotations_zup = rot_zup_smplx[:, smplx_idx, :, :]

    # Rest-pose joint centers in Z-up facing +X (rotate YUP_TO_ZUP's -Y forward by +90 deg around Z)
    align_plus_x = R.from_euler("z", 0.5 * np.pi).as_matrix()
    j_rest_plus_x = (j_rest @ YUP_TO_ZUP.T @ align_plus_x.T)[smplx_idx]
    return list(LAFAN_22_NAMES), positions_zup, rotations_zup, j_rest_plus_x


def load_motion_prediction_file(
    pred_path: Path,
    *,
    override_fps: float | None = None,
    smplx_npz_path: Path = DEFAULT_SMPLX_NPZ,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, float]:
    """Load either GVHMR `hmr4d_results.pt` or SMPL-X `.npz` and return Z-up 22-joint kinematics + source FPS."""
    pred_path = pred_path.expanduser().resolve()
    if pred_path.suffix == ".pt":
        import torch

        raw = torch.load(pred_path, map_location="cpu", weights_only=False)
        params = raw["smpl_params_global"] if "smpl_params_global" in raw else raw
        to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        global_orient = to_np(params["global_orient"])
        body_pose = to_np(params["body_pose"])
        transl = to_np(params["transl"])
        betas = to_np(params["betas"]) if "betas" in params else None
        fps = float(override_fps if override_fps is not None else raw.get("fps", 30.0))
        names, pos, rot, j_rest = smplx_fk_from_npz(
            global_orient, body_pose, transl, betas, smplx_npz_path=smplx_npz_path, input_yup=True
        )
        return names, pos, rot, j_rest, fps

    if pred_path.suffix == ".npz":
        data = np.load(pred_path, allow_pickle=True)
        files = set(data.files)
        global_orient = data["root_orient"] if "root_orient" in files else data["global_orient"]
        body_pose = data["pose_body"] if "pose_body" in files else data["body_pose"]
        if body_pose.ndim == 2 and body_pose.shape[1] > 63:
            body_pose = body_pose[:, :63]
        transl = data["trans"] if "trans" in files else data["transl"]
        betas = data["betas"] if "betas" in files else None
        raw_fps = (
            data["mocap_frame_rate"].item()
            if "mocap_frame_rate" in files
            else (data["fps"].item() if "fps" in files else 30.0)
        )
        fps = float(override_fps if override_fps is not None else raw_fps)
        # AMASS / FineDance .npz files are already Z-up unless explicitly tagged Y-up
        input_yup = bool(data["is_yup"].item()) if "is_yup" in files else False
        names, pos, rot, j_rest = smplx_fk_from_npz(
            global_orient, body_pose, transl, betas, smplx_npz_path=smplx_npz_path, input_yup=input_yup
        )
        return names, pos, rot, j_rest, fps

    raise ValueError(f"Unsupported motion file extension (expected .pt or .npz): {pred_path}")


def calibrated_smplx_offsets_for_light(model: mujoco.MjModel) -> dict[str, dict]:
    """Compute exact analytical SO(3) rotation offsets mapping SMPL-X 22 joints to Humanoid-Light links.

    In SMPL-X zero pose (T-pose facing +X after `align_plus_x @ YUP_TO_ZUP`), every human joint has
    world orientation `R_h0 = align_plus_x @ YUP_TO_ZUP`. Setting `Humanoid-Light` into the matching
    T-pose (`left_shoulder_roll = +80 deg`, `right_shoulder_roll = -80 deg`) aligns every link's
    world orientation `R_l_tpose` directly with `R_h0`, giving `offset = R_h0.T @ R_l_tpose`.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[model.joint("left_shoulder_roll_joint").qposadr[0]] = np.deg2rad(80.0)
    data.qpos[model.joint("right_shoulder_roll_joint").qposadr[0]] = np.deg2rad(-80.0)
    mujoco.mj_forward(model, data)

    align_plus_x = R.from_euler("z", 0.5 * np.pi).as_matrix()
    r_human_tpose = align_plus_x @ YUP_TO_ZUP  # world rotation of every SMPL-X joint in +X-facing T-pose

    body_to_human = {
        "pelvis_link": "Hips",
        "torso_roll_link": "Spine2",
        "head_link": "Head",
        "left_hip_yaw_link": "LeftUpLeg",
        "left_knee_link": "LeftLeg",
        "left_ankle_roll_link": "LeftToe",
        "left_shoulder_yaw_link": "LeftArm",
        "left_elbow_link": "LeftForeArm",
        "left_wrist_link": "LeftHand",
        "right_hip_yaw_link": "RightUpLeg",
        "right_knee_link": "RightLeg",
        "right_ankle_roll_link": "RightToe",
        "right_shoulder_yaw_link": "RightArm",
        "right_elbow_link": "RightForeArm",
        "right_wrist_link": "RightHand",
    }

    offsets: dict[str, dict] = {}
    for robot_body, human_name in body_to_human.items():
        r_light = data.xmat[model.body(robot_body).id].reshape(3, 3).copy()
        offset_mat = r_human_tpose.T @ r_light
        offsets[robot_body] = {
            "human": human_name,
            "rotation_wxyz": R.from_matrix(offset_mat).as_quat(scalar_first=True).tolist(),
        }
    return offsets


def retarget_smplx_to_light_50hz(
    pred_path: Path,
    output_npz: Path,
    *,
    source_fps_override: float | None = None,
    start_s: float = 0.0,
    duration_s: float | None = None,
    target_fps: float = 50.0,
    smplx_npz_path: Path = DEFAULT_SMPLX_NPZ,
    progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Retarget a GVHMR `.pt` or SMPL-X `.npz` file to 50 Hz Humanoid-Light `(qpos, qvel)` via Mink/DAQP IK."""
    import retarget_humanoid_light_lafan as lafan_retarget
    from robots.humanoid.humanoid_light.config import ASSET_DIR

    if progress_cb:
        progress_cb(0.45, "SMPL-X 포워드 키네마틱스 및 체형 스케일 계산 중...")

    mink, _, _ = lafan_retarget.prepare()
    scene_xml = ASSET_DIR / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    zero_data = mujoco.MjData(model)
    mujoco.mj_forward(model, zero_data)

    names, positions, rotations, j_rest, source_fps = load_motion_prediction_file(
        pred_path, override_fps=source_fps_override, smplx_npz_path=smplx_npz_path
    )
    total_source_seconds = (len(positions) - 1) / source_fps
    if total_source_seconds <= 0.04:
        raise ValueError(f"Source motion too short ({len(positions)} frames @ {source_fps:.2f} Hz)")

    clip_start = max(0.0, float(start_s))
    max_dur = max(0.04, total_source_seconds - clip_start - 1e-6)
    clip_dur = max_dur if duration_s is None else min(float(duration_s), max_dur)

    offsets = calibrated_smplx_offsets_for_light(model)
    meshes, feet = lafan_retarget.geometry(model)

    # Subject-specific leg morphology scale from SMPL-X shaped rest skeleton
    idx = {n: i for i, n in enumerate(names)}
    robot_leg = sum(
        np.linalg.norm(
            zero_data.xpos[model.body(f"left_{end}_link").id]
            - zero_data.xpos[model.body(f"left_{first}_link").id]
        )
        for first, end in (("hip_pitch", "knee"), ("knee", "ankle_pitch"))
    )
    human_leg = (
        np.linalg.norm(j_rest[idx["LeftLeg"]] - j_rest[idx["LeftUpLeg"]])
        + np.linalg.norm(j_rest[idx["LeftFoot"]] - j_rest[idx["LeftLeg"]])
    )
    scale = float(robot_leg / human_leg)
    ankle_z_ref = float(0.5 * (j_rest[idx["LeftFoot"], 2] + j_rest[idx["RightFoot"], 2]))
    hips_z_ref = float(j_rest[idx["Hips"], 2])
    # Shift rest Z references to floor=0 using rest toes
    toe_floor = float(min(j_rest[idx["LeftToe"], 2], j_rest[idx["RightToe"], 2]))
    positions_grounded = positions.copy()
    floor_q05 = float(
        np.quantile(
            np.minimum(positions_grounded[:, idx["LeftToe"], 2], positions_grounded[:, idx["RightToe"], 2]),
            0.05,
        )
    )
    positions_grounded[:, :, 2] -= floor_q05
    reference = (
        scale,
        float(zero_data.qpos[2]),
        hips_z_ref - toe_floor,
        ankle_z_ref - toe_floor,
    )

    controls = json.loads((ASSET_DIR / "control.json").read_text())["groups"]
    joint_names = [
        model.joint(j).name
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    speeds: dict[str, float] = {}
    for jname in joint_names:
        for group in controls.values():
            if any(re.fullmatch(pat, jname) for pat in group["joint_names_expr"]):
                for pat, val in group["velocity_limit"].items():
                    if re.fullmatch(pat, jname):
                        speeds[jname] = float(val)

    root_tgt, target_r, foot_tgt, contacts, human_aligned, _, _ = lafan_retarget.targets_for_clip(
        names,
        positions_grounded,
        rotations,
        source_fps,
        clip_start,
        clip_dur,
        offsets,
        model,
        reference,
        feet,
        target_fps,
    )

    if progress_cb:
        progress_cb(0.60, f"Humanoid-Light 50Hz Mink/DAQP 역기구학 최적화 중 ({len(root_tgt)} 프레임)...")

    qpos = lafan_retarget.run_ik(
        mink,
        model,
        root_tgt,
        target_r,
        foot_tgt,
        contacts,
        target_fps,
        meshes,
        speeds,
        human_aligned,
        names,
        hard_stance=False,
    )
    qvel = lafan_retarget.differentiate(model, qpos, target_fps)
    report = lafan_retarget.quality(
        model,
        qpos,
        qvel,
        target_r,
        foot_tgt,
        contacts,
        target_fps,
        meshes,
        speeds,
        human_aligned,
        names,
        root_tgt,
        hard_stance=False,
    )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    src_sha = hashlib.sha256(pred_path.read_bytes()).hexdigest()
    np.savez_compressed(
        output_npz,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        frequency=np.asarray(target_fps, dtype=np.float64),
        joint_names=np.asarray(["root"] + joint_names),
        split_points=np.array([0, len(qpos)], dtype=np.int64),
        source_contact=contacts,
        source_sha256=np.asarray(src_sha),
        training_ready=np.asarray(False),
    )
    output_npz.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return qpos.astype(np.float32), qvel.astype(np.float32), report


def _ensure_yt_dlp() -> str:
    """Return a working yt-dlp executable or python -m yt_dlp invocation."""
    exe = shutil.which("yt-dlp")
    if exe:
        return exe
    RETARGET_DEPS_DIR.mkdir(parents=True, exist_ok=True)
    if str(RETARGET_DEPS_DIR) not in sys.path:
        sys.path.insert(0, str(RETARGET_DEPS_DIR))
    if importlib.util.find_spec("yt_dlp") is None:
        subprocess.run(
            ["uv", "pip", "install", "--target", str(RETARGET_DEPS_DIR), "yt-dlp"],
            check=True,
        )
    return "__python_module_yt_dlp__"


def download_and_decode_video(
    source_url_or_path: str,
    start_s: float,
    duration_s: float,
    work_dir: Path,
    *,
    max_height: int = 360,
    progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[Path, np.ndarray, float]:
    """Download YouTube URL (or copy local video), trim `[start_s, start_s+duration_s]`, and decode RGB frames."""
    work_dir.mkdir(parents=True, exist_ok=True)
    src = source_url_or_path.strip()
    if not src:
        raise ValueError("영상 URL 또는 로컬 파일 경로를 입력해 주세요.")

    raw_video = work_dir / "source_raw.mp4"
    trimmed_video = work_dir / f"trimmed_{start_s:.2f}_{duration_s:.2f}.mp4"

    if src.startswith(("http://", "https://", "www.youtube.com", "youtu.be")):
        if progress_cb:
            progress_cb(0.08, "YouTube 영상 다운로드 중 (yt-dlp)...")
        yt_cmd = _ensure_yt_dlp()
        base_cmd = (
            [sys.executable, "-m", "yt_dlp"]
            if yt_cmd == "__python_module_yt_dlp__"
            else [yt_cmd]
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{RETARGET_DEPS_DIR}:{env.get('PYTHONPATH', '')}"
        subprocess.run(
            [
                *base_cmd,
                "-f",
                "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
                "--merge-output-format",
                "mp4",
                "-o",
                str(raw_video),
                "--force-overwrites",
                src,
            ],
            check=True,
            env=env,
        )
    else:
        local_path = Path(src).expanduser().resolve()
        if not local_path.is_file():
            raise FileNotFoundError(f"로컬 영상 파일을 찾을 수 없습니다: {local_path}")
        raw_video = local_path

    if progress_cb:
        progress_cb(0.18, f"원본 구간 트림 중 ({start_s:.1f}s ~ {start_s + duration_s:.1f}s)...")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(0.0, start_s):.3f}",
            "-i",
            str(raw_video),
            "-t",
            f"{max(0.5, duration_s):.3f}",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(trimmed_video),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    frames, fps = decode_video_rgb_frames(trimmed_video, max_height=max_height)
    return trimmed_video, frames, fps


def decode_video_rgb_frames(video_path: Path, *, max_height: int = 360) -> tuple[np.ndarray, float]:
    """Decode an MP4 into `(N, H, W, 3)` uint8 RGB frames and exact floating-point FPS using ffprobe + imageio/ffmpeg."""
    import imageio.v3 as iio

    meta = iio.immeta(video_path, plugin="pyav") if _has_pyav() else {}
    fps = float(meta.get("fps", 0.0) or 0.0)
    if fps <= 0.0:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate,width,height",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        num, den = stream["r_frame_rate"].split("/")
        fps = float(num) / float(den)
        orig_w, orig_h = int(stream["width"]), int(stream["height"])
    else:
        orig_w = orig_h = 0

    if orig_h == 0:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        orig_w, orig_h = int(stream["width"]), int(stream["height"])

    scale_ratio = min(1.0, float(max_height) / float(orig_h))
    out_h = max(2, int(round(orig_h * scale_ratio)) // 2 * 2)
    out_w = max(2, int(round(orig_w * scale_ratio)) // 2 * 2)

    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"scale={out_w}:{out_h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    frame_bytes = out_h * out_w * 3
    n_frames = len(raw) // frame_bytes
    if n_frames < 1:
        raise RuntimeError(f"Decoded 0 video frames from {video_path}")
    frames = np.frombuffer(raw[: n_frames * frame_bytes], dtype=np.uint8).reshape(
        n_frames, out_h, out_w, 3
    )
    return frames.copy(), fps


def _has_pyav() -> bool:
    return importlib.util.find_spec("av") is not None


def run_gvhmr_extraction(
    trimmed_video: Path,
    work_dir: Path,
    *,
    static_camera: bool = True,
    progress_cb: Callable[[float, str], None] | None = None,
) -> Path:
    """Invoke external GVHMR checkout (`GVHMR_ROOT` or `outputs/gvhmr/GVHMR`) to produce `hmr4d_results.pt`."""
    cached_pts = list(work_dir.rglob("hmr4d_results.pt"))
    if cached_pts:
        return cached_pts[0]

    candidates = []
    if os.environ.get("GVHMR_ROOT"):
        candidates.append(Path(os.environ["GVHMR_ROOT"]).expanduser().resolve())
    candidates.extend(
        [
            REPO_ROOT / "outputs/gvhmr/GVHMR",
            REPO_ROOT / "third_party/GVHMR",
            Path.home() / "GVHMR",
        ]
    )
    gvhmr_root = next((c for c in candidates if (c / "tools/demo/demo.py").is_file()), None)
    if gvhmr_root is None:
        raise RuntimeError(
            "GVHMR 추론 엔진이 아직 설치되지 않았습니다 (현재 S2R 본 학습 GPU 보호 중). "
            "지금 바로 테스트하시려면 아래 'SMPL-X / GVHMR (.pt/.npz)' 칸에 추출된 모션 파일 경로를 넣으시거나, "
            "본 학습 종료 후 GVHMR 체크포인트를 outputs/gvhmr/GVHMR 에 연결하면 영상 URL만으로 원클릭 실행됩니다."
        )

    gvhmr_python = os.environ.get("GVHMR_PYTHON", sys.executable)
    if progress_cb:
        progress_cb(0.28, "GVHMR 3D 전신 모션(SMPL-X) 추출 중...")

    cmd = [
        gvhmr_python,
        "tools/demo/demo.py",
        f"--video={trimmed_video}",
        f"--output_root={work_dir / 'gvhmr_out'}",
    ]
    if static_camera:
        cmd.append("-s")
    subprocess.run(cmd, cwd=str(gvhmr_root), check=True)

    found = list((work_dir / "gvhmr_out").rglob("hmr4d_results.pt"))
    if not found:
        raise FileNotFoundError(f"GVHMR finished without creating hmr4d_results.pt under {work_dir / 'gvhmr_out'}")
    return found[0]


def synced_video_frame_index(
    motion_frame: int,
    clip_start_frame: int,
    num_video_frames: int,
    video_fps: float,
    motion_fps: float = 50.0,
) -> int:
    """Map current 50 Hz robot motion frame to exact `[0, num_video_frames - 1]` RGB frame index."""
    if num_video_frames <= 1:
        return 0
    rel_motion = max(0, int(motion_frame) - int(clip_start_frame))
    elapsed_s = rel_motion / float(motion_fps)
    vid_idx = int(round(elapsed_s * float(video_fps)))
    return min(max(0, vid_idx), num_video_frames - 1)


def append_clip_to_motion_loader(
    motion_cmd,
    qpos_50hz: np.ndarray,
    qvel_50hz: np.ndarray,
    clip_key: str,
    step_dt: float = 0.02,
) -> tuple[int, int]:
    """Append a 50 Hz `(qpos, qvel)` Humanoid-Light trajectory to `motion_cmd.motion` in memory.

    Returns `(new_clip_idx, clip_start_frame)`.
    """
    import torch

    q = np.asarray(qpos_50hz, dtype=np.float32)
    v = np.asarray(qvel_50hz, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 33 or v.shape != (q.shape[0], 32) or q.shape[0] < 2:
        raise ValueError(f"Expected qpos (T>=2, 33) and qvel (T, 32), got {q.shape} and {v.shape}")

    motion = motion_cmd.motion
    device = motion.joint_pos.device
    old_total = int(motion.time_step_total)
    new_frames = int(q.shape[0])

    ang_vel_w = _rotate_quat_vec(q[:, 3:7], v[:, 3:6]).astype(np.float32)

    add_tensors = {
        "root_pos_w": torch.as_tensor(q[:, :3], dtype=torch.float32, device=device),
        "root_quat_w": torch.as_tensor(q[:, 3:7], dtype=torch.float32, device=device),
        "root_lin_vel_w": torch.as_tensor(v[:, :3], dtype=torch.float32, device=device),
        "root_ang_vel_w": torch.as_tensor(ang_vel_w, dtype=torch.float32, device=device),
        "joint_pos": torch.as_tensor(q[:, 7:], dtype=torch.float32, device=device),
        "joint_vel": torch.as_tensor(v[:, 6:], dtype=torch.float32, device=device),
    }
    for attr, tensor in add_tensors.items():
        setattr(motion, attr, torch.cat([getattr(motion, attr), tensor], dim=0))

    new_total = old_total + new_frames
    motion.time_step_total = new_total
    motion.motion_ends = torch.cat(
        [motion.motion_ends, torch.tensor([new_total], dtype=torch.long, device=device)],
        dim=0,
    )
    motion.motion_names.append(clip_key)
    new_clip_idx = len(motion.motion_names) - 1

    # Resize MotionCommand bin statistics so uniform/external sampling never goes out of bounds
    new_bin_count = int(new_total // (0.2 / step_dt)) + 1
    if hasattr(motion_cmd, "bin_count") and new_bin_count > motion_cmd.bin_count:
        pad = new_bin_count - motion_cmd.bin_count
        motion_cmd.bin_count = new_bin_count
        motion_cmd.bin_failed_count = torch.cat(
            [motion_cmd.bin_failed_count, torch.zeros(pad, dtype=torch.float, device=device)]
        )
        motion_cmd._current_bin_failed = torch.cat(
            [motion_cmd._current_bin_failed, torch.zeros(pad, dtype=torch.float, device=device)]
        )
        motion_cmd._external_probs = torch.full(
            (new_bin_count,), 1.0 / new_bin_count, device=device
        )

    return new_clip_idx, old_total


class VideoDanceStudioPanel:
    """Interactive Viser panel + 3D virtual studio screen for YouTube/Video -> GVHMR -> SMPL-X -> Robot sync."""

    def __init__(self, viewer, server) -> None:
        self.viewer = viewer
        self.server = server
        self._synced_videos: dict[int, dict] = {}
        self._worker: threading.Thread | None = None
        self._last_rendered_key: tuple[int, int] | None = None
        self._screen_shape: tuple[int, int] | None = None
        self._screen_handle = None
        self._preview_handle = None
        self._setup_gui()

    def _setup_gui(self) -> None:
        placeholder = np.zeros((180, 320, 3), dtype=np.uint8)
        placeholder[:, :] = (22, 26, 36)

        with self.server.gui.add_folder("🎬 Video Dance Studio", expand_by_default=True):
            self._upload_btn = self.server.gui.add_upload_button(
                "📁 Upload Local .mp4",
                mime_type="video/*",
            )
            self._url_input = self.server.gui.add_text(
                "YouTube URL / MP4",
                initial_value="",
                hint="https://youtu.be/... 또는 로컬 .mp4 경로",
            )
            self._pred_input = self.server.gui.add_text(
                "SMPL-X / GVHMR (.pt/.npz)",
                initial_value="",
                hint="(선택) 이미 추출된 hmr4d_results.pt 또는 SMPL-X .npz 경로",
            )
            self._label_input = self.server.gui.add_text(
                "Clip Name",
                initial_value="Idol_Dance",
            )
            self._start_input = self.server.gui.add_number(
                "Start (s)",
                initial_value=0.0,
                min=0.0,
                max=3600.0,
                step=0.5,
            )
            self._dur_input = self.server.gui.add_number(
                "Duration (s)",
                initial_value=8.0,
                min=1.0,
                max=60.0,
                step=0.5,
            )
            self._static_cam_cb = self.server.gui.add_checkbox(
                "Fixed Camera (Dance Practice -s)",
                initial_value=True,
            )
            self._show_3d_screen_cb = self.server.gui.add_checkbox(
                "3D Virtual Dance Screen",
                initial_value=True,
            )
            self._follow_robot_cb = self.server.gui.add_checkbox(
                "Screen Follows Robot",
                initial_value=True,
            )
            self._run_btn = self.server.gui.add_button("🚀 Extract, Retarget & Sync")
            self._progress = self.server.gui.add_progress_bar(value=0.0, visible=False)
            self._status_md = self.server.gui.add_markdown(
                "_대기 중: 영상 URL(또는 .mp4 업로드 / SMPL-X 파일) 입력 후 실행하세요._"
            )
            self._preview_handle = self.server.gui.add_image(
                placeholder,
                label="Synced Reference Video",
                format="jpeg",
                jpeg_quality=80,
            )

        @self._upload_btn.on_upload
        def _(event) -> None:
            uploaded = event.target.value
            safe_name = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", uploaded.name) or "uploaded.mp4"
            dest = STUDIO_OUTPUT_DIR / "uploads" / safe_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(uploaded.content)
            self._url_input.value = str(dest)
            self._label_input.value = Path(safe_name).stem
            self._set_status(None, f"📥 업로드 완료: `{safe_name}` ({len(uploaded.content) / 1048576:.1f} MB)")

        @self._show_3d_screen_cb.on_update
        def _(_) -> None:
            if self._screen_handle is not None:
                self._screen_handle.visible = bool(self._show_3d_screen_cb.value)

        @self._run_btn.on_click
        def _(_) -> None:
            if self._worker is not None and self._worker.is_alive():
                self._set_status(None, "⚠️ 이미 변환 작업이 실행 중입니다.")
                return
            self._worker = threading.Thread(target=self._run_pipeline_worker, daemon=True)
            self._worker.start()

    def _set_status(self, progress_pct: float | None, message: str) -> None:
        if progress_pct is not None:
            self._progress.visible = True
            self._progress.value = float(np.clip(progress_pct * 100.0, 0.0, 100.0))
        self._status_md.content = message

    def _run_pipeline_worker(self) -> None:
        try:
            url_or_video = self._url_input.value.strip()
            pred_override = self._pred_input.value.strip()
            label = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", self._label_input.value.strip()) or "Dance_Clip"
            start_s = float(self._start_input.value)
            dur_s = float(self._dur_input.value)
            static_cam = bool(self._static_cam_cb.value)

            job_hash = hashlib.sha256(
                f"{url_or_video}|{pred_override}|{start_s:.3f}|{dur_s:.3f}|{static_cam}".encode()
            ).hexdigest()[:10]
            work_dir = STUDIO_OUTPUT_DIR / f"{label}_{job_hash}"
            work_dir.mkdir(parents=True, exist_ok=True)

            video_frames: np.ndarray | None = None
            video_fps: float | None = None
            trimmed_video: Path | None = None

            if url_or_video:
                trimmed_video, video_frames, video_fps = download_and_decode_video(
                    url_or_video,
                    start_s=start_s,
                    duration_s=dur_s,
                    work_dir=work_dir,
                    progress_cb=self._set_status,
                )

            if pred_override:
                pred_path = Path(pred_override).expanduser().resolve()
                if not pred_path.is_file():
                    raise FileNotFoundError(f"모션 파일을 찾을 수 없습니다: {pred_path}")
                # When pred_override is an untrimmed file and no video was trimmed, slice inside retargeter
                retarget_start = 0.0 if trimmed_video is not None else start_s
                retarget_dur = dur_s
            else:
                if trimmed_video is None:
                    raise ValueError("YouTube URL / 로컬 영상 경로 또는 SMPL-X(.pt/.npz) 경로를 입력해 주세요.")
                pred_path = run_gvhmr_extraction(
                    trimmed_video,
                    work_dir=work_dir,
                    static_camera=static_cam,
                    progress_cb=self._set_status,
                )
                retarget_start = 0.0
                retarget_dur = None

            out_npz = work_dir / f"{label}_light_50hz.npz"
            qpos, qvel, report = retarget_smplx_to_light_50hz(
                pred_path,
                out_npz,
                source_fps_override=video_fps,
                start_s=retarget_start,
                duration_s=retarget_dur,
                target_fps=50.0,
                progress_cb=self._set_status,
            )

            self._set_status(0.92, "Viser 모션 버퍼 Hot-Swap 및 영상 싱크 연결 중...")
            clip_key = f"studio/{label}/0-{len(qpos)}"
            self.register_and_activate_clip(
                qpos=qpos,
                qvel=qvel,
                label=label,
                clip_key=clip_key,
                video_frames=video_frames,
                video_fps=video_fps or 30.0,
            )
            self._set_status(
                1.0,
                f"✅ **완료!** `{label}` (`{len(qpos)}` 프레임 / `{report['duration_s']:.1f}s` @ 50Hz) 재생 중",
            )
        except Exception as exc:
            traceback.print_exc()
            self._set_status(None, f"❌ **오류**: {exc}")

    def register_and_activate_clip(
        self,
        *,
        qpos: np.ndarray,
        qvel: np.ndarray,
        label: str,
        clip_key: str,
        video_frames: np.ndarray | None,
        video_fps: float,
    ) -> int:
        with self.viewer._sim_lock:
            base_env = self.viewer.env.unwrapped
            motion_cmd = base_env.command_manager.get_term("motion")
            new_clip_idx, clip_start = append_clip_to_motion_loader(
                motion_cmd, qpos, qvel, clip_key=clip_key, step_dt=float(base_env.step_dt)
            )
            device = next(self.viewer.policy.parameters()).device
            self.viewer._replay_goal_tables = {
                name: ts.precompute_replay_goal(motion_cmd.motion).to(device)
                for name, ts in self.viewer.task_specs.items()
                if ts.precompute_replay_goal is not None
            }
            if video_frames is not None and len(video_frames) > 0:
                self._synced_videos[new_clip_idx] = {
                    "frames": video_frames,
                    "fps": float(video_fps),
                    "clip_start": int(clip_start),
                }

            pos = len(self.viewer._ordered_playlist)
            title = f"{pos:02d}: 🎬 {label} [{clip_key}]"
            self.viewer._ordered_playlist.append((title, new_clip_idx))
            self.viewer.motion_names = list(motion_cmd.motion.motion_names)
            self.viewer._playlist_pos = pos
            self.viewer._selected_clip_idx = new_clip_idx

            if self.viewer._clip_dropdown is not None:
                self.viewer._suppress_dd_cb = True
                self.viewer._clip_dropdown.options = [it[0] for it in self.viewer._ordered_playlist]
                self.viewer._clip_dropdown.value = title
                self.viewer._suppress_dd_cb = False

            self._last_rendered_key = None
            self.viewer._apply_selected_clip()
            self.viewer.reset_wm_visualizer()
        return new_clip_idx

    def update_synced_video_frame(self) -> None:
        clip_idx = getattr(self.viewer, "_selected_clip_idx", -1)
        entry = self._synced_videos.get(clip_idx)
        if entry is None:
            if self._screen_handle is not None:
                self._screen_handle.visible = False
            return

        base_env = self.viewer.env.unwrapped
        motion_cmd = base_env.command_manager.get_term("motion")
        motion_frame = int(motion_cmd.time_steps[0].item())
        frames: np.ndarray = entry["frames"]
        v_idx = synced_video_frame_index(
            motion_frame=motion_frame,
            clip_start_frame=entry["clip_start"],
            num_video_frames=len(frames),
            video_fps=entry["fps"],
            motion_fps=50.0,
        )

        # Update 3D position even if frame index repeats when robot moves
        rgb = frames[v_idx]
        h, w = rgb.shape[:2]
        render_h = 1.35
        render_w = render_h * (float(w) / float(h))

        if bool(self._follow_robot_cb.value):
            root_xyz = base_env.sim.data.qpos[0, :3].detach().cpu().numpy()
            screen_pos = (float(root_xyz[0]) - 0.8, float(root_xyz[1]) + 1.25, 0.95)
        else:
            screen_pos = (-1.0, 1.35, 0.95)

        # Upright billboard facing +X viewer (local +X = +Y right, local +Y = -Z down)
        screen_wxyz = (0.5, -0.5, -0.5, 0.5)

        key = (clip_idx, v_idx)
        if self._last_rendered_key != key:
            self._last_rendered_key = key
            if self._preview_handle is not None:
                self._preview_handle.image = rgb
            if self._screen_handle is None or self._screen_shape != (h, w):
                self._screen_shape = (h, w)
                self._screen_handle = self.server.scene.add_image(
                    "/video_studio/screen",
                    rgb,
                    render_width=render_w,
                    render_height=render_h,
                    format="jpeg",
                    jpeg_quality=80,
                    cast_shadow=False,
                    receive_shadow=False,
                    wxyz=screen_wxyz,
                    position=screen_pos,
                    visible=bool(self._show_3d_screen_cb.value),
                )
            else:
                self._screen_handle.image = rgb
                self._screen_handle.position = screen_pos
                self._screen_handle.visible = bool(self._show_3d_screen_cb.value)
        elif self._screen_handle is not None:
            self._screen_handle.position = screen_pos
            self._screen_handle.visible = bool(self._show_3d_screen_cb.value)
