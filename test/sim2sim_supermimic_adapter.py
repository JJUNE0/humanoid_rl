"""Self-contained Sim-to-Sim inference & coupled-PD adapter for Humanoid-Light SuperMimic.

Runs in both Python 3.13 (.venv_supermimic_light) and Python 3.11 (env_isaaclab)
using only PyTorch and NumPy (no mjlab / tensordict / rsl_rl runtime dependency).

Can also export a standalone TorchScript actor bundle:
  .venv_supermimic_light/bin/python test/sim2sim_supermimic_adapter.py \
      --checkpoint runs/light_supermimic_walk1_t4/t4_walk1_fixed_20261118/training \
      --export-torchscript outputs/sim2sim/walk1_actor_jit.pt
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_JSON_PATH = (
    REPO_ROOT / "src/robots/humanoid/humanoid_light/assets/control.json"
)
MJCF_XML_PATH = (
    REPO_ROOT / "src/robots/humanoid/humanoid_light/assets/humanoid_light.xml"
)

JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_yaw_joint",
    "torso_pitch_joint",
    "torso_roll_joint",
    "head_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_joint",
)

NUM_GOAL_FRAMES = 10
GOAL_FRAME_STRIDE = 5


def _install_numpy2_pickle_shim() -> None:
    """Allow NumPy 1.26 (IsaacLab Python 3.11) to unpickle NumPy 2.x RNG states in last.pt."""
    if "numpy._core" not in sys.modules:
        import numpy.core
        import numpy.core.multiarray

        sys.modules["numpy._core"] = numpy.core
        sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
        if hasattr(numpy.core, "_multiarray_umath"):
            sys.modules["numpy._core._multiarray_umath"] = numpy.core._multiarray_umath


def resolve_checkpoint_file(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    if (path / "last.pt").is_file():
        return path / "last.pt"
    if (path / "latest.json").is_file():
        latest = json.loads((path / "latest.json").read_text())
        candidate = path / latest["checkpoint"] / "last.pt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find last.pt or latest.json under checkpoint path: {path}"
    )


def load_checkpoint_compat(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[dict[str, Any], Path, str]:
    ckpt_file = resolve_checkpoint_file(path)
    sha = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()
    _install_numpy2_pickle_shim()
    saved = torch.load(ckpt_file, map_location=device, weights_only=False)
    return saved, ckpt_file, sha


# ---------------------------------------------------------------------------
# Pure-PyTorch quaternion & frame math (wxyz convention, matching mjlab/IsaacLab)
# ---------------------------------------------------------------------------


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1)


def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[..., 1:]
    q_w = q[..., 0:1]
    uv = 2.0 * torch.linalg.cross(q_vec, v, dim=-1)
    return v + q_w * uv + torch.linalg.cross(q_vec, uv, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[..., 1:]
    q_w = q[..., 0:1]
    uv = 2.0 * torch.linalg.cross(q_vec, v, dim=-1)
    return v - q_w * uv + torch.linalg.cross(q_vec, uv, dim=-1)


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1).clamp(min=1e-12)
    o = torch.stack(
        (
            1.0 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1.0 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1.0 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    mat = matrix_from_quat(quat)
    return mat[..., :2].flatten(-2, -1)


# ---------------------------------------------------------------------------
# Pure-PyTorch SuperMimic Actor (matches rsl_rl EmpiricalNormalization + MultiEncoderAE)
# ---------------------------------------------------------------------------


class FrozenEmpiricalNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-2) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("_mean", torch.zeros(1, dim))
        self.register_buffer("_var", torch.ones(1, dim))
        self.register_buffer("_std", torch.ones(1, dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)


def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    curr = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(curr, h), nn.LayerNorm(h), nn.GELU()]
        curr = h
    layers.append(nn.Linear(curr, output_dim))
    return nn.Sequential(*layers)


class StandaloneMotionActor(nn.Module):
    """Exact inference mirror of supermimic.models.MultiEncoderAE for the 'motion' task."""

    def __init__(
        self,
        obs_dim: int = 62,
        motion_goal_dim: int = 580,
        action_dim: int = 26,
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] = (1024, 512, 256),
        history_length: int = 1,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.motion_goal_dim = motion_goal_dim
        self.action_dim = action_dim
        self.history_length = history_length

        self.obs_normalizer = FrozenEmpiricalNorm(obs_dim)
        self.goal_normalizers = nn.ModuleDict(
            {"motion": FrozenEmpiricalNorm(motion_goal_dim)}
        )
        self.encoders = nn.ModuleDict(
            {"motion": _build_mlp(motion_goal_dim, hidden_dims, latent_dim)}
        )
        self.decoder = _build_mlp(
            latent_dim + history_length * obs_dim, hidden_dims, action_dim
        )

    def load_supermimic_state_dict(self, actor_state_dict: dict[str, torch.Tensor]) -> None:
        filtered = {
            k: v
            for k, v in actor_state_dict.items()
            if not k.startswith("goal_normalizers.velocity.")
            and not k.startswith("encoders.velocity.")
        }
        self.load_state_dict(filtered, strict=True)
        self.eval()

    def forward(self, rep_state: torch.Tensor, actor_goal: torch.Tensor) -> torch.Tensor:
        """Compute raw policy actions from (B, 62) rep_state and (B, 580) actor_goal."""
        goal_n = self.goal_normalizers["motion"](actor_goal)
        z = torch.tanh(self.encoders["motion"](goal_n))
        obs_n = self.obs_normalizer(rep_state)
        return self.decoder(torch.cat([z, obs_n], dim=-1))


def build_standalone_actor_from_saved(
    saved: dict[str, Any], device: str | torch.device
) -> StandaloneMotionActor:
    runner_cfg = saved["settings"]["runner"]
    hidden_dims = tuple(int(h) for h in runner_cfg["actor_hidden_dims"])
    latent_dim = int(runner_cfg.get("latent_dim", 32))
    history_length = int(runner_cfg.get("history_length", 1))
    if history_length != 1 or runner_cfg.get("actor_obs", "rep_state") != "rep_state":
        raise ValueError("StandaloneMotionActor expects history_length=1 and actor_obs='rep_state'")
    actor = StandaloneMotionActor(
        obs_dim=62,
        motion_goal_dim=580,
        action_dim=26,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        history_length=history_length,
    ).to(device)
    actor.load_supermimic_state_dict(saved["actor"])
    return actor


# ---------------------------------------------------------------------------
# Observation & Goal Builders (exact match to GenCoordRepresentation)
# ---------------------------------------------------------------------------


def build_rep_state(
    root_pos_local: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
    qj_mj: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    vj_mj: torch.Tensor,
) -> torch.Tensor:
    """Build 62-D gencoord rep_state: [z(1), g_b(3), v_b(3), w_b(3), qj(26), vj(26)]."""
    z = root_pos_local[..., 2:3]
    world_down = torch.zeros_like(root_lin_vel_w)
    world_down[..., 2] = -1.0
    g_b = quat_apply_inverse(root_quat_wxyz, world_down)
    v_b = quat_apply_inverse(root_quat_wxyz, root_lin_vel_w)
    w_b = quat_apply_inverse(root_quat_wxyz, root_ang_vel_w)
    return torch.cat([z, g_b, v_b, w_b, qj_mj, vj_mj], dim=-1)


class PackedMotionLibrary:
    """Pure PyTorch/NumPy loader for packed SuperMimic motion arrays."""

    def __init__(self, packed_dir: Path, device: str | torch.device = "cpu") -> None:
        packed_dir = Path(packed_dir)
        joint_names = tuple(json.loads((packed_dir / "joint_names.json").read_text()))
        if joint_names != JOINT_NAMES:
            raise ValueError(f"Packed joint order mismatch: {joint_names}")

        def load_t(name: str) -> torch.Tensor:
            arr = np.load(packed_dir / f"{name}.npy")
            return torch.from_numpy(arr).to(device=device, dtype=torch.float32)

        self.root_pos_w = load_t("root_pos_w")
        self.root_quat_w = load_t("root_quat_w")
        self.root_lin_vel_w = load_t("root_lin_vel_w")
        self.root_ang_vel_w = load_t("root_ang_vel_w")
        self.joint_pos = load_t("joint_pos")
        self.joint_vel = load_t("joint_vel")
        self.motion_ends = torch.as_tensor(
            np.load(packed_dir / "motion_ends.npy"), dtype=torch.long, device=device
        )
        self.motion_names: list[str] = json.loads(
            (packed_dir / "motion_names.json").read_text()
        )
        self.time_step_total = int(self.joint_pos.shape[0])
        self.num_clips = int(self.motion_ends.numel())

    def clip_bounds(self, clip_idx: int) -> tuple[int, int]:
        start = 0 if clip_idx == 0 else int(self.motion_ends[clip_idx - 1].item())
        end = int(self.motion_ends[clip_idx].item())
        return start, end


def pack_manifest_pure_numpy(
    manifest_path: Path,
    repo_root: Path = REPO_ROOT,
    split: str = "train",
    chunk_frames: int = 501,
    cache_dir: Path | None = None,
) -> Path:
    """Pack a screening manifest into numpy arrays without importing humanoid_rl/mjlab."""
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    settings_key = {"split": split, "limit": None, "max_frames": None, "chunk_frames": chunk_frames}
    digest = hashlib.sha256(b"light-supermimic-packed-v1" + manifest_bytes)
    digest.update(json.dumps(settings_key, sort_keys=True).encode())
    digest.update(MJCF_XML_PATH.read_bytes())
    fingerprint = digest.hexdigest()

    cache_root = Path(cache_dir or (repo_root / "data/cache/supermimic/humanoid_light"))
    destination = cache_root / fingerprint
    if (destination / "index.json").is_file():
        return destination

    cache_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[Path, list[dict]] = {}
    for rec in manifest["splits"][split]:
        p = (repo_root / rec["path"]).resolve()
        grouped.setdefault(p, []).append(rec)

    chunks = []
    for p in sorted(grouped):
        for rec in grouped[p]:
            start, stop = int(rec["start_frame"]), int(rec["stop_frame"])
            for first in range(start, stop, chunk_frames):
                last = min(first + chunk_frames, stop)
                if last - first >= 2:
                    chunks.append((p, first, last, rec))

    def rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        uv = 2.0 * np.cross(q[:, 1:], v)
        return v + q[:, :1] * uv + np.cross(q[:, 1:], uv)

    fields_lists: dict[str, list[np.ndarray]] = {
        k: []
        for k in ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel")
    }
    ends, names, sources = [], [], []
    offset = 0
    prev_path: Path | None = None
    full_q: np.ndarray | None = None
    full_v: np.ndarray | None = None
    for p, start, stop, rec in chunks:
        if p != prev_path:
            with np.load(p, allow_pickle=False) as npz:
                full_q = np.asarray(npz["qpos"], dtype=np.float32)
                full_v = np.asarray(npz["qvel"], dtype=np.float32)
            prev_path = p
        assert full_q is not None and full_v is not None
        q = full_q[start:stop]
        v = full_v[start:stop]
        fields_lists["root_pos_w"].append(q[:, :3])
        fields_lists["root_quat_w"].append(q[:, 3:7])
        fields_lists["root_lin_vel_w"].append(v[:, :3])
        fields_lists["root_ang_vel_w"].append(rotate_np(q[:, 3:7], v[:, 3:6]).astype(np.float32))
        fields_lists["joint_pos"].append(q[:, 7:])
        fields_lists["joint_vel"].append(v[:, 6:])
        offset += stop - start
        ends.append(offset)
        names.append(f"{rec['source']}/{p.stem}/{start}-{stop}")
        sources.append({"path": rec["path"], "start": start, "stop": stop})

    destination.mkdir(parents=True, exist_ok=True)
    for k, arrs in fields_lists.items():
        np.save(destination / f"{k}.npy", np.concatenate(arrs, axis=0))
    np.save(destination / "motion_ends.npy", np.asarray(ends, dtype=np.int64))
    (destination / "joint_names.json").write_text(json.dumps(list(JOINT_NAMES)))
    (destination / "motion_names.json").write_text(json.dumps(names))
    meta = {
        "version": 1,
        "fingerprint": fingerprint,
        "split": split,
        "frames": offset,
        "clips": len(ends),
        "sources": sources,
        "frequency": 50,
    }
    (destination / "index.json").write_text(json.dumps(meta, indent=2))
    return destination


def build_motion_actor_goal(
    motion: PackedMotionLibrary,
    motion_times: torch.Tensor,
    root_pos_local: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Build 580-D actor_goal: [window_joint_pos_vel(520), window_rel_ori_rot6d(60)]."""
    clip = torch.searchsorted(motion.motion_ends, motion_times, right=True)
    clip = clip.clamp(max=motion.motion_ends.numel() - 1)
    last_valid = motion.motion_ends[clip] - 1
    offsets = (
        torch.arange(NUM_GOAL_FRAMES, device=motion_times.device) * GOAL_FRAME_STRIDE
    )
    idx = torch.minimum(motion_times[:, None] + offsets, last_valid[:, None])

    content = torch.cat(
        [motion.joint_pos[idx], motion.joint_vel[idx]], dim=-1
    ).reshape(motion_times.shape[0], -1)

    target_quat = motion.root_quat_w[idx]  # (E, 10, 4)
    num_envs = motion_times.shape[0]
    root_q_exp = root_quat_wxyz[:, None, :].expand(num_envs, NUM_GOAL_FRAMES, 4)
    q10 = quat_inv(root_q_exp.reshape(-1, 4))
    rel_ori_w = quat_mul(q10, target_quat.reshape(-1, 4))
    rel_ori_rot6d = quat_to_rot6d(rel_ori_w).reshape(num_envs, -1)
    return torch.cat([content, rel_ori_rot6d], dim=-1)


# ---------------------------------------------------------------------------
# Action Scale & Coupled-PD Motor-Coordinate Target Mapping
# ---------------------------------------------------------------------------


def coupling_matrix(pair: str, g1: float, g2: float) -> tuple[tuple[float, float], tuple[float, float]]:
    if pair == "left_ankle":
        return ((-g1, g1), (g2, g2))
    if pair == "right_ankle":
        return ((g1, g1), (-g2, g2))
    if pair == "torso":
        return ((-g1, g1), (-g2, -g2))
    raise ValueError(f"Unknown coupled pair {pair!r}")


def compute_coupled_action_scale_tensor(
    action_scale_factor: float = 0.25,
    control_json: Path = CONTROL_JSON_PATH,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    source = json.loads(Path(control_json).read_text())
    stiffness = {n: float(source["joints"][n]["stiffness"]) for n in JOINT_NAMES}
    effort_limit = {n: float(source["joints"][n]["effort_limit"]) for n in JOINT_NAMES}
    effective = dict(stiffness)
    for pair, group_name in (
        ("left_ankle", "ankles"),
        ("right_ankle", "ankles"),
        ("torso", "torso_pitch_roll"),
    ):
        grp = source["groups"][group_name]
        mat = coupling_matrix(pair, float(grp["gear_ratio_1"]), float(grp["gear_ratio_2"]))
        gamma = float(grp["gamma"])
        for col, suffix in enumerate(("pitch", "roll")):
            effective[f"{pair}_{suffix}_joint"] *= gamma * sum(row[col] ** 2 for row in mat)
    scales = [action_scale_factor * effort_limit[n] / effective[n] for n in JOINT_NAMES]
    return torch.tensor(scales, dtype=torch.float32, device=device).unsqueeze(0)


def parse_mjcf_joint_limits(
    mjcf_path: Path = MJCF_XML_PATH, device: str | torch.device = "cpu"
) -> torch.Tensor:
    import xml.etree.ElementTree as ET

    root = ET.parse(mjcf_path).getroot()
    ranges: dict[str, tuple[float, float]] = {}
    for elem in root.iter("joint"):
        name = elem.attrib.get("name")
        if name in JOINT_NAMES and "range" in elem.attrib:
            lo, hi = map(float, elem.attrib["range"].split())
            ranges[name] = (lo, hi)
    ordered = [ranges[n] for n in JOINT_NAMES]
    return torch.tensor(ordered, dtype=torch.float32, device=device)


def raw_action_to_joint_targets_mj(
    raw_action: torch.Tensor,
    env_settings: dict[str, Any],
    *,
    scale_tensor: torch.Tensor | None = None,
    limits_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert raw policy output (E, 26) to joint-space PD position targets (E, 26) in JOINT_NAMES order."""
    mode = env_settings.get("action_mode", "joint_range")
    if mode == "coupled_scale":
        if scale_tensor is None:
            scale_tensor = compute_coupled_action_scale_tensor(
                float(env_settings.get("action_scale", 0.25)), device=raw_action.device
            )
        return raw_action * scale_tensor
    if mode == "joint_range":
        if limits_tensor is None:
            limits_tensor = parse_mjcf_joint_limits(device=raw_action.device)
        clamped = raw_action.clamp(-1.0, 1.0)
        span = torch.where(clamped >= 0.0, limits_tensor[:, 1], -limits_tensor[:, 0])
        return clamped * span
    raise ValueError(f"Unknown action_mode: {mode!r}")


def joint_targets_mj_to_isaac_actuator_slots(
    q_target_mj: torch.Tensor,
    control_json: Path = CONTROL_JSON_PATH,
) -> torch.Tensor:
    """Convert joint-space PD targets (E, 26) in JOINT_NAMES order to the targets expected by
    HUMANOID_LIGHT_CFG's actuators in cocelo-rl-isaaclab.

    Uncoupled DelayedPDActuator joints take joint-space targets directly.
    CoupledDelayedPDActuator joints (left_ankle, right_ankle, torso_pitch_roll) expect
    [motor_1_target, motor_2_target] = J @ [pitch_target, roll_target] in their [pitch, roll] slots.
    """
    source = json.loads(Path(control_json).read_text())
    out = q_target_mj.clone()
    name_to_idx = {n: i for i, n in enumerate(JOINT_NAMES)}

    for pair, group_name in (
        ("left_ankle", "ankles"),
        ("right_ankle", "ankles"),
        ("torso", "torso_pitch_roll"),
    ):
        grp = source["groups"][group_name]
        g1 = float(grp["gear_ratio_1"])
        g2 = float(grp["gear_ratio_2"])
        p_idx = name_to_idx[f"{pair}_pitch_joint"]
        r_idx = name_to_idx[f"{pair}_roll_joint"]
        pitch = q_target_mj[:, p_idx]
        roll = q_target_mj[:, r_idx]
        if pair == "left_ankle":
            m1 = g1 * (roll - pitch)
            m2 = g2 * (roll + pitch)
        elif pair == "right_ankle":
            m1 = g1 * (roll + pitch)
            m2 = g2 * (roll - pitch)
        elif pair == "torso":
            m1 = g1 * (roll - pitch)
            m2 = -g2 * (roll + pitch)
        else:
            raise AssertionError(pair)
        out[:, p_idx] = m1
        out[:, r_idx] = m2

    return out


def export_torchscript_bundle(checkpoint_path: Path, export_path: Path) -> dict[str, Any]:
    saved, ckpt_file, sha = load_checkpoint_compat(checkpoint_path, device="cpu")
    actor = build_standalone_actor_from_saved(saved, device="cpu")
    scripted = torch.jit.script(actor)
    export_path = Path(export_path).expanduser().resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(export_path))
    export_sha = hashlib.sha256(export_path.read_bytes()).hexdigest()
    meta = {
        "source_checkpoint": str(ckpt_file),
        "source_checkpoint_sha256": sha,
        "exported_torchscript": str(export_path),
        "exported_torchscript_sha256": export_sha,
        "progress": saved.get("progress", {}),
        "environment": saved["settings"]["environment"],
        "joint_names": list(JOINT_NAMES),
    }
    meta_path = export_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export-torchscript", type=Path, required=True)
    args = parser.parse_args()
    meta = export_torchscript_bundle(args.checkpoint, args.export_torchscript)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
