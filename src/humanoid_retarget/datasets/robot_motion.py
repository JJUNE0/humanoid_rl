"""Numeric-only reader for local LocoMuJoCo motion archives; no JAX dependency."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RobotMotion:
    qpos: np.ndarray
    qvel: np.ndarray
    joint_names: tuple[str, ...]
    fps: float
    split_points: np.ndarray

    @property
    def num_frames(self):
        return len(self.qpos)


def load_robot_motion(path):
    """Read root xyz/wxyz plus hinge joints in the archive's joint order.

Object-valued upstream metadata is deliberately left unread. Neither upstream
Python classes nor pickle deserialization are required for these numeric states.
The source sampling frequency is preserved; this function does not resample.
"""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        qpos, qvel = archive["qpos"], archive["qvel"]
        names = tuple(archive["joint_names"].tolist())
        fps = float(archive["frequency"])
        split_points = archive["split_points"]
    if (not names or len(set(names)) != len(names)
            or qpos.ndim != 2 or qvel.ndim != 2 or len(qpos) == 0
            or qpos.shape != (len(qpos), len(names) + 6)
            or qvel.shape != (len(qpos), len(names) + 5)):
        raise ValueError(f"Expected one floating root and named hinge joints: {path}")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid motion states or sampling frequency: {path}")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=-1), 1, atol=1e-4):
        raise ValueError(f"Non-unit root wxyz quaternions: {path}")
    if (split_points.ndim != 1 or len(split_points) < 2
            or split_points[0] != 0 or split_points[-1] != len(qpos)
            or not np.issubdtype(split_points.dtype, np.integer)
            or not (np.diff(split_points) > 0).all()):
        raise ValueError(f"Invalid trajectory boundaries: {path}")
    return RobotMotion(qpos, qvel, names, fps, split_points)
