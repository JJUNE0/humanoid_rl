"""Resample floating-base joint trajectories with quaternion SLERP."""

import numpy as np


def resample_robot_motion(qpos, qvel, source_fps, target_fps):
    if source_fps <= 0 or target_fps <= 0 or len(qpos) < 2 or len(qvel) != len(qpos):
        raise ValueError("Resampling needs aligned states and positive frequencies")
    source_time = np.arange(len(qpos), dtype=np.float64)/source_fps
    count = int(np.floor(source_time[-1]*target_fps+1e-8))+1
    target_time = np.arange(count, dtype=np.float64)/target_fps
    interpolate = lambda values: np.stack([
        np.interp(target_time, source_time, values[:, column])
        for column in range(values.shape[1])], axis=-1)
    positions, velocities = interpolate(qpos), interpolate(qvel)
    lower = np.minimum(np.floor(target_time*source_fps).astype(int), len(qpos)-2)
    fraction = (target_time-source_time[lower])*source_fps
    a, b = qpos[lower, 3:7].astype(np.float64), qpos[lower+1, 3:7].astype(np.float64)
    dot = np.sum(a*b, axis=-1)
    b = np.where((dot < 0)[:, None], -b, b)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    sine = np.sin(angle)
    near = sine < 1e-6
    denominator = np.where(near, 1, sine)
    wa, wb = np.sin((1-fraction)*angle)/denominator, np.sin(fraction*angle)/denominator
    wa, wb = np.where(near, 1-fraction, wa), np.where(near, fraction, wb)
    rotation = wa[:, None]*a+wb[:, None]*b
    positions[:, 3:7] = rotation/np.linalg.norm(rotation, axis=-1, keepdims=True)
    return positions.astype(np.float32), velocities.astype(np.float32)
