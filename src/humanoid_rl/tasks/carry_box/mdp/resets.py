"""Jointly sample a physically consistent box and support layout on reset."""

import torch
from mjlab.envs.mdp import reset_joints_by_offset, reset_root_state_uniform
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import quat_from_euler_xyz


def sample_range(bounds, count, device):
    return torch.empty(count, device=device).uniform_(*bounds)


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def reset_carry_scene(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n, device = len(env_ids), env.device
    cfg = env.cfg.carry
    origins = env.scene.env_origins[env_ids]
    box = env.scene["box"]
    gid = box.indexing.geom_ids[0]
    half = env.sim.model.geom_size[env_ids, gid]
    mass = sample_range(cfg.box_mass, n, device)
    bid = box.indexing.body_ids[0]
    env.sim.model.body_mass[env_ids, bid] = mass
    # Solid cuboid inertia for the sampled dimensions, centered in its body frame.
    squared = (2 * half).square()
    env.sim.model.body_inertia[env_ids, bid] = (
        mass[:, None]
        / 12
        * torch.stack(
            (
                squared[:, 1] + squared[:, 2],
                squared[:, 0] + squared[:, 2],
                squared[:, 0] + squared[:, 1],
            ),
            dim=-1,
        )
    )
    source = torch.zeros(n, 3, device=device)
    source[:, 0] = sample_range(cfg.source_x, n, device)
    source[:, 1] = sample_range(cfg.source_y, n, device)
    source[:, 2] = sample_range(cfg.support_top_height, n, device)
    target = source.clone()
    target[:, 0] += sample_range(cfg.target_dx, n, device)
    target[:, 1] += sample_range(cfg.target_dy, n, device)
    target[:, 2] = sample_range(cfg.support_top_height, n, device)
    identity = torch.zeros(n, 4, device=device)
    identity[:, 0] = 1
    for name, top in (("source_support", source), ("target_support", target)):
        entity = env.scene[name]
        support_half = env.sim.model.geom_size[env_ids, entity.indexing.geom_ids[0]]
        center = top + origins
        center[:, 2] -= support_half[:, 2]
        entity.write_mocap_pose_to_sim(
            torch.cat((center, identity), -1), env_ids=env_ids
        )

    # Bound the center offset so even a yawed box stays wholly on the source.
    yaw = sample_range(cfg.box_yaw, n, device)
    extent = torch.stack(
        (
            yaw.cos().abs() * half[:, 0] + yaw.sin().abs() * half[:, 1],
            yaw.sin().abs() * half[:, 0] + yaw.cos().abs() * half[:, 1],
        ),
        -1,
    )
    source_half = env.sim.model.geom_size[
        env_ids, env.scene["source_support"].indexing.geom_ids[0], :2
    ]
    clearance = (source_half - extent - 0.005).clamp_min(0)
    offset = torch.empty(n, 2, device=device).uniform_(-1, 1)
    offset *= torch.minimum(clearance, torch.full_like(clearance, cfg.box_xy_offset))
    position = source + origins
    position[:, :2] += offset
    position[:, 2] += half[:, 2] + 0.001
    quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    box.write_root_state_to_sim(
        torch.cat((position, quat, torch.zeros(n, 6, device=device)), -1),
        env_ids=env_ids,
    )
    if not hasattr(env, "carry_goal_w"):
        env.carry_goal_w = torch.zeros(env.num_envs, 3, device=device)
    goal = target + origins
    goal[:, 2] += half[:, 2]
    env.carry_goal_w[env_ids] = goal
    reset_root_state_uniform(env, env_ids, pose_range={"yaw": cfg.robot_yaw})
    reset_joints_by_offset(
        env, env_ids, position_range=(-0.01, 0.01), velocity_range=(0, 0)
    )
