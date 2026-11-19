"""Deliberately coarse task conditions, to refine after the environment works."""

import torch


def placed(env):
    box = env.scene["box"].data
    error = torch.linalg.vector_norm(box.root_link_pos_w - env.carry_goal_w, dim=-1)
    speed = torch.linalg.vector_norm(box.root_link_lin_vel_w, dim=-1)
    supported = env.scene["box_target_contact"].data.found.flatten(1).any(-1)
    held = env.scene["hand_box_contact"].data.found.flatten(1).any(-1)
    return (error < 0.2) & (speed < 0.2) & supported & ~held


def fallen(env):
    robot = env.scene["robot"].data
    height = robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (height < 0.35) | (robot.projected_gravity_b[:, 2] > -0.2)
