"""Actor observations: proprioception, relative box pose, goal, previous action."""

import torch
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    yaw_quat,
)

from robots.humanoid.g1.config import KEY_BODY_NAMES


def hand_foot_relative_positions(env):
    """Four link XYZs relative to root, with world heading removed (12D).

    Same frame and link origins as the AMP key-body features. Positions refer
    to wrist/ankle link origins, not palm/sole contact points.
    """
    robot = env.scene["robot"]
    if not hasattr(env, "_carry_key_body_ids"):
        env._carry_key_body_ids, _ = robot.find_bodies(KEY_BODY_NAMES, preserve_order=True)
    data = robot.data
    displacement = data.body_link_pos_w[:, env._carry_key_body_ids] - data.root_link_pos_w[:, None]
    heading = yaw_quat(data.root_link_quat_w)[:, None].expand(-1, len(KEY_BODY_NAMES), -1)
    return quat_apply_inverse(heading, displacement).flatten(1)


def proprioception(env):
    robot = env.scene["robot"].data
    return torch.cat(
        (
            robot.joint_pos - robot.default_joint_pos,
            robot.joint_vel - robot.default_joint_vel,
            robot.root_link_lin_vel_b,
            robot.root_link_ang_vel_b,
            robot.projected_gravity_b,
        ),
        -1,
    )


def box_relative_pose(env):
    robot, box = env.scene["robot"].data, env.scene["box"].data
    heading = yaw_quat(robot.root_link_quat_w)
    position = quat_apply_inverse(heading, box.root_link_pos_w - robot.root_link_pos_w)
    relative_quat = quat_mul(quat_conjugate(heading), box.root_link_quat_w)
    # Two rotation-matrix columns avoid quaternion sign ambiguity.
    rotation = matrix_from_quat(relative_quat)[..., :2].reshape(env.num_envs, 6)
    return torch.cat((position, rotation), -1)


def goal_relative_position(env):
    robot = env.scene["robot"].data
    # ObservationManager probes dimensions before the first reset event runs.
    if not hasattr(env, "carry_goal_w"):
        env.carry_goal_w = torch.zeros(env.num_envs, 3, device=env.device)
    return quat_apply_inverse(
        yaw_quat(robot.root_link_quat_w),
        env.carry_goal_w - robot.root_link_pos_w,
    )
