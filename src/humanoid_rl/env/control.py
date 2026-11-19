"""I-BFM control costs, kept separate from the AMP imitation reward."""

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

TERM_NAMES = (
    "action_rate", "feet_orientation", "ankle_roll", "joint_position_limit",
    "slippage", "undesired_contact",
)


class ControlTerms:
    def __init__(self, env, cfg):
        self.cfg = cfg
        robot = env.scene["robot"]
        self.ankles, _ = robot.find_joints(".*_ankle_roll_joint")
        # Resolve in sensor order so contact masks and foot kinematics align.
        names = env.scene["control_feet"].primary_names
        self.feet, _ = robot.find_bodies(
            [name.split("/")[-1] for name in names], preserve_order=True
        )
        if len(self.feet) != 2:
            raise ValueError(f"Expected two foot contact primaries, got {names}")

    @torch.no_grad()
    def __call__(self, env, actions, previous_actions):
        data = env.scene["robot"].data
        force = env.scene["control_feet"].data.force
        contact = force.norm(dim=-1) > 1.0
        quat = data.body_link_quat_w[:, self.feet]
        gravity = torch.zeros_like(force)
        gravity[..., 2] = -1
        foot_gravity = quat_apply_inverse(quat.flatten(0, 1), gravity.flatten(0, 1)).reshape_as(force)
        # MuJoCo's contact-primary force sign can oppose the support force;
        # use force magnitude for contact gating, as for the original slip term.
        orientation = (foot_gravity[..., :2].norm(dim=-1) * contact).sum(-1)
        velocity = data.body_link_lin_vel_w[:, self.feet].norm(dim=-1)
        limits = data.joint_pos_limits
        center = limits.mean(-1)
        half = (limits[..., 1] - limits[..., 0]) * 0.95 / 2
        violation = ((center - half - data.joint_pos).clamp_min(0)
                     + (data.joint_pos - center - half).clamp_min(0)).sum(-1)
        undesired = (env.scene["control_undesired"].data.force.abs() > 1).flatten(1).any(-1).float()
        return torch.stack((
            ((actions - previous_actions) * self.cfg.action_rate_scale).square().sum(-1),
            orientation,
            data.joint_pos[:, self.ankles].square().sum(-1),
            violation,
            (velocity * contact).sum(-1),
            undesired,
        ), -1)
