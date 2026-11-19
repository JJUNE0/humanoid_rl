"""Reconstruct the teacher's input conventions from native mjlab state."""

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

from humanoid_rl.tasks.carry_box.stage.config import validate_schema
from humanoid_rl.tasks.carry_box.stage.model import FrozenCarryPredictor


def state101(root_pos, root_quat, root_ang_vel_w, joint_pos, joint_vel, defaults,
             object_pos, object_quat, object_com_velocity_w, object_ang_velocity_w,
             goal_pos, hand_points, hand_contact):
    """Match carry_stage_state_v2.actor_state, including full-root coordinates."""
    rotation = matrix_from_quat(quat_mul(quat_conjugate(root_quat), object_quat))
    gravity = torch.zeros_like(root_pos)
    gravity[:, 2] = -1
    hand_quat = root_quat[:, None].expand(-1, 2, -1)
    return torch.cat((
        joint_pos - defaults, joint_vel,
        quat_apply_inverse(root_quat, gravity),
        .25 * quat_apply_inverse(root_quat, root_ang_vel_w),
        quat_apply_inverse(root_quat, object_pos - root_pos),
        rotation[:, :, 0], rotation[:, :, 2],
        quat_apply_inverse(root_quat, object_com_velocity_w),
        quat_apply_inverse(root_quat, object_ang_velocity_w),
        quat_apply_inverse(root_quat, goal_pos - object_pos),
        hand_contact.float(),
        quat_apply_inverse(hand_quat, hand_points - root_pos[:, None]).flatten(1),
        quat_apply_inverse(hand_quat, hand_points - object_pos[:, None]).flatten(1),
        torch.ones(len(root_pos), 2, device=root_pos.device),
        root_pos,
    ), -1)


class StageObserver:
    def __init__(self, env, cfg):
        validate_schema(cfg.schema)
        self.cfg = cfg
        self.predictor = FrozenCarryPredictor(cfg, env.device)
        robot = env.scene["robot"]
        self.joints, _ = robot.find_joints(cfg.schema["joint_names"], preserve_order=True)
        self.hands, _ = robot.find_bodies(("left_wrist_yaw_link", "right_wrist_yaw_link"), preserve_order=True)
        tensor = lambda key: torch.tensor(cfg.schema[key], device=env.device, dtype=torch.float32)
        self.defaults = tensor("default_joint_positions")
        self.hand_offsets = tensor("hand_contact_offsets")
        self.geom_position = tensor("box_geom_position")
        self.geom_quat = tensor("box_geom_quaternion_wxyz")
        self.com_position = tensor("box_com_position")
        # Each vector world has its own spatial origin. The classifier was
        # trained in a single world and must never receive tiled-world offsets.
        self.contact_groups = []
        names = env.scene["hand_box_contact"].primary_names
        for side in ("left", "right"):
            group = [i for i, name in enumerate(names) if f"{side}_hand" in name]
            if not group:
                raise ValueError(f"Cannot resolve {side} hand-box contact from {names}")
            self.contact_groups.append(group)
        self.last_state = self.last_context = None

    @torch.no_grad()
    def raw_state(self, env):
        robot, box = env.scene["robot"].data, env.scene["box"].data
        # Current box body == geometry centre/frame. Build a virtual original
        # body so upright boxes have exactly the teacher's quaternion/offset.
        geom_quat = self.geom_quat.expand(env.num_envs, -1)
        object_quat = quat_mul(box.root_link_quat_w, quat_conjugate(geom_quat))
        geom_offset_w = quat_apply(object_quat, self.geom_position.expand(env.num_envs, -1))
        object_pos = box.root_link_pos_w - geom_offset_w
        com_delta_w = quat_apply(object_quat, (self.com_position - self.geom_position).expand(env.num_envs, -1))
        object_com_velocity_w = box.root_link_lin_vel_w + torch.cross(box.root_link_ang_vel_w, com_delta_w, dim=-1)
        # The goal describes an upright, world-aligned box geometry. Convert
        # its centre to the same virtual teacher body convention.
        goal_offset_w = quat_apply(quat_conjugate(geom_quat), self.geom_position.expand(env.num_envs, -1))
        goal = env.carry_goal_w - goal_offset_w
        hand_quat = robot.body_link_quat_w[:, self.hands]
        points = robot.body_link_pos_w[:, self.hands] + quat_apply(hand_quat, self.hand_offsets[None].expand(env.num_envs, -1, -1))
        found = env.scene["hand_box_contact"].data.found.flatten(1)
        contact = torch.stack([found[:, group].any(-1) for group in self.contact_groups], -1)
        origin = env.scene.env_origins
        return state101(
            robot.root_link_pos_w - origin, robot.root_link_quat_w, robot.root_link_ang_vel_w,
            robot.joint_pos[:, self.joints], robot.joint_vel[:, self.joints], self.defaults,
            object_pos - origin, object_quat, object_com_velocity_w, box.root_link_ang_vel_w,
            goal - origin, points - origin[:, None], contact,
        )

    @torch.no_grad()
    def __call__(self, env):
        self.last_state = self.raw_state(env)
        self.last_context = self.predictor(self.last_state)
        return self.last_context

    @torch.no_grad()
    def metrics(self):
        context = self.last_context
        if context is None:
            return {}
        normalized = (self.last_state - self.predictor.input_mean) / self.predictor.input_std
        return {"stage/failure_probability_mean": context[:, 27].mean().item(),
                "stage/confidence_mean": context[:, :9].amax(-1).mean().item(),
                "stage/weighted_progress_mean": (context[:, :9] * context[:, 18:27]).sum(-1).mean().item(),
                "stage/input_clip_fraction": (normalized.abs() > 20).float().mean().item()}


def frozen_stage_context(env):
    if not hasattr(env, "carry_stage_observer"):
        env.carry_stage_observer = StageObserver(env, env.cfg.stage_model)
    return env.carry_stage_observer(env)
