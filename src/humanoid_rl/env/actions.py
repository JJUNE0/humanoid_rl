"""Bound enlarged SAC position targets by the model's physical joint limits."""

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction


def joint_range_targets(action, offset, limits):
    """Physical joint targets shared by simulation and learned Light dynamics."""
    action = action.clamp(-1., 1.)
    span = torch.where(action >= 0, limits[..., 1] - offset, offset - limits[..., 0])
    return offset + action * span


class LimitedJointPositionAction(JointPositionAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        limits = self._entity.data.joint_pos_limits[:, self.target_ids]
        self._processed_actions = torch.clamp(
            self._processed_actions, limits[..., 0], limits[..., 1]
        )


@dataclass(kw_only=True)
class LimitedJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env):
        return LimitedJointPositionAction(self, env)


class JointRangePositionAction(JointPositionAction):
    """Zero holds the default pose; +/-1 reaches each asymmetric joint limit."""

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        limits = self._entity.data.joint_pos_limits[:, self.target_ids]
        self._processed_actions = joint_range_targets(actions, self._offset, limits)


@dataclass(kw_only=True)
class JointRangePositionActionCfg(JointPositionActionCfg):
    def build(self, env):
        return JointRangePositionAction(self, env)


class ReferenceJointPositionAction(JointPositionAction):
    """Teacher mapping: default + clip(raw*5, -5, 5) * coefficient*effort/kp."""

    def _find_targets(self, cfg):
        return self._entity.find_joints(cfg.actuator_names, preserve_order=True)

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        scaled = (actions*self.cfg.normalize_action_to).clamp(-self.cfg.action_clip, self.cfg.action_clip)
        targets = self._offset+scaled*self._scale
        limits = self._entity.data.joint_pos_limits[:, self.target_ids]
        self._processed_actions = torch.clamp(targets, limits[..., 0], limits[..., 1])

    def apply_actions(self):
        super().apply_actions()
        zeros = torch.zeros_like(self._processed_actions)
        self._entity.set_joint_velocity_target(zeros, joint_ids=self.target_ids)
        self._entity.set_joint_effort_target(zeros, joint_ids=self.target_ids)


@dataclass(kw_only=True)
class ReferenceJointPositionActionCfg(JointPositionActionCfg):
    normalize_action_to: float = 5.0
    action_clip: float = 5.0

    def build(self, env):
        return ReferenceJointPositionAction(self, env)
