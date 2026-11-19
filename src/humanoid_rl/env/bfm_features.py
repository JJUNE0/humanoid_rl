"""Shared expert/simulator BFM observations in metre/radian/second/wxyz units."""

import torch
from mjlab.utils.lab_api.math import (
    matrix_from_quat, quat_apply, quat_apply_inverse, quat_conjugate, quat_mul, yaw_quat,
)


FEATURE_VERSION = "h1-bfm-state44-priv298-v1"


def proprioception(joint_pos, joint_vel, default_joint_pos, root_quat, root_angular_w):
    gravity = torch.zeros_like(root_angular_w)
    gravity[..., 2] = -1
    return torch.cat((joint_pos-default_joint_pos, joint_vel,
                      quat_apply_inverse(root_quat, gravity),
                      0.25*quat_apply_inverse(root_quat, root_angular_w)), dim=-1)


def privileged_state(body_pos, body_quat, body_linear_w, body_angular_w):
    """Original BFM max-local-self layout: height, relative XYZ, tangent/normal, velocities."""
    root_pos, root_quat = body_pos[..., 0, :], body_quat[..., 0, :]
    heading = yaw_quat(root_quat).unsqueeze(-2).expand_as(body_quat)
    relative_pos = quat_apply_inverse(heading, body_pos-root_pos.unsqueeze(-2))
    local_rotation = matrix_from_quat(quat_mul(quat_conjugate(heading), body_quat))
    # BFM quat_to_tan_norm uses the local X and Z axes, not the first two columns.
    rotation_6d = torch.cat((local_rotation[..., :, 0], local_rotation[..., :, 2]), dim=-1)
    return torch.cat((root_pos[..., 2:3], relative_pos[..., 1:, :].flatten(-2),
                      rotation_6d.flatten(-2),
                      quat_apply_inverse(heading, body_linear_w).flatten(-2),
                      quat_apply_inverse(heading, body_angular_w).flatten(-2)), dim=-1)


def from_environment(env):
    data = env.scene["robot"].data
    position = data.body_link_pos_w-env.scene.env_origins[:, None]
    rotation, linear, angular = data.body_link_quat_w, data.body_link_lin_vel_w, data.body_link_ang_vel_w
    profile = env.cfg.motion_prior_agent
    for virtual in profile.virtual_bodies:
        parent = env.scene["robot"].body_names.index(virtual["parent"])
        offset = quat_apply(rotation[:,parent],torch.tensor(virtual["position"],device=env.device).expand(env.num_envs,3))
        position = torch.cat((position,(position[:,parent]+offset)[:,None]),1)
        linear = torch.cat((linear,(linear[:,parent]+torch.cross(angular[:,parent],offset,dim=-1))[:,None]),1)
        angular = torch.cat((angular,angular[:,parent:parent+1]),1)
        rotation = torch.cat((rotation,rotation[:,parent:parent+1]),1)
    state = proprioception(data.joint_pos_biased, data.joint_vel, data.default_joint_pos,
                           data.root_link_quat_w, data.root_link_ang_vel_w)
    if env.cfg.motion_prior.observation_noise:
        n, scales = data.joint_pos.shape[-1], profile.noise_scales
        amplitude = torch.tensor([scales["joint_position"]]*n+[scales["joint_velocity"]]*n
                                 +[scales["gravity"]]*3+[scales["angular_velocity"]]*3,device=env.device)
        state = state+(2*torch.rand_like(state)-1)*amplitude
    return {
        "state": state,
        "privileged_state": privileged_state(position,rotation,linear,angular),
    }


class ObservationHistory:
    """Four preceding state/action pairs, grouped in the upstream sorted-key layout."""

    def __init__(self, num_envs, length, joint_dim, device):
        self.length, self.joint_dim = length, joint_dim
        self.states = torch.zeros(num_envs, length, 2*joint_dim+6, device=device)
        self.actions = torch.zeros(num_envs, length, joint_dim, device=device)

    def push(self, state, action):
        self.states = torch.roll(self.states, 1, dims=1)
        self.actions = torch.roll(self.actions, 1, dims=1)
        self.states[:, 0], self.actions[:, 0] = state, action

    def reset(self, ids):
        self.states[ids], self.actions[ids] = 0, 0

    def observation(self):
        n = self.joint_dim
        fields = (self.actions, self.states[..., 2*n+3:2*n+6], self.states[..., :n],
                  self.states[..., n:2*n], self.states[..., 2*n:2*n+3])
        return torch.cat([value.flatten(1) for value in fields], dim=-1)
