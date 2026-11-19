"""Reference initialization and robot-configured auxiliary physical control costs."""

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_apply_inverse, quat_mul

from humanoid_rl.env.bfm_features import from_environment


def actor_state(env):
    return from_environment(env)["state"]


def original_time_out(env):
    return env.episode_length_buf > env.max_episode_length


def reset_reference(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "motion_dataset"):
        # Scene-only playback can use the neutral pose without loading motions.
        robot = env.scene["robot"]
        root = robot.data.default_root_state[env_ids].clone()
        root[:, :3] += env.scene.env_origins[env_ids]
        robot.write_root_state_to_sim(root, env_ids=env_ids)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos[env_ids],
                                      robot.data.default_joint_vel[env_ids], env_ids=env_ids)
        return
    qpos, qvel = env.motion_dataset.sample_states(len(env_ids), env.device)
    root = torch.cat((qpos[:, :7].clone(), qvel[:, :3], quat_apply(qpos[:, 3:7], qvel[:, 3:6])), dim=-1)
    root[:, :2] = 0
    root[:, :3] += env.scene.env_origins[env_ids]
    lying = torch.rand(len(env_ids), device=env.device) < env.cfg.motion_prior.lie_down_probability
    if lying.any():
        count = int(lying.sum())
        profile = env.cfg.motion_prior_agent
        root[lying, 2] = env.scene.env_origins[env_ids[lying], 2]+profile.lying_root_height
        if profile.lying_reset_mode == "reference":
            # Original BFM retains the sampled motion pose/velocity and applies
            # a world-X +/-90-degree rotation to the reference root rotation.
            sign = 1 if torch.rand((), device=env.device) < .5 else -1
            roll = torch.full((count,), sign*torch.pi/2, device=env.device)
            zero = torch.zeros_like(roll)
            root[lying, 3:7] = quat_mul(quat_from_euler_xyz(roll, zero, zero), root[lying, 3:7])
        else:
            roll = torch.zeros(count, device=env.device)
            pitch = torch.full_like(roll, torch.pi/2)
            yaw = torch.empty_like(roll).uniform_(-torch.pi, torch.pi)
            root[lying, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
            root[lying, 7:] = 0
            qpos[lying, 7:] = env.scene["robot"].data.default_joint_pos[env_ids[lying]]
            qvel[lying] = 0
    robot = env.scene["robot"]
    robot.write_root_state_to_sim(root, env_ids=env_ids)
    robot.write_joint_state_to_sim(qpos[:, 7:], qvel[:, 6:], env_ids=env_ids)


def fallen(env):
    data = env.scene["robot"].data
    return ((data.root_link_pos_w[:, 2]-env.scene.env_origins[:, 2] < 0.2)
            | (data.projected_gravity_b[:, :2].abs() > 0.8).any(-1))


class AuxiliaryRewards:
    def __init__(self, env):
        robot = env.scene["robot"]
        self.profile = env.cfg.motion_prior_agent
        self.settings = self.profile.rewards
        if tuple(robot.joint_names) != self.profile.joint_names:
            raise ValueError("Motion-prior joint order differs from the simulated robot")
        sensor_names = tuple(name.split("/")[-1] for name in env.scene["prior_feet"].primary_names)
        if set(sensor_names) != set(self.profile.feet_body_names):
            raise ValueError("Foot contact sensor bodies differ from the robot configuration")
        self.feet, _ = robot.find_bodies(sensor_names, preserve_order=True)
        if len(self.feet) != 2:
            raise ValueError("Motion prior requires exactly two foot contact primaries")
        self.ankle_roll, _ = robot.find_joints(self.profile.ankle_roll_joint_names, preserve_order=True) if self.profile.ankle_roll_joint_names else ([], [])
        self.action_scale = float(env.cfg.robot_control["normalize_action_to"])

    @torch.no_grad()
    def __call__(self, env, action, previous_action):
        data = env.scene["robot"].data
        force = env.scene["prior_feet"].data.force
        contact = force.norm(dim=-1) > self.settings.contact_force_threshold
        if self.settings.feet_orientation_contact == "vertical":
            # Original BFM uses upward force on the foot > threshold. mjlab's
            # netforce uses the primary->secondary convention, hence the sign.
            orientation_contact = (force[..., 2]*self.settings.sensor_force_to_robot
                                   > self.settings.contact_force_threshold)
        else:
            orientation_contact = contact  # Explicit compatibility with old H1 runs.
        quat = data.body_link_quat_w[:, self.feet]
        gravity = torch.zeros(*quat.shape[:-1], 3, device=env.device)
        gravity[..., 2] = -1
        foot_gravity = quat_apply_inverse(quat, gravity)
        limits = data.joint_pos_limits
        centre = limits.mean(-1)
        half = self.settings.soft_joint_position_limit*(limits[..., 1]-limits[..., 0])/2
        violation = ((centre-half-data.joint_pos).clamp_min(0)+(data.joint_pos-centre-half).clamp_min(0)).sum(-1)
        costs = {
            "penalty_action_rate": ((action-previous_action)*self.action_scale).square().sum(-1),
            "penalty_feet_ori": (foot_gravity[..., :2].norm(dim=-1)*orientation_contact).sum(-1),
            "limits_dof_pos": violation,
            "penalty_slippage": (data.body_link_lin_vel_w[:, self.feet].norm(dim=-1)*contact).sum(-1),
            "penalty_undesired_contact": (env.scene["prior_undesired"].data.force.abs()
                                          > self.settings.contact_force_threshold).flatten(1).any(-1).float(),
        }
        if self.ankle_roll:
            costs["penalty_ankle_roll"] = data.joint_pos[:, self.ankle_roll].square().sum(-1)
        return {key: costs[key].unsqueeze(-1) for key in self.settings.weights}
