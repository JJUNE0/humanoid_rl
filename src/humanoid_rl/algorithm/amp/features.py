"""Identical heading-invariant features for expert motions and mjlab states."""

import numpy as np
import torch
from robots.humanoid.g1.config import KEY_BODY_NAMES
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    yaw_quat,
)

FEATURE_VERSION = "g1-body83-v1"
KEY_BODIES = KEY_BODY_NAMES
FEATURE_GROUPS = {
    "root_height": 1,
    "root_rotation_6d": 6,
    "root_linear_velocity_heading": 3,
    "root_angular_velocity_heading": 3,
    "joint_position": 29,
    "joint_velocity": 29,
    "hand_foot_position_heading": 12,
}
FEATURE_DIM = sum(FEATURE_GROUPS.values())
OBJECT_FEATURE_VERSION = "g1-body-object104-v1"
OBJECT_FEATURE_GROUPS = {
    "box_position_heading": 3,
    "box_rotation_heading_6d": 6,
    "box_relative_linear_velocity_heading": 3,
    "box_relative_angular_velocity_heading": 3,
    "hand_to_box_position_heading": 6,
}
OBJECT_FEATURE_DIM = FEATURE_DIM + sum(OBJECT_FEATURE_GROUPS.values())
# Robot71, box geometry pose/velocity13, dimensions3, mass1,
# source/target support top XYZ6, desired box geometry centre XYZ3.
OBJECT_REFERENCE_DIM = 97


def feature_schema(observe_object=False):
    if observe_object:
        return OBJECT_FEATURE_VERSION, FEATURE_GROUPS | OBJECT_FEATURE_GROUPS
    return FEATURE_VERSION, dict(FEATURE_GROUPS)


def object_features(root_pos, root_quat, root_linear_w, root_angular_w,
                    object_pos, object_quat, object_linear_w, object_angular_w,
                    hand_positions_w):
    """Geometry-centred box interaction features in the robot heading frame.

    Velocity differences are rotated world-frame velocity differences, not
    derivatives in a rotating frame. Hands are the same wrist link origins
    used by body_features, retargeted through native mjlab FK on both sides.
    """
    heading = yaw_quat(root_quat)
    rotation = matrix_from_quat(quat_mul(quat_conjugate(heading), object_quat))[..., :2].flatten(-2)
    hand_heading = heading.unsqueeze(-2).expand(*hand_positions_w.shape[:-1], 4)
    return torch.cat((
        quat_apply_inverse(heading, object_pos - root_pos), rotation,
        quat_apply_inverse(heading, object_linear_w - root_linear_w),
        quat_apply_inverse(heading, object_angular_w - root_angular_w),
        quat_apply_inverse(hand_heading, object_pos.unsqueeze(-2) - hand_positions_w).flatten(-2),
    ), -1)


def body_features(
    root_pos,
    root_quat,
    linear_velocity_w,
    angular_velocity_w,
    joint_pos,
    joint_vel,
    key_body_pos_w,
):
    """Inputs use metres, radians, seconds and wxyz quaternions; output [..., 83]."""
    heading = yaw_quat(root_quat)
    rotation = matrix_from_quat(quat_mul(quat_conjugate(heading), root_quat))[
        ..., :2
    ].flatten(-2)
    body_heading = heading.unsqueeze(-2).expand(*key_body_pos_w.shape[:-1], 4)
    local_key_pos = quat_apply_inverse(
        body_heading, key_body_pos_w - root_pos.unsqueeze(-2)
    )
    return torch.cat(
        (
            root_pos[..., 2:3],
            rotation,
            quat_apply_inverse(heading, linear_velocity_w),
            quat_apply_inverse(heading, angular_velocity_w),
            joint_pos,
            joint_vel,
            local_key_pos.flatten(-2),
        ),
        dim=-1,
    )


class EnvironmentFeatures:
    def __init__(self, env, observe_object=False):
        robot = env.scene["robot"]
        if robot.num_joints != 29:
            raise ValueError("AMP feature schema requires the 29-joint G1")
        self.joint_names = tuple(robot.joint_names)
        self.body_ids, _ = robot.find_bodies(KEY_BODIES, preserve_order=True)
        self.observe_object = observe_object
        self.feature_version, self.feature_groups = feature_schema(observe_object)
        self.feature_dim = sum(self.feature_groups.values())

    def __call__(self, env):
        data = env.scene["robot"].data
        origins = env.scene.env_origins
        features = body_features(
            data.root_link_pos_w - origins,
            data.root_link_quat_w,
            data.root_link_lin_vel_w,
            data.root_link_ang_vel_w,
            data.joint_pos,
            data.joint_vel,
            data.body_link_pos_w[:, self.body_ids] - origins[:, None, :],
        )
        if self.observe_object:
            box = env.scene["box"].data
            interaction = object_features(
                data.root_link_pos_w, data.root_link_quat_w,
                data.root_link_lin_vel_w, data.root_link_ang_vel_w,
                box.root_link_pos_w, box.root_link_quat_w,
                box.root_link_lin_vel_w, box.root_link_ang_vel_w,
                data.body_link_pos_w[:, self.body_ids[:2]],
            )
            features = torch.cat((features, interaction), -1)
        return features


class ExpertFeatures:
    """Encode G1 joint motions with the same model used by mjlab simulation."""

    def __init__(self, joint_names, observe_object=False):
        import mujoco
        from robots.humanoid.g1.config import get_spec

        self.model = get_spec().compile()
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(joint_names)
        self.qpos_ids = [
            self.model.jnt_qposadr[self.model.joint(name).id] for name in joint_names
        ]
        self.body_ids = [self.model.body(name).id for name in KEY_BODIES]
        self.observe_object = observe_object

    def encode(self, motion, scene=None):
        import mujoco

        names = motion["joint_names"].tolist()
        if (
            len(names) != 29
            or len(set(names)) != 29
            or set(names) != set(self.joint_names)
        ):
            raise ValueError("Expert joint_names must match the 29 mjlab G1 joints")
        order = [names.index(name) for name in self.joint_names]
        q = np.asarray(motion["dof_pos"][:, order], dtype=np.float32)
        qd = np.asarray(motion["dof_vel"][:, order], dtype=np.float32)
        pos = np.asarray(motion["root_pos"], dtype=np.float32)
        quat = np.asarray(motion["root_quat"], dtype=np.float32)
        key_pos = np.empty((len(pos), len(KEY_BODIES), 3), np.float32)
        for i in range(len(pos)):
            self.data.qpos[:3], self.data.qpos[3:7] = pos[i], quat[i]
            self.data.qpos[self.qpos_ids] = q[i]
            mujoco.mj_kinematics(self.model, self.data)
            key_pos[i] = self.data.xpos[self.body_ids]
        pos, quat, q, qd, key_pos = map(torch.from_numpy, (pos, quat, q, qd, key_pos))
        linear = torch.from_numpy(np.asarray(motion["root_lin_vel"], np.float32))
        angular_local = torch.from_numpy(
            np.asarray(motion["root_ang_vel_local"], np.float32)
        )
        features = body_features(
            pos, quat, linear, quat_apply(quat, angular_local), q, qd, key_pos
        )
        # Raw reference states also support optional robot-only reference resets.
        references = torch.cat((pos, quat, linear, angular_local, q, qd), -1)
        if self.observe_object:
            if scene is None:
                raise ValueError("Object AMP needs the episode's box/support geometry")
            tensor = lambda value: torch.as_tensor(np.asarray(value), dtype=torch.float32)
            body_pos, body_quat = tensor(motion["object_pos"]), tensor(motion["object_quat"])
            geom_pos = tensor(scene["geom_position"]).expand(len(pos), -1)
            geom_quat = tensor(scene["geom_quaternion"]).expand(len(pos), -1)
            offset_w = quat_apply(body_quat, geom_pos)
            box_pos = body_pos + offset_w
            box_quat = quat_mul(body_quat, geom_quat)
            box_angular = quat_apply(body_quat, tensor(motion["object_ang_vel_local"]))
            # Recorded free-joint qvel is velocity at the body origin. Native
            # mjlab box origin is its geometry centre, so shift the velocity.
            box_linear = tensor(motion["object_lin_vel"]) + torch.cross(box_angular, offset_w, dim=-1)
            features = torch.cat((features, object_features(
                pos, quat, linear, quat_apply(quat, angular_local),
                box_pos, box_quat, box_linear, box_angular, key_pos[:, :2],
            )), -1)
            size = tensor(motion["box_size_xyz_m"])
            mass = tensor(motion["box_mass_kg"])
            if size.shape != (3,) or not (size > 0).all() or mass.numel() != 1 or mass <= 0:
                raise ValueError("Object AMP needs positive per-episode box dimensions/mass")
            # The recorded goal is the original free-body position for an
            # upright world-aligned geometry, not that geometry's centre.
            goal = tensor(motion["goal"]) + quat_apply(quat_conjugate(geom_quat[0]), geom_pos[0])
            references = torch.cat((references, box_pos, box_quat, box_linear, box_angular,
                                    size.expand(len(pos), -1), mass.reshape(1, 1).expand(len(pos), 1),
                                    tensor(scene["source_top"]).expand(len(pos), -1),
                                    tensor(scene["target_top"]).expand(len(pos), -1),
                                    goal.expand(len(pos), -1)), -1)
            if references.shape[-1] != OBJECT_REFERENCE_DIM:
                raise AssertionError("Unexpected object reference layout")
        return features, references
