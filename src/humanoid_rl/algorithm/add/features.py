"""Frame-aligned residuals, including absolute robot/box trajectory tracking.

The existing AMP encoder supplies named joints, heading-frame velocities and
hand/foot/box features. ADD additionally tracks world root XYZ/rotation and box
XYZ relative to each world's origin. An absolute trajectory cannot be checked
using AMP's heading-invariant, root-relative features alone.
"""

from copy import copy
import hashlib

import torch
from mjlab.utils.lab_api.math import matrix_from_quat

from humanoid_rl.algorithm.amp.features import EnvironmentFeatures, FEATURE_GROUPS as BODY_GROUPS, OBJECT_FEATURE_GROUPS


FEATURE_VERSION = "g1-add-body-object109-v1"
FEATURE_GROUPS = {
    "root_position_world": 3,
    "root_rotation_world_6d": 6,
    **{name: width for name, width in BODY_GROUPS.items()
       if name not in ("root_height", "root_rotation_6d")},
    **OBJECT_FEATURE_GROUPS,
    "box_position_world": 3,
}
FEATURE_DIM = sum(FEATURE_GROUPS.values())


def reference_features(states, references):
    rotation = matrix_from_quat(references[:, 3:7])[..., :2].flatten(-2)
    return torch.cat((references[:, :3], rotation, states[:, 7:], references[:, 71:74]), -1)


def tracking_dataset(dataset):
    """Reuse the immutable expert loader/cache and preserve its source identity."""
    result = copy(dataset)
    result.train = copy(dataset.train)
    result.train.states = reference_features(dataset.train.states, dataset.train.references)
    result.train.fingerprint = hashlib.sha256(
        (FEATURE_VERSION + dataset.train.fingerprint).encode()
    ).hexdigest()
    if dataset.validation is not None or len(dataset.train.paths) != 1:
        raise ValueError("ADD currently expects exactly one selected reference clip")
    return result


class EnvironmentAddFeatures:
    """Return target(frame) - simulated(frame) before the adapter resets done worlds."""

    def __init__(self, env, reference):
        self.body = EnvironmentFeatures(env, observe_object=True)
        self.joint_names = self.body.joint_names
        self.reference = reference.to(device=env.device, dtype=torch.float32)
        self.feature_version, self.feature_groups = FEATURE_VERSION, FEATURE_GROUPS
        self.feature_dim = FEATURE_DIM
        if self.reference.ndim != 2 or self.reference.shape[1] != FEATURE_DIM:
            raise ValueError("ADD reference features must have shape [frames, 109]")
        if len(self.reference) != env.cfg.fixed_scene["frames"]:
            raise ValueError("ADD reference length and fixed scene differ")

    def current(self, env):
        body = self.body(env)
        robot, box = env.scene["robot"].data, env.scene["box"].data
        rotation = matrix_from_quat(robot.root_link_quat_w)[..., :2].flatten(-2)
        return torch.cat((robot.root_link_pos_w-env.scene.env_origins, rotation,
                          body[:, 7:], box.root_link_pos_w-env.scene.env_origins), -1)

    def __call__(self, env):
        frame = env.episode_length_buf.clamp(0, len(self.reference)-1)
        return self.reference[frame] - self.current(env)


def tracking_errors(residuals):
    """Physical error metrics independent of the moving discriminator reward."""
    residuals = residuals.reshape(-1, FEATURE_DIM)
    return {
        "root_position_error_m": residuals[:, :3].norm(dim=-1).mean().item(),
        "joint_position_rmse_rad": residuals[:, 15:44].square().mean().sqrt().item(),
        "hand_foot_position_rmse_m": residuals[:, 73:85].square().mean().sqrt().item(),
        "box_position_error_m": residuals[:, 106:109].norm(dim=-1).mean().item(),
    }
