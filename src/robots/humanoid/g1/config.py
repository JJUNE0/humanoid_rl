"""Native mjlab G1 definitions shared by tasks and expert feature extraction.

Keep mjlab as the source of the model and actuator defaults. A recorded
controller profile is an explicit override in control.py.
"""

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

KEY_BODY_NAMES = (
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
)

# Native MJCF qpos/qvel order, excluding the floating root.
JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

__all__ = ["G1_ACTION_SCALE", "JOINT_NAMES", "KEY_BODY_NAMES", "get_g1_robot_cfg", "get_spec"]
