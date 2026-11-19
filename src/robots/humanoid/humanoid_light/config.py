"""The converted USD model, with its original joint axes, masses and limits."""

from pathlib import Path

ASSET_DIR = Path(__file__).with_name("assets")
MJCF_PATH = ASSET_DIR / "humanoid_light.xml"
ROOT_BODY_NAME = "pelvis_link"


def get_spec():
    import mujoco
    return mujoco.MjSpec.from_file(str(MJCF_PATH))


# Keep the MJCF depth-first order, including the torso/head before the arms.
JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_yaw_joint", "torso_pitch_joint", "torso_roll_joint", "head_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_joint",
)
