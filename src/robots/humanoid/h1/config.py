"""Load the copied H1 model without depending on LocoMuJoCo or its asset package.

The MJCF includes its own floor and lights. Actuators are the source model's
torque motors; an RL task must explicitly configure its action/PD controller.
"""

from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent
MJCF_PATH = ASSET_DIR / "h1.xml"
ROOT_BODY_NAME = "pelvis"
ROOT_JOINT_NAME = "root"
# MuJoCo qpos/qvel order, excluding the floating root. LocoMuJoCo's generic
# environment action specification lists these in a different order.
JOINT_NAMES = (
    "hip_rotation_l", "hip_adduction_l", "hip_flexion_l", "knee_angle_l", "ankle_angle_l",
    "hip_rotation_r", "hip_adduction_r", "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
    "back_bkz",
    "l_arm_shy", "l_arm_shx", "l_arm_shz", "left_elbow",
    "r_arm_shy", "r_arm_shx", "r_arm_shz", "right_elbow",
)
ACTUATOR_NAMES = tuple(f"{name}_actuator" for name in JOINT_NAMES)
KEY_SITE_NAMES = (
    "left_hand_mimic", "right_hand_mimic", "left_foot_mimic", "right_foot_mimic",
)


def get_spec():
    """Return a fresh MuJoCo specification using only this directory's files."""
    import mujoco

    return mujoco.MjSpec.from_file(str(MJCF_PATH))
