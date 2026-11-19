"""Contact costs compatible with CMU/LAFAN hand-supported and floor motions."""

from pathlib import Path
from robots.humanoid.humanoid_light.config import JOINT_NAMES
from humanoid_rl.tasks.motion_prior.agents.config import AuxiliaryRewardConfig, MotionPriorAgentConfig


def make_config():
    return MotionPriorAgentConfig(
        robot="humanoid_light", task="Humanoid-Light-Motion-Prior", joint_names=JOINT_NAMES,
        root_body_name="pelvis_link", feet_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        undesired_contact_patterns=("head_link",),
        ankle_roll_joint_names=("left_ankle_roll_joint", "right_ankle_roll_joint"),
        rewards=AuxiliaryRewardConfig.load(Path(__file__).with_name("rewards.json")),
        lying_root_height=.4,
    )
