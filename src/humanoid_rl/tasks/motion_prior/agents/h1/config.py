"""Select H1 model joints and contact bodies for motion-prior pretraining."""

from pathlib import Path

from robots.humanoid.h1.config import JOINT_NAMES
from humanoid_rl.tasks.motion_prior.agents.config import AuxiliaryRewardConfig, MotionPriorAgentConfig


def make_config():
    return MotionPriorAgentConfig(
        robot="h1", task="H1-Motion-Prior", joint_names=JOINT_NAMES,
        feet_body_names=("left_ankle_link", "right_ankle_link"),
        undesired_contact_patterns=(".*pelvis.*", ".*hip.*", ".*shoulder.*"),
        ankle_roll_joint_names=(),
        rewards=AuxiliaryRewardConfig.load(Path(__file__).with_name("rewards.json")),
    )
