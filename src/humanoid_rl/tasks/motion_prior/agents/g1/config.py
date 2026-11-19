"""Original BFM G1 observations, contacts and randomization selections."""

from pathlib import Path

from robots.humanoid.g1.config import JOINT_NAMES
from humanoid_rl.tasks.motion_prior.agents.config import AuxiliaryRewardConfig, MotionPriorAgentConfig


def make_config():
    return MotionPriorAgentConfig(
        robot="g1", task="G1-Motion-Prior", joint_names=JOINT_NAMES,
        feet_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        undesired_contact_patterns=(".*pelvis.*", ".*hip.*", ".*shoulder.*"),
        ankle_roll_joint_names=("left_ankle_roll_joint", "right_ankle_roll_joint"),
        virtual_bodies=({"name":"head_link","parent":"torso_link","position":[0.,0.,.35]},),
        noise_scales={"joint_position":.01,"joint_velocity":.5,"gravity":.05,"angular_velocity":.05},
        randomization={"mass_scale":[.95,1.05],"com_offset":[-.02,.02],"friction":[.5,1.25],
                       "joint_offset":[-.02,.02],"push_interval_s":[1.,3.],"push_velocity":.5},
        strict_timeout=True,
        rewards=AuxiliaryRewardConfig.load(Path(__file__).with_name("rewards.json")),
    )
