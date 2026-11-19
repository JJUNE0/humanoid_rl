"""Robot selections and physical-cost settings shared by motion-prior tasks."""

from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path


SUPPORTED_REWARDS = frozenset((
    "penalty_action_rate", "penalty_feet_ori", "limits_dof_pos", "penalty_slippage",
    "penalty_undesired_contact", "penalty_ankle_roll",
))


@dataclass(frozen=True)
class AuxiliaryRewardConfig:
    weights: dict[str, float]
    contact_force_threshold: float = 1.0
    soft_joint_position_limit: float = 0.95
    feet_orientation_contact: str = "vertical"
    # mjlab netforce points primary -> secondary; negate to get force on the robot.
    sensor_force_to_robot: float = -1.0

    def __post_init__(self):
        unknown = set(self.weights)-SUPPORTED_REWARDS
        if unknown:
            raise ValueError(f"Unsupported auxiliary rewards: {sorted(unknown)}")
        if any(not math.isfinite(value) or value > 0 for value in self.weights.values()):
            raise ValueError("Auxiliary penalty weights must be finite and nonpositive")
        if not math.isfinite(self.contact_force_threshold) or self.contact_force_threshold < 0:
            raise ValueError("contact_force_threshold must be finite and nonnegative")
        if not 0 < self.soft_joint_position_limit <= 1:
            raise ValueError("soft_joint_position_limit must be in (0,1]")
        if self.feet_orientation_contact not in ("vertical", "magnitude"):
            raise ValueError("feet_orientation_contact must be vertical or magnitude")
        if self.sensor_force_to_robot not in (-1.0, 1.0):
            raise ValueError("sensor_force_to_robot must be -1 or 1")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def with_overrides(self, weights=None, parameters=None):
        parameters = parameters or {}
        if "weights" in parameters:
            raise ValueError("Set weights through auxiliary_rewards")
        return replace(self, weights=self.weights | (weights or {}), **parameters)


@dataclass(frozen=True)
class MotionPriorAgentConfig:
    robot: str
    task: str
    joint_names: tuple[str, ...]
    feet_body_names: tuple[str, str]
    undesired_contact_patterns: tuple[str, ...]
    ankle_roll_joint_names: tuple[str, ...]
    rewards: AuxiliaryRewardConfig
    root_body_name: str = "pelvis"
    action_observation_units: str = "controller"
    history_uses_previous_action: bool = True
    lying_reset_mode: str = "reference"
    lying_root_height: float = 0.5
    virtual_bodies: tuple[dict, ...] = ()
    noise_scales: dict[str, float] = field(default_factory=dict)
    randomization: dict = field(default_factory=dict)
    strict_timeout: bool = False

    def __post_init__(self):
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must contain unique joints in model order")
        if len(self.feet_body_names) != 2 or len(set(self.feet_body_names)) != 2:
            raise ValueError("Exactly two distinct foot bodies are required")
        if not set(self.ankle_roll_joint_names) <= set(self.joint_names):
            raise ValueError("Ankle-roll joints must belong to joint_names")
        if "penalty_ankle_roll" in self.rewards.weights and not self.ankle_roll_joint_names:
            raise ValueError(f"{self.robot} has no ankle-roll joints for penalty_ankle_roll")
        if self.action_observation_units not in ("controller", "normalized"):
            raise ValueError("action_observation_units must be controller or normalized")
        if not isinstance(self.history_uses_previous_action, bool):
            raise ValueError("history_uses_previous_action must be a boolean")
        if self.lying_reset_mode not in ("reference", "neutral"):
            raise ValueError("lying_reset_mode must be reference or neutral")
        if not math.isfinite(self.lying_root_height) or self.lying_root_height <= 0:
            raise ValueError("lying_root_height must be finite and positive")

    def with_rewards(self, weights=None, parameters=None):
        return replace(self, rewards=self.rewards.with_overrides(weights, parameters))

    def to_dict(self):
        # JSON round-trip makes the in-memory and saved representation identical.
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("joint_names", "feet_body_names", "undesired_contact_patterns", "ankle_roll_joint_names"):
            value[key] = tuple(value[key])
        value["virtual_bodies"] = tuple(value.get("virtual_bodies", ()))
        value["rewards"] = AuxiliaryRewardConfig(**value["rewards"])
        return cls(**value)
