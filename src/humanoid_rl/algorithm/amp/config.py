"""Configuration for an AMP-only motion imitation experiment."""

from dataclasses import dataclass


@dataclass
class ControlConfig:
    # Active terms in I-BFM's saved carry checkpoint (not its optional task terms).
    action_rate: float = -0.1
    feet_orientation: float = -0.4
    ankle_roll: float = -4.0
    joint_position_limit: float = -10.0
    slippage: float = -4.0
    undesired_contact: float = -1.0
    action_rate_scale: float = 5.0
    actor_weight: float = 0.02
    ema_decay: float = 0.99


@dataclass
class AmpConfig:
    data_root: str
    hidden_dims: tuple[int, ...] = (512, 256)
    learning_rate: float = 1e-4
    batch_size: int = 512
    gradient_penalty: float = 10.0
    weight_decay: float = 1e-4
    reward_scale: float = 1.0
    min_feature_std: float = 0.1
    phase_balanced: bool = True
    reference_init_probability: float = 0.5
    validation_every: int = 100
    cache_dir: str = "outputs/amp_cache"
    max_motions: int | None = None
    # Explicit overfit experiment: one clip, no held-out expert validation.
    single_motion: str | None = None
    fixed_initial_state: bool = False
    # False preserves existing robot-only checkpoints, replay and expert caches.
    observe_object: bool = False
    # Defaults preserve old checkpoints and their environment semantics.
    terminate_on_fall: bool = True
    terminate_on_success: bool = False
    action_scale_multiplier: float = 1.0
    controller_profile: str = "native"
    control: ControlConfig | None = None

    def validate(self):
        import math

        if not self.data_root:
            raise ValueError("AMP needs data_root (--amp-data or --amp-config)")
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("AMP hidden_dims must contain positive layer widths")
        for name in ("learning_rate", "reward_scale", "min_feature_std"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"AMP {name} must be positive and finite")
        if self.batch_size < 1 or self.validation_every < 1:
            raise ValueError("AMP batch_size and validation_every must be positive")
        if not 0 <= self.reference_init_probability <= 1:
            raise ValueError("AMP reference_init_probability must be in [0, 1]")
        for name in ("gradient_penalty", "weight_decay"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"AMP {name} must be nonnegative and finite")
        if self.max_motions is not None and self.max_motions < 1:
            raise ValueError("AMP max_motions must be positive")
        if self.single_motion is not None:
            from pathlib import Path
            path = Path(self.single_motion)
            if path.is_absolute() or ".." in path.parts or path.name != "motion.npz":
                raise ValueError("single_motion must be a relative episode/motion.npz path")
            if self.max_motions is not None:
                raise ValueError("single_motion selects the clip; omit max_motions")
        if not isinstance(self.fixed_initial_state, bool):
            raise ValueError("fixed_initial_state must be a boolean")
        if self.fixed_initial_state:
            if not self.single_motion or not self.observe_object:
                raise ValueError("fixed_initial_state requires single_motion and observe_object")
            if self.reference_init_probability != 0:
                raise ValueError("fixed_initial_state requires reference_init_probability: 0")
        if not isinstance(self.observe_object, bool):
            raise ValueError("AMP observe_object must be a boolean")
        if not math.isfinite(self.action_scale_multiplier) or self.action_scale_multiplier <= 0:
            raise ValueError("action_scale_multiplier must be positive and finite")
        if self.controller_profile not in ("native", "reference"):
            raise ValueError("controller_profile must be native or reference")
        if self.controller_profile == "reference":
            if not self.fixed_initial_state or not self.single_motion:
                raise ValueError("reference controller requires fixed_initial_state and single_motion")
            if self.action_scale_multiplier != 1:
                raise ValueError("reference controller includes action normalization; action_scale_multiplier must be 1")
        if self.control is not None:
            for name in ("action_rate", "feet_orientation", "ankle_roll", "joint_position_limit", "slippage", "undesired_contact"):
                value = getattr(self.control, name)
                if not math.isfinite(value) or value > 0:
                    raise ValueError(f"Control penalty {name} must be finite and nonpositive")
            if not 0 <= self.control.ema_decay < 1:
                raise ValueError("Control ema_decay must be in [0, 1)")
            if not math.isfinite(self.control.actor_weight) or self.control.actor_weight < 0:
                raise ValueError("Control actor_weight must be finite and nonnegative")
            if not math.isfinite(self.control.action_rate_scale) or self.control.action_rate_scale <= 0:
                raise ValueError("Control action_rate_scale must be positive and finite")


def config_from_values(values, data_override=None, *, config_type=AmpConfig):
    values = dict(values)
    if data_override is not None:
        values["data_root"] = str(data_override)
    values.setdefault("data_root", "")
    if "hidden_dims" in values:
        values["hidden_dims"] = tuple(values["hidden_dims"])
    if isinstance(values.get("control"), dict):
        values["control"] = ControlConfig(**values["control"])
    result = config_type(**values)
    result.validate()
    return result
