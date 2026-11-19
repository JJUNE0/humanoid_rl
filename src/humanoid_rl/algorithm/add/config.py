"""ADD uses the shared imitation/data/control settings and a fixed reference clip."""

from dataclasses import dataclass
import math

from humanoid_rl.algorithm.amp.config import AmpConfig, config_from_values as shared_config


@dataclass
class AddConfig(AmpConfig):
    fixed_initial_state: bool = True
    observe_object: bool = True
    reference_init_probability: float = 0.0
    # MimicKit's ADD regularizes the output weights and both positive/negative logits.
    logit_regularization: float = 0.05

    def validate(self):
        super().validate()
        if not self.single_motion or not self.fixed_initial_state or not self.observe_object:
            raise ValueError("ADD currently requires single_motion, fixed_initial_state and observe_object")
        if not math.isfinite(self.logit_regularization) or self.logit_regularization < 0:
            raise ValueError("ADD logit_regularization must be nonnegative and finite")


def config_from_values(values, data_override=None):
    return shared_config(values, data_override, config_type=AddConfig)
