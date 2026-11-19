"""Public method adapters. Learning implementations keep their existing imports.

This registry is deliberately independent of torch and simulators: selecting a
method must not import another method's incompatible runtime dependencies.
"""

from dataclasses import dataclass, field, replace
import importlib


@dataclass(frozen=True)
class Method:
    name: str
    environment: str
    tasks: dict
    description: str
    task_environments: dict = field(default_factory=dict)

    def adapter(self):
        return importlib.import_module(f"humanoid_rl.methods.{self.name}")


METHODS = {
    "bfm_zero": Method("bfm_zero", "mjlab", {
        "Humanoid-Light-Motion-Prior": "humanoid_light",
        "G1-Motion-Prior": "g1", "H1-Motion-Prior": "h1",
    }, "Original FB-CPR-Aux learner with local mjlab tasks"),
    "flashsac": Method("flashsac", "mjlab", {
        "G1-Carry-Box": "g1", "Mjlab-Cartpole-Balance": "cartpole",
    }, "FlashSAC; task, AMP and ADD rewards"),
    "supermimic": Method("supermimic", "supermimic", {
        "G1-Carry-Box": "g1", "G1-Stand": "g1", "G1-Walk": "g1",
        "Humanoid-Light-Motion-Tracking": "humanoid_light",
    }, "Light tracking with the official learner; preserved G1 carry/stand/walk implementation",
       {"Humanoid-Light-Motion-Tracking": "supermimic_light"}),
}


def get_method(name, task=None, robot=None):
    if name not in METHODS:
        raise ValueError(f"Unknown method {name!r}; choose from {', '.join(METHODS)}")
    method = METHODS[name]
    if task is not None and task not in method.tasks:
        raise ValueError(f"{name} does not support task {task!r}. Supported: {', '.join(method.tasks)}.")
    if robot is not None and method.tasks.get(task) != robot:
        raise ValueError(f"Robot {robot!r} does not match task {task!r}")
    return replace(method, environment=method.task_environments[task]) if task in method.task_environments else method


def validate_native_training(method, task, values):
    """Check CLI names without importing the selected learning implementation."""
    from humanoid_rl.train import parser
    allowed = {action.dest for action in parser(["--algorithm", method, "--task", task])._actions}
    unknown = set(values.get("training", {})) - allowed
    if unknown:
        raise ValueError(f"Unknown {method} training settings: {sorted(unknown)}")
