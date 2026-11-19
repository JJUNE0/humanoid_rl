"""Translate a common launch into FlashSAC's task/AMP/ADD training CLI."""

from humanoid_rl.algorithm.configuration import load_train_config, merge_config
from humanoid_rl.methods import validate_native_training


def settings(task, overrides):
    mode = overrides.get("training", {}).get("reward_mode")
    values = load_train_config("flashsac")
    if mode in ("amp", "add"):
        values = merge_config(values, load_train_config(mode))
    values = merge_config(values, overrides)
    validate_native_training("flashsac", task, values)
    return values


def inputs(settings):
    return []


def command(plan):
    args = [plan.python, "-m", "humanoid_rl.train", "--algorithm", "flashsac", "--task", plan.task,
            "--train-config", str(plan.run_dir / "method.json")]
    if plan.resume:
        args += ["--resume", str(plan.resume)]
    return args


def check_resume(path, task):
    import json
    metadata = json.loads((path / "run.json").read_text())
    if metadata.get("algorithm", "flashsac") != "flashsac" or metadata.get("task") != task:
        raise ValueError("Resume checkpoint must match the FlashSAC method and task")


def prepare(plan):
    pass
