"""Adapter for the preserved G1 trainer, in its own Python 3.10 environment."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

TASKS = {"G1-Carry-Box": "carry", "G1-Stand": "stand", "G1-Walk": "walk"}


def settings(task, overrides):
    if task == "Humanoid-Light-Motion-Tracking":
        from humanoid_rl.algorithm.configuration import merge_config
        default = Path(__file__).parents[1] / "algorithm/supermimic/configs/light.json"
        return merge_config(json.loads(default.read_text()), overrides)
    if set(overrides) - {"training"}:
        raise ValueError("Existing SuperMimic settings belong under config.training")
    training = deepcopy(overrides.get("training", {}))
    if set(training) & {"task", "output", "resume"}:
        raise ValueError("SuperMimic task/output/resume are controlled by the common launcher")
    return {"training": training}


def command(plan):
    if plan.task == "Humanoid-Light-Motion-Tracking":
        args = [plan.python, "-m", "humanoid_rl.algorithm.supermimic.train",
                "--config", str(plan.run_dir / "method.json"), "--output", str(plan.run_dir / "training")]
        return args + (["--resume", str(plan.resume)] if plan.resume else [])
    args = [plan.python, "-m", "super_mimic", "train", "--task", TASKS[plan.task],
            "--output", str(plan.run_dir / "training")]
    for name, value in plan.settings["training"].items():
        if value is None or value is False:
            continue
        args.append("--" + name)
        if value is not True:
            args.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    if plan.resume:
        args.append("--resume")
    return args


def inputs(settings):
    if "motion" in settings:
        motion = settings["motion"]
        result = [{"path": motion["manifest"], "root": motion["root"], "kind": "motion_manifest"}]
        init_from = settings.get("training", {}).get("init_from")
        if init_from:
            result.append({"path": str(init_from), "kind": "file" if str(init_from).endswith(".pt") else "directory"})
        return result
    training = settings["training"]
    result = []
    for key in ("base_cache", "sidecar", "reward_cache", "world_model_checkpoint",
                "initialize_from", "pretrained_policy", "pretrained_critic"):
        path = training.get(key)
        if path and str(path).lower() != "none":
            result.append({"path": str(path), "kind": "directory" if key.endswith("cache") or key == "sidecar" else "file"})
    return result
def checkpoint_file(path):
    return path / "last.pt" if path.is_dir() else path


def check_resume(path, task):
    if task == "Humanoid-Light-Motion-Tracking":
        metadata = json.loads((path / "run.json").read_text())
        if metadata.get("task") != task or not (path / "last.pt").is_file():
            raise ValueError("Light SuperMimic resume requires a matching checkpoint directory")
        return
    if not checkpoint_file(path).is_file():
        raise ValueError("SuperMimic resume requires last.pt or its containing directory")


def prepare(plan):
    if plan.resume and plan.task != "Humanoid-Light-Motion-Tracking":
        destination = plan.run_dir / "training" / "last.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_file(plan.resume), destination)
