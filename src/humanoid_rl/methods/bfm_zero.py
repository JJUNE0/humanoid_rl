"""Translate a common launch into the existing BFM-Zero CLI."""

from humanoid_rl.algorithm.configuration import load_train_config, merge_config
from humanoid_rl.methods import validate_native_training


def settings(task, overrides):
    values = merge_config(load_train_config("bfm_zero", task=task), overrides)
    validate_native_training("bfm_zero", task, values)
    return values


def inputs(settings):
    motion = settings["motion"]
    if motion.get("manifest"):
        return [{"path": motion["manifest"], "kind": "motion_manifest", "root": motion["manifest_root"]}]
    return [{"path": motion["root"], "kind": "directory"}]


RESUME_TRAINING = ("num_envs", "total_env_steps", "device", "buffer_device", "save_every", "log_every", "log_dir", "save_replay")
RESUME_MOTION = ("root", "manifest_root", "manifest", "cache_dir")


def resume_settings(values, path, task, overrides):
    import json
    check_resume(path, task)
    for expression in overrides:
        key = expression.partition("=")[0]
        allowed = {"training." + k for k in RESUME_TRAINING} | {"motion." + k for k in RESUME_MOTION}
        if key not in allowed:
            raise ValueError(f"BFM resume restores learning settings; cannot override {key}")
    restored = json.loads((path / "run.json").read_text())["train_config"]
    for key in RESUME_TRAINING:
        if key in values["training"]:
            restored["training"][key] = values["training"][key]
    for key in RESUME_MOTION:
        if key in values["motion"]:
            restored["motion"][key] = values["motion"][key]
    return restored


def command(plan):
    args = [plan.python, "-m", "humanoid_rl.train", "--algorithm", "bfm_zero", "--task", plan.task]
    if plan.resume:
        # BFM restores its learning settings from the checkpoint. Only explicit
        # runtime/path overrides may accompany a resume, never --train-config.
        args += ["--resume", str(plan.resume)]
        # The entry point must disable upstream function-level compilation before
        # importing the learner when resuming an uncompiled checkpoint.
        args.append("--compile" if plan.settings["training"]["compile"] else "--no-compile")
        for key in ("num_envs", "total_env_steps", "device", "buffer_device", "save_every", "log_every", "log_dir"):
            args += ["--" + key.replace("_", "-"), str(plan.settings["training"][key])]
        motion = plan.settings["motion"]
        args += ["--motion-data", str(motion.get("manifest_root", motion["root"])),
                 "--motion-cache", str(motion["cache_dir"])]
        if motion.get("manifest"):
            args += ["--motion-manifest", str(motion["manifest"])]
        args.append("--save-replay" if plan.settings["training"].get("save_replay") else "--no-save-replay")
    else:
        args += ["--train-config", str(plan.run_dir / "method.json")]
    return args


def check_resume(path, task):
    import json
    metadata = json.loads((path / "run.json").read_text())
    if metadata.get("algorithm") != "bfm_zero" or metadata.get("task") != task:
        raise ValueError("Resume checkpoint must match the BFM-Zero method and task")


def prepare(plan):
    pass
