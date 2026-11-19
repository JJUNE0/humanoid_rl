"""Optional W&B tracking, shared by live training and TensorBoard backfills."""

import hashlib


def experiment_identity(metadata, directory):
    mode = metadata.get("reward_mode", "amp")
    is_add = mode == "add"
    amp = metadata.get("add" if is_add else "amp") or {}
    config = amp.get("config", {})
    object_amp = config.get("observe_object", False)
    frozen = metadata.get("stage_model") is not None
    control = config.get("control") is not None
    label = ("object_add" if object_amp else "body_add") if is_add else ("object_amp" if object_amp else "body_amp")
    label += "_frozen_mlp" if frozen else "_no_mlp"
    if control:
        label += "_control"
    fixed = config.get("fixed_initial_state", False)
    reference_control = config.get("controller_profile", "native") == "reference"
    single = config.get("single_motion") is not None
    if single:
        label += "_single_motion"
    if fixed:
        label += "_fixed_scene"
    if reference_control:
        label += "_reference_control"
    task = metadata["task"]
    identity = hashlib.sha256(f"{task}/{directory.name}".encode()).hexdigest()[:12]
    return {
        "id": identity,
        "name": f"{label}_{metadata['observation_dim']}d_{directory.name}",
        "tags": ["mjlab", "flashsac", "g1", "carry-box",
                 ("object-add" if object_amp else "body-add") if is_add else ("object-amp" if object_amp else "body-amp"),
                 "frozen-mlp" if frozen else "no-mlp", "control-critic" if control else "no-control-critic"]
                + (["reference-tracking", "phase-conditioned"] if is_add else [])
                + (["single-motion", "no-heldout-expert"] if single else [])
                + (["fixed-scene", "frame-0-reset", "no-rsi"] if fixed else [])
                + (["reference-controller", "explicit-pd"] if reference_control else []),
        "config": metadata | {"uses_frozen_mlp": frozen, "uses_object_amp": object_amp and not is_add,
                              "uses_object_add": object_amp and is_add,
                              "uses_control_critic": control, "amp_pair_dim": None if is_add else 2 * amp.get("feature_dim", 83),
                              "add_pair_dim": 2 * amp.get("feature_dim", 109) if is_add else None,
                              "uses_fixed_scene": fixed, "uses_single_motion": single,
                              "uses_reference_controller": reference_control,
                              "source_run_directory": str(directory)},
    }


class WandbLogger:
    def __init__(self, metadata, directory, *, project, entity=None, name=None,
                 mode="online", collect_system_stats=True, job_type="training", storage=None):
        import wandb

        identity = experiment_identity(metadata, directory)
        self.num_envs = metadata["num_envs"]
        self.run = wandb.init(
            project=project, entity=entity, id=identity["id"], name=name or identity["name"],
            group="g1-carry-add" if metadata.get("reward_mode") == "add" else "g1-carry-amp",
            tags=identity["tags"], config=identity["config"],
            resume="allow" if mode == "online" else None, mode=mode, job_type=job_type,
            dir=str(storage or directory), save_code=False,
            settings=wandb.Settings(disable_git=True, save_code=False, console="off",
                                    x_disable_stats=not collect_system_stats),
        )
        self.run.define_metric("vector_steps")
        self.run.define_metric("env_steps")
        self.run.define_metric("*", step_metric="vector_steps")

    def log(self, metrics, env_steps):
        # Keep W&B's internal history counter separate: evaluation can arrive
        # after training metrics for the same source step, including backfills.
        self.run.log(dict(metrics) | {"env_steps": int(env_steps),
                                      "vector_steps": env_steps / self.num_envs})

    def metadata(self):
        return {"entity": self.run.entity, "project": self.run.project,
                "id": self.run.id, "url": self.run.url, "mode": self.run.settings.mode}

    def finish(self, exit_code=0):
        self.run.finish(exit_code=exit_code)
