"""Build the original FB-CPR-Aux configurations from configs/train.json."""

import math
from dataclasses import replace
from pathlib import Path

from humanoid_rl.algorithm.configuration import merge_config
from humanoid_rl.algorithm.bfm_zero.upstream import enable_bfm_zero
from humanoid_rl.tasks.motion_prior.agents import get_agent_config
from humanoid_retarget.datasets.paths import relocate_motion


def resolve_agent_config(values, task="H1-Motion-Prior", *, legacy=False):
    """Resolve robot defaults, experiment overrides and a saved immutable snapshot."""
    snapshot = values.get("motion_prior_agent")
    profile = get_agent_config(task, snapshot)
    parameters = values.get("auxiliary_reward_parameters", {})
    if legacy and snapshot is None:
        parameters = {"feet_orientation_contact": "magnitude"} | parameters
        profile = replace(profile, action_observation_units="normalized", history_uses_previous_action=False,
                          lying_reset_mode="neutral", lying_root_height=0.25)
    profile = profile.with_rewards(values.get("auxiliary_rewards"), parameters)
    values["auxiliary_rewards"] = dict(profile.rewards.weights)
    values["motion_prior_agent"] = profile.to_dict()
    return profile


def config_from_args(args):
    values = merge_config({}, args.train_settings)
    for name in values["training"]:
        if hasattr(args, name) and getattr(args, name) is not None:
            value = getattr(args, name)
            values["training"][name] = str(value) if isinstance(value, Path) else value
    for name in values["model"]:
        if hasattr(args, name) and getattr(args, name) is not None:
            values["model"][name] = getattr(args, name)
    for option, field in (("motion_data", "root"), ("motion_files", "files"),
                          ("motion_limit", "limit"), ("motion_max_frames", "max_frames")):
        if getattr(args, option, None) is not None:
            value = getattr(args, option)
            values["motion"][field] = str(value) if isinstance(value, Path) else value
    values["motion"] = relocate_motion(values["motion"], root=getattr(args, "motion_data", None),
                                      manifest=getattr(args, "motion_manifest", None),
                                      cache=getattr(args, "motion_cache", None))
    if args.episode_length is not None:
        values["environment"]["episode_length_s"] = args.episode_length
    if getattr(args, "history_length", None) is not None:
        values["environment"]["history_length"] = args.history_length
    if getattr(args, "gamma", None) is not None:
        values["agent"]["discount"] = args.gamma
    resolve_agent_config(values, args.task)
    validate_config(values)
    return values


def validate_config(values):
    training, model = values["training"], values["model"]
    for name in ("num_envs", "total_env_steps", "buffer_size", "min_buffer", "batch_size", "seed_steps", "save_every", "log_every", "updates_per_step"):
        value = training[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1 or value != int(value):
            raise ValueError(f"BFM training.{name} must be a positive integer")
        training[name] = int(value)
    if training["buffer_size"] < max(training["num_envs"], training["min_buffer"], training["batch_size"]):
        raise ValueError("Replay capacity must cover num_envs, min_buffer and batch_size")
    for name, value in model.items():
        if name == "actor_std":
            if not math.isfinite(value) or value <= 0:
                raise ValueError("actor_std must be positive")
        elif not isinstance(value, int) or value < 1:
            raise ValueError(f"BFM model.{name} must be a positive integer")
    if training["batch_size"] % model["sequence_length"]:
        raise ValueError("BFM batch_size must be divisible by sequence_length")
    if min(model["hidden_layers"], model["embedding_layers"]) < 2:
        raise ValueError("BFM residual networks require hidden_layers and embedding_layers >= 2")
    if not 0 <= values["agent"]["discount"] < 1:
        raise ValueError("BFM discount must be in [0,1)")
    if not isinstance(training["compile"],bool) or not isinstance(training["mixed_precision"],bool):
        raise ValueError("compile and mixed_precision must be JSON booleans")
    for key in ("limit", "max_frames"):
        if values["motion"].get(key) is not None and values["motion"][key] < (2 if key == "max_frames" else 1):
            raise ValueError(f"Invalid motion.{key}")
    if not math.isfinite(values["motion"]["clip_seconds"]) or values["motion"]["clip_seconds"] <= 0:
        raise ValueError("motion.clip_seconds must be positive")
    if training.get("eval_every", 0):
        raise ValueError("BFM tracking evaluation is not connected yet; use eval_every=0")
    keep = training.get("keep_checkpoints")
    if keep is not None and (not isinstance(keep,int) or keep < 1):
        raise ValueError("keep_checkpoints must be a positive integer")


def make_agent_config(values):
    values = merge_config({}, values)
    task = values.get("motion_prior_agent", {}).get("task", "H1-Motion-Prior")
    resolve_agent_config(values, task)
    enable_bfm_zero()
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig, FBcprAuxAgentTrainConfig
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
    from humanoidverse.agents.nn_filters import DictInputFilterConfig
    from humanoidverse.agents.nn_models import ActorArchiConfig, BackwardArchiConfig, DiscriminatorArchiConfig, ForwardArchiConfig, RewardNormalizerConfig
    from humanoidverse.agents.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig

    import torch
    train, model = values["training"], values["model"]
    actor_keys = ["state", "last_action", "history_actor"]
    critic_keys = ["state", "privileged_state", "last_action", "history_actor"]
    expert_keys = ["state", "privileged_state"]
    filt = lambda keys: DictInputFilterConfig(key=keys)
    def forward():
        return ForwardArchiConfig(hidden_dim=model["hidden_dim"], model="residual",
                                  hidden_layers=model["hidden_layers"], embedding_layers=model["embedding_layers"],
                                  num_parallel=2, ensemble_mode="batch", input_filter=filt(critic_keys))
    return FBcprAuxAgentConfig(
        model=FBcprAuxModelConfig(
            device=torch.device(train["device"]).type, amp=train["mixed_precision"],
            seq_length=model["sequence_length"], actor_std=model["actor_std"],
            archi=FBcprAuxModelArchiConfig(
                z_dim=model["z_dim"], norm_z=True, f=forward(), critic=forward(), aux_critic=forward(),
                b=BackwardArchiConfig(hidden_dim=model["backward_hidden_dim"], hidden_layers=1, norm=True, input_filter=filt(expert_keys)),
                actor=ActorArchiConfig(model="residual", hidden_dim=model["hidden_dim"],
                                      hidden_layers=model["hidden_layers"], embedding_layers=model["embedding_layers"], input_filter=filt(actor_keys)),
                discriminator=DiscriminatorArchiConfig(hidden_dim=model["discriminator_hidden_dim"],
                                                        hidden_layers=model["discriminator_hidden_layers"], input_filter=filt(expert_keys))),
            obs_normalizer=ObsNormalizerConfig(normalizers={
                key: BatchNormNormalizerConfig(momentum=0.01) for key in critic_keys}, allow_mismatching_keys=True),
            norm_aux_reward=RewardNormalizerConfig(translate=False, scale=True)),
        train=FBcprAuxAgentTrainConfig(**(values["agent"] | {"batch_size": train["batch_size"]})),
        aux_rewards=list(values["auxiliary_rewards"]),
        aux_rewards_scaling=values["auxiliary_rewards"], compile=train["compile"], cudagraphs=False,
    )
