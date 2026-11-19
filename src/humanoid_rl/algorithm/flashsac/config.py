"""Reuse upstream hyperparameters with the Wolf FlashSAC training defaults."""

import math

from omegaconf import OmegaConf

from humanoid_rl.algorithm.flashsac.upstream import FLASH_ROOT, enable_flashsac


def make_agent_config(args):
    enable_flashsac()
    from flash_rl.agents.flashSAC.agent import FlashSACConfig

    # Only trusted, checked-in upstream config expressions are evaluated.
    OmegaConf.register_new_resolver(
        "eval",
        lambda value: eval(value, {"__builtins__": {}}, {"int": int}),
        replace=True,
    )
    cfg = OmegaConf.create(
        {
            "seed": args.seed,
            "gamma": 0.99 if getattr(args, "gamma", None) is None else args.gamma,
            "n_step": args.n_step,
            "num_interaction_steps": math.ceil(args.total_env_steps / args.num_envs),
            "updates_per_interaction_step": args.updates_per_step,
            "agent": OmegaConf.load(FLASH_ROOT / "configs" / "agent" / "flashSAC.yaml"),
        }
    )
    cfg.agent = OmegaConf.merge(cfg.agent, getattr(args, "train_settings", {}).get("agent", {}))
    cfg.agent.device_type = args.device
    cfg.agent.buffer_device_type = args.buffer_device
    cfg.agent.buffer_max_length = args.buffer_size
    cfg.agent.buffer_min_length = args.min_buffer
    cfg.agent.sample_batch_size = args.batch_size
    cfg.agent.use_compile = args.compile
    cfg.agent.use_amp = args.mixed_precision and args.device.startswith("cuda")
    cfg.agent.asymmetric_observation = False
    values = OmegaConf.to_container(cfg.agent, resolve=True)
    values.pop("agent_type")
    return FlashSACConfig(**values)
