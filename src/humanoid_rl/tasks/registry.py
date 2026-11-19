"""Environment factories, independent of mjlab's default PPO runner config."""

MOTION_PRIOR_TASKS = ("H1-Motion-Prior", "G1-Motion-Prior", "Humanoid-Light-Motion-Prior")
TASKS = ("G1-Carry-Box", "Mjlab-Cartpole-Balance") + MOTION_PRIOR_TASKS


def make_task_cfg(task, carry=None, *, observe_hand_feet=True, stage_model=None):
    if task in MOTION_PRIOR_TASKS:
        from humanoid_rl.tasks.motion_prior.env_cfg import make_env_cfg
        from humanoid_rl.tasks.motion_prior.agents import get_agent_config

        return make_env_cfg(agent_cfg=get_agent_config(task))
    if task == "G1-Carry-Box":
        from humanoid_rl.tasks.carry_box.env_cfg import make_env_cfg

        return make_env_cfg(carry, observe_hand_feet=observe_hand_feet, stage_model=stage_model)
    if task == "Mjlab-Cartpole-Balance":
        from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg

        cfg = cartpole_balance_env_cfg()
        cfg.auto_reset = False
        cfg.scale_rewards_by_dt = False
        return cfg
    raise ValueError(f"Unknown task {task!r}; choose from {TASKS}")
