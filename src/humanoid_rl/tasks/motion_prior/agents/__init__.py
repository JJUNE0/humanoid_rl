"""Select robot-specific motion-prior settings without duplicating robot assets."""

from humanoid_rl.tasks.motion_prior.agents.config import MotionPriorAgentConfig


def get_agent_config(task="H1-Motion-Prior", snapshot=None):
    if snapshot is not None:
        config = MotionPriorAgentConfig.from_dict(snapshot)
    elif task == "H1-Motion-Prior":
        from humanoid_rl.tasks.motion_prior.agents.h1.config import make_config
        config = make_config()
    elif task == "G1-Motion-Prior":
        from humanoid_rl.tasks.motion_prior.agents.g1.config import make_config
        config = make_config()
    elif task == "Humanoid-Light-Motion-Prior":
        from humanoid_rl.tasks.motion_prior.agents.humanoid_light.config import make_config
        config = make_config()
    else:
        raise ValueError(f"No motion-prior agent settings for {task!r}")
    expected_robot = {"H1-Motion-Prior": "h1", "G1-Motion-Prior": "g1",
                      "Humanoid-Light-Motion-Prior": "humanoid_light"}.get(task)
    if config.task != task or config.robot != expected_robot:
        raise ValueError("Saved motion-prior robot/task configuration differs from the selected task")
    return config
