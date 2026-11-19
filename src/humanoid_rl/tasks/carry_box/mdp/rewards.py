"""Placeholder sparse reward; no dense shaping or AMP yet."""

from humanoid_rl.tasks.carry_box.mdp.terminations import placed


def placement_bonus(env):
    # Success also terminates the episode, so this bonus is emitted once.
    return placed(env).float()
