"""Run the untouched upstream G1 MuJoCo environment and record its conventions.

Run with an interpreter containing BFM-Zero's MuJoCo dependencies. This does not
train a policy, launch Isaac Sim, or modify the upstream checkout.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BFM_ROOT = ROOT/"third_party/BFM-Zero"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/bfm_zero_original_audit/report.json")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2 to check history alignment")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(BFM_ROOT))
    import numpy as np
    import mujoco
    import torch
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig

    torch.manual_seed(4728)
    np.random.seed(4728)
    torch.set_num_threads(2)
    config = HumanoidVerseIsaacConfig(
        device="cpu", lafan_tail_path=str(BFM_ROOT/"humanoidverse/data/lafan_29dof_10s-clipped.pkl"),
        disable_obs_noise=True, disable_domain_randomization=True,
        include_last_action=True, include_history_actor=True, root_height_obs=True,
        camera_render_save_dir=str(args.output.parent/"rendering"),
        hydra_overrides=["simulator=mujoco", "robot=g1/g1_29dof_hard_waist",
                         "robot.control.action_scale=0.25", "robot.control.action_clip_value=5.0",
                         "robot.control.normalize_action_to=5.0", "env.config.lie_down_init=True",
                         "env.config.lie_down_init_prob=0.3"],
    )
    env, _ = config.build(num_envs=1)
    observation, _ = env.reset(reset_to_default_pose=True, to_numpy=False)
    raw_costs, observations = [], []
    for step in range(args.steps):
        action = torch.full((1, 29), .01*(step+1))
        previous = observation["last_action"].clone()
        observation, reward, terminated, truncated, info = env.step(action, to_numpy=False)
        for value in observation.values():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all()
        torch.testing.assert_close(observation["last_action"], action*5)
        torch.testing.assert_close(observation["history_actor"][:, :29], previous)
        raw_costs.append({key: float(value.mean()) for key, value in info["aux_rewards"].items()})
        observations.append({"step": step+1, "action": float(action[0, 0]),
                             "last_action": float(observation["last_action"][0, 0]),
                             "history_action_latest": float(observation["history_actor"][0, 0]),
                             "terminated": bool(terminated.any()), "truncated": bool(truncated.any())})
        assert all(np.isfinite(value) for value in raw_costs[-1].values())
    train_tree = ast.parse((BFM_ROOT/"humanoidverse/train.py").read_text())
    weights = next(ast.literal_eval(node.value) for node in ast.walk(train_tree)
                   if isinstance(node, ast.keyword) and node.arg == "aux_rewards_scaling")
    sources = ("humanoidverse/envs/legged_base_task/legged_robot_base.py",
               "humanoidverse/envs/legged_robot_motions/legged_robot_motions.py",
               "humanoidverse/simulator/mujoco/mujoco.py", "humanoidverse/train.py")
    report = {
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BFM_ROOT, text=True).strip(),
        "source_sha256": {name: hashlib.sha256((BFM_ROOT/name).read_bytes()).hexdigest() for name in sources},
        "backend": "original BFM-Zero MuJoCo CPU", "policy": "small deterministic test actions; no training",
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "mujoco": mujoco.__version__},
        "num_envs": 1, "steps": args.steps, "observation_noise": False, "domain_randomization": False,
        "observation_dims": {key: list(value.shape) for key, value in observation.items() if torch.is_tensor(value)},
        "control_dt": env.base_env.dt, "auxiliary_weights": weights,
        "control": {key: env.base_env.config.robot.control[key]
                    for key in ("action_scale", "normalize_action_to", "action_rescale", "clip_torques")},
        "action_observation_scale_check": "passed: last_action = normalized action * 5",
        "history_alignment_check": "passed: latest history action = preceding observation last_action",
        "action_trace": observations, "raw_auxiliary_rewards": raw_costs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"Original G1 environment: {args.steps} steps passed; report: {args.output}")


if __name__ == "__main__":
    main()
