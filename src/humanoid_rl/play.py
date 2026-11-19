"""Interactive scene inspection using mjlab's existing native and Viser viewers."""

import argparse
import json
import math
import os
from pathlib import Path

from humanoid_rl.tasks.registry import TASKS


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--task", choices=TASKS)
    cli.add_argument("--list-tasks", action="store_true")
    cli.add_argument("--agent", choices=("zero", "random", "trained"))
    cli.add_argument(
        "--checkpoint",
        type=Path,
        help="FlashSAC checkpoint directory containing actor.pt and run.json",
    )
    cli.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
    cli.add_argument(
        "--headless", action="store_true", help="Run without a viewer; requires --steps"
    )
    cli.add_argument(
        "--paused",
        action="store_true",
        help="Open viewer paused, for initial scene inspection",
    )
    cli.add_argument("--num-envs", type=int, default=1)
    cli.add_argument(
        "--env-index",
        type=int,
        default=0,
        help="Environment shown in the viewer/screenshot",
    )
    cli.add_argument(
        "--steps", type=int, help="Stop after this many vector control steps"
    )
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--device", default="cuda:0")
    cli.add_argument(
        "--scene-config",
        type=Path,
        help="Override carry scene sampling ranges using YAML",
    )
    cli.add_argument(
        "--episode-length", type=float, help="Override episode duration in seconds"
    )
    cli.add_argument(
        "--screenshot", type=Path, help="Save the initial randomized scene to PNG"
    )
    return cli


def configure_scene(args, metadata=None):
    """Use the same task factory and sampling configuration as training."""
    import yaml
    from mjlab.viewer import ViewerConfig

    from humanoid_rl.tasks.carry_box.env_cfg import CarrySceneCfg
    from humanoid_rl.tasks.registry import make_task_cfg

    values = metadata.get("carry") if metadata is not None else None
    if args.scene_config:
        values = yaml.safe_load(args.scene_config.read_text())
        if not isinstance(values, dict):
            raise ValueError("--scene-config must contain a YAML mapping")
    carry = None
    if values is not None:
        carry = CarrySceneCfg(
            **{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
        )
    # Older checkpoints predate the extra XYZs and must remain 108-dimensional.
    observe_hand_feet = metadata.get("observe_hand_feet", False) if metadata is not None else True
    stage_model = None
    if metadata is not None and metadata.get("stage_model") is not None:
        from humanoid_rl.tasks.carry_box.stage.config import StageModelConfig
        stage_model = StageModelConfig.from_metadata(metadata["stage_model"], args.checkpoint)
    if metadata is not None and metadata.get("algorithm") == "bfm_zero":
        from humanoid_rl.algorithm.configuration import merge_config
        from humanoid_rl.algorithm.bfm_zero.config import resolve_agent_config
        from humanoid_rl.tasks.motion_prior.env_cfg import MotionPriorEnvConfig, make_env_cfg
        values = merge_config({}, metadata["train_config"])
        agent_cfg = resolve_agent_config(values, args.task, legacy=True)
        cfg = make_env_cfg(MotionPriorEnvConfig(**values["environment"]), metadata["robot_control"], agent_cfg=agent_cfg)
    else:
        cfg = make_task_cfg(args.task, carry, observe_hand_feet=observe_hand_feet, stage_model=stage_model)
    if metadata is not None and metadata.get("reward_mode") in ("amp", "add"):
        from humanoid_rl.env.imitation.amp import configure_amp_environment
        if metadata["reward_mode"] == "add":
            from humanoid_rl.algorithm.add.config import config_from_values
        else:
            from humanoid_rl.algorithm.amp.config import config_from_values

        amp_config = config_from_values(metadata[metadata["reward_mode"]]["config"])
        fixed_scene = None
        controller = None
        if amp_config.fixed_initial_state:
            from humanoid_rl.tasks.carry_box.fixed_scene import load_scene_bundle
            fixed_scene = load_scene_bundle(args.checkpoint)
        if amp_config.controller_profile == "reference":
            from robots.humanoid.g1.control import load_controller_bundle
            controller = load_controller_bundle(args.checkpoint)
            if controller["fingerprint"] != metadata["reference_controller"]["fingerprint"]:
                raise ValueError("Checkpoint reference controller and metadata differ")
        # Normal checkpoints use ordinary spawns. The overfit experiment uses
        # its portable, exact first frame; neither playback uses random-time RSI.
        if metadata["reward_mode"] == "add":
            from humanoid_rl.env.imitation.add import configure_add_environment
            configure_add_environment(cfg, amp_config, fixed_scene=fixed_scene, controller=controller)
        else:
            configure_amp_environment(cfg, amp_config=amp_config, fixed_scene=fixed_scene, controller=controller)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.auto_reset = False
    if args.episode_length is not None:
        cfg.episode_length_s = args.episode_length
    elif metadata is not None:
        cfg.episode_length_s = metadata["episode_length_s"]
    cfg.viewer.env_idx = args.env_index
    cfg.viewer.width = 1280
    cfg.viewer.height = 720
    if args.task == "G1-Carry-Box":
        cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
        cfg.viewer.entity_name = "robot"
        cfg.viewer.distance = 5.5
        cfg.viewer.elevation = -20
        cfg.viewer.azimuth = 135
    return cfg


def make_playback_env(env):
    """Extend the training adapter with mjlab's viewer protocol."""
    if hasattr(env.cfg, "motion_prior"):
        from humanoid_rl.algorithm.bfm_zero.play import make_playback_env as make_bfm_playback
        return make_bfm_playback(env)
    from humanoid_rl.env.mjlab_adapter import MjlabAdapter

    class PlaybackEnv(MjlabAdapter):
        @property
        def unwrapped(self):
            return self.env

        @property
        def cfg(self):
            return self.env.cfg

        @property
        def device(self):
            return self.env.device

        def get_observations(self):
            # Viser can reset selected envs directly through `unwrapped`.
            self.observations = self.env.observation_manager.compute(
                update_history=False
            )["actor"].clone()
            return self.observations

        def reset(self, *, env_ids=None):
            observations, extras = self.env.reset(env_ids=env_ids)
            self.observations = observations["actor"].clone()
            return self.observations, extras

    return PlaybackEnv(env)


def make_policy(args, env, metadata=None):
    import torch

    if args.agent == "zero":
        # A normalized zero action holds the configured default PD joint targets.
        return lambda obs: torch.zeros(env.num_envs, env.action_dim, device=env.device)
    if args.agent == "random":
        return lambda obs: env.random_actions()
    if metadata is None:
        raise ValueError("Trained playback requires checkpoint metadata")
    if metadata.get("algorithm") == "bfm_zero":
        from humanoid_rl.algorithm.bfm_zero.play import make_policy as make_bfm_policy
        return make_bfm_policy(args, env, metadata)
    if (metadata["observation_dim"], metadata["action_dim"]) != (
        env.observation_dim,
        env.action_dim,
    ):
        raise ValueError(
            "Checkpoint observation/action dimensions do not match the task"
        )
    cfg = metadata["agent"]
    if cfg["asymmetric_observation"]:
        raise ValueError(
            "This playback entrypoint currently supports the shared actor/critic observation"
        )
    from humanoid_rl.algorithm.flashsac.upstream import enable_flashsac

    enable_flashsac()
    from flash_rl.agents.flashSAC.network import FlashSACActor

    # Load only the actor: inference needs no replay, critic or optimizer.
    actor = FlashSACActor(
        num_blocks=cfg["actor_num_blocks"],
        input_dim=env.observation_dim,
        hidden_dim=cfg["actor_hidden_dim"],
        action_dim=env.action_dim,
    ).to(env.device)
    checkpoint = torch.load(
        args.checkpoint / "actor.pt", map_location=env.device, weights_only=True
    )
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["network_state_dict"].items()
    }
    actor.load_state_dict(state, strict=True)
    actor.eval()

    @torch.no_grad()
    def policy(observations):
        mean, _ = actor.get_mean_and_std(observations, training=False)
        return torch.tanh(mean)

    return policy


def run(args, metadata=None):
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    cfg = configure_scene(args, metadata)
    env = ManagerBasedRlEnv(
        cfg=cfg,
        device=args.device,
        render_mode="rgb_array" if args.screenshot else None,
    )
    try:
        playback = make_playback_env(env)
        policy = make_policy(args, playback, metadata)
        print(
            f"Task={args.task} agent={args.agent} envs={args.num_envs} seed={args.seed}",
            flush=True,
        )
        if args.task == "G1-Carry-Box":
            print("Orange box / cyan source support / green target support", flush=True)
        if args.screenshot:
            import imageio.v3 as imageio

            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(args.screenshot, env.render())
            print(f"Screenshot: {args.screenshot.resolve()}", flush=True)
        if args.headless:
            with torch.no_grad():
                for _ in range(args.steps):
                    actions = policy(playback.get_observations())
                    if not torch.isfinite(actions).all():
                        raise FloatingPointError("Nonfinite policy actions")
                    transition, _ = playback.step(actions)
                    if not torch.isfinite(transition["next_observation"]).all():
                        raise FloatingPointError("Nonfinite playback observation")
            print(f"Playback completed: {args.steps} vector steps.", flush=True)
            return
        from mjlab.viewer import NativeMujocoViewer, VerbosityLevel, ViserPlayViewer

        viewer_type = NativeMujocoViewer if args.viewer == "native" else ViserPlayViewer
        viewer = viewer_type(playback, policy, verbosity=VerbosityLevel.INFO)
        if args.paused:
            viewer.pause()
        if args.viewer == "native":
            print(
                "Enter: resample scene | Space: pause/play | Right: single step | ,/.: switch env | A: show all",
                flush=True,
            )
        if args.viewer == "native":
            import threading

            threads_before = set(threading.enumerate())
            try:
                viewer.run(num_steps=args.steps)
            finally:
                viewer.close()
                # MuJoCo 3.11's passive viewer closes asynchronously. Its daemon
                # render thread must finish simulate.destroy() before Python's
                # atexit GLFW termination, or finite play runs can segfault.
                # launch_passive exposes no thread handle; Python includes the
                # target name in the newly created thread's public name.
                for thread in threading.enumerate():
                    if thread not in threads_before and thread.name.endswith(
                        "(_launch_internal)"
                    ):
                        thread.join()
        else:
            viewer.run(num_steps=args.steps)
        if viewer.get_status().last_error:
            raise RuntimeError(viewer.get_status().last_error)
    finally:
        env.close()


def main():
    cli = parser()
    args = cli.parse_args()
    if args.list_tasks:
        print("\n".join(TASKS))
        return
    if args.task is None:
        cli.error("--task is required (or use --list-tasks)")
    if args.num_envs < 1 or not 0 <= args.env_index < args.num_envs:
        cli.error(
            "--num-envs must be positive and --env-index must select an existing environment"
        )
    if args.steps is not None and args.steps < 1:
        cli.error("--steps must be positive")
    if args.headless and (args.steps is None or args.paused):
        cli.error("--headless requires --steps and cannot be combined with --paused")
    if args.episode_length is not None and (
        not math.isfinite(args.episode_length) or args.episode_length <= 0
    ):
        cli.error("--episode-length must be positive and finite")
    if args.scene_config and args.task != "G1-Carry-Box":
        cli.error("--scene-config applies to G1-Carry-Box")
    args.agent = args.agent or ("trained" if args.checkpoint else "zero")
    if args.agent == "trained" and not args.checkpoint:
        cli.error("--agent trained requires --checkpoint")
    if args.checkpoint and args.agent != "trained":
        cli.error("--checkpoint is only used with --agent trained")
    metadata = None
    if args.checkpoint:
        if not (args.checkpoint/"run.json").is_file():
            cli.error(f"Checkpoint file missing: {args.checkpoint/'run.json'}")
        metadata = json.loads((args.checkpoint / "run.json").read_text())
        files = ("model/model.safetensors",) if metadata.get("algorithm") == "bfm_zero" else ("actor.pt",)
        for name in files:
            if not (args.checkpoint / name).is_file():
                cli.error(f"Checkpoint file missing: {args.checkpoint / name}")
        if metadata["task"] != args.task:
            cli.error("Checkpoint task must match --task")
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if args.viewer == "auto":
        args.viewer = "native" if has_display else "viser"
    if not args.headless and args.viewer == "native" and not has_display:
        cli.error(
            "Native viewer needs a desktop display; use --viewer viser or --headless --steps N"
        )
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    if args.headless or not has_display:
        os.environ.setdefault("MUJOCO_GL", "egl")
    run(args, metadata)


if __name__ == "__main__":
    main()
