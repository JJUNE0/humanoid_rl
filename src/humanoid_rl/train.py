"""Train registered mjlab tasks with FlashSAC or original BFM-Zero."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from humanoid_rl.tasks.registry import TASKS, MOTION_PRIOR_TASKS
from humanoid_rl.algorithm.configuration import load_train_config, merge_config


def parser(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--algorithm", choices=("flashsac", "bfm_zero"), default="flashsac")
    probe.add_argument("--task")
    probe.add_argument("--train-config", type=Path)
    probe.add_argument("--reward-mode")
    probe.add_argument("--amp-config", "--imitation-config")
    probe.add_argument("--amp-data", "--imitation-data")
    selected, _ = probe.parse_known_args(argv)
    try:
        settings = load_train_config(selected.algorithm, selected.train_config, task=selected.task)
        mode = selected.reward_mode or settings.get("training", {}).get("reward_mode")
        if mode is None and (selected.amp_config or selected.amp_data):
            mode = "amp"
        if selected.algorithm == "flashsac" and mode in ("amp", "add"):
            settings = merge_config(load_train_config("flashsac"), load_train_config(mode))
            if selected.train_config:
                settings = merge_config(settings, json.loads(selected.train_config.read_text()))
    except (OSError, ValueError) as error:
        probe.error(str(error))
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--algorithm", choices=("flashsac", "bfm_zero"), default=selected.algorithm)
    result.add_argument("--train-config", type=Path, help="JSON overrides for algorithm/<name>/configs/train.json")
    result.add_argument("--print-config", action="store_true", help="Print resolved training settings without creating a simulator")
    result.add_argument("--task", choices=TASKS)
    result.add_argument("--list-tasks", action="store_true")
    result.add_argument("--num-envs", type=int, default=1024)
    result.add_argument("--total-env-steps", type=int, default=10_000_000)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--observe-hand-feet", action=argparse.BooleanOptionalAction, default=None,
        help="Append root-relative wrist/ankle XYZs (12D); enabled for new carry runs. Resume restores the saved schema.",
    )
    result.add_argument("--stage-model", type=Path, help="Frozen carry MLP raw checkpoint or portable stage_model.pt bundle")
    result.add_argument("--stage-reference", type=Path, help="Source episode defining the raw MLP's joint/box/hand conventions")
    result.add_argument(
        "--scene-config", type=Path, help="YAML mapping of CarrySceneCfg ranges"
    )
    result.add_argument(
        "--episode-length", type=float, help="Episode duration in seconds"
    )
    result.add_argument("--buffer-size", type=int, default=1_000_000)
    result.add_argument("--min-buffer", type=int, default=100_000)
    result.add_argument("--buffer-device", default=None)
    result.add_argument("--batch-size", type=int, default=2048)
    result.add_argument("--updates-per-step", type=float, default=2.0)
    result.add_argument("--gamma", type=float)
    result.add_argument("--seed-steps", type=int)
    result.add_argument("--motion-data", type=Path)
    result.add_argument("--motion-manifest", type=Path, help="BFM screened manifest at its current storage location")
    result.add_argument("--motion-cache", type=Path, help="BFM feature cache at its current storage location")
    result.add_argument("--motion-files", nargs="+")
    result.add_argument("--motion-limit", type=int)
    result.add_argument("--motion-max-frames", type=int)
    result.add_argument("--hidden-dim", type=int)
    result.add_argument("--hidden-layers", type=int)
    result.add_argument("--z-dim", type=int)
    result.add_argument("--sequence-length", type=int)
    result.add_argument("--history-length", type=int)
    result.add_argument(
        "--n-step", type=int, help="Default: 3 for task rewards, 1 for AMP/ADD relabeling"
    )
    result.add_argument(
        "--reward-mode",
        choices=("task", "amp", "add"),
        help="AMP/ADD replace task rewards; imitation config selects termination rules",
    )
    result.add_argument(
        "--amp-config", "--imitation-config", type=Path, help="AMP/ADD discriminator/data configuration YAML"
    )
    result.add_argument(
        "--amp-data", "--imitation-data",
        type=Path,
        help="Expert rollout root containing prior_train/validation.txt",
    )
    result.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.compile (off during initial bring-up)",
    )
    result.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    result.add_argument(
        "--resume", type=Path, help="Checkpoint directory containing run.json"
    )
    result.add_argument(
        "--save-replay",
        action=argparse.BooleanOptionalAction,
        help="Also save replay for training continuation",
    )
    result.add_argument("--log-dir", type=Path, default=Path("logs/flashsac"))
    result.add_argument("--wandb-project", help="Also log metrics/configuration to this W&B project")
    result.add_argument("--wandb-entity", help="W&B account/team (defaults to the authenticated account)")
    result.add_argument("--wandb-name", help="Optional display name for this invocation")
    result.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    result.add_argument(
        "--save-every",
        type=int,
        default=1_000_000,
        help="Total environment transitions",
    )
    result.add_argument("--log-every", type=int, default=100, help="Vector steps")
    result.add_argument("--keep-checkpoints", type=int, help="Keep this many recent checkpoints in a new BFM run")
    result.add_argument("--eval-every", type=int, default=0, help="Deterministic evaluation interval in vector steps (0 disables)")
    result.add_argument("--eval-num-envs", type=int, default=64)
    result.add_argument("--eval-steps", type=int, default=500, help="50 Hz: 500 steps = 10 seconds")
    allowed = {action.dest for action in result._actions}
    unknown = set(settings.get("training", {}))-allowed
    if unknown:
        result.error(f"Unknown training settings: {sorted(unknown)}")
    result.set_defaults(**settings.get("training", {}))
    if selected.algorithm == "bfm_zero":
        result.set_defaults(**settings.get("model", {}))
    result.set_defaults(train_settings=settings,
                        explicit_options={item.split("=",1)[0] for item in argv if item.startswith("--")})
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    if args.list_tasks:
        print("\n".join(TASKS))
        return
    if args.task is None:
        cli.error("--task is required (or use --list-tasks)")
    if (args.wandb_entity or args.wandb_name or args.wandb_mode != "online") and not args.wandb_project:
        cli.error("W&B options require --wandb-project")
    if args.algorithm == "bfm_zero":
        if args.scene_config or args.amp_config or args.amp_data or args.stage_model or args.stage_reference or args.reward_mode or args.n_step:
            cli.error("BFM-Zero uses its train.json motion/environment settings; carry/AMP flags do not apply")
        if not args.compile:
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MUJOCO_GL", "egl")
        from humanoid_rl.algorithm.bfm_zero.runner import run
        try:
            run(args)
        except (ValueError, FileNotFoundError) as error:
            cli.error(str(error))
        return
    if args.task in MOTION_PRIOR_TASKS:
        cli.error("Motion-prior tasks require --algorithm bfm_zero")
    if args.print_config:
        settings = merge_config({}, args.train_settings)
        for name in settings["training"]:
            if hasattr(args, name):
                settings["training"][name] = getattr(args, name)
        print(json.dumps(settings, indent=2, default=str))
        return
    for name in (
        "num_envs",
        "total_env_steps",
        "buffer_size",
        "min_buffer",
        "batch_size",
        "n_step",
        "save_every",
        "log_every",
        "eval_num_envs",
        "eval_steps",
    ):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            cli.error(f"--{name.replace('_', '-')} must be positive")
    if args.eval_every < 0:
        cli.error("--eval-every must be nonnegative")
    if args.buffer_size < max(args.num_envs, args.min_buffer, args.batch_size):
        cli.error("--buffer-size must cover num-envs, min-buffer and batch-size")
    if not math.isfinite(args.updates_per_step) or args.updates_per_step <= 0:
        cli.error("--updates-per-step must be positive and finite")
    if args.gamma is not None and (not math.isfinite(args.gamma) or not 0 <= args.gamma < 1):
        cli.error("--gamma must be in [0,1)")
    if args.episode_length is not None and (
        not math.isfinite(args.episode_length) or args.episode_length <= 0
    ):
        cli.error("--episode-length must be positive and finite")
    if not args.compile:
        # Upstream also has function-level @torch.compile decorators.
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import yaml

    from humanoid_rl.tasks.carry_box.env_cfg import CarrySceneCfg
    from humanoid_rl.tasks.registry import make_task_cfg

    from humanoid_rl.algorithm.flashsac.runner import run

    metadata = None
    carry = None
    if args.resume:
        metadata = json.loads((args.resume / "run.json").read_text())
        if metadata["task"] != args.task:
            cli.error("Resume task must match the checkpoint")
        if args.total_env_steps <= metadata["env_steps"]:
            cli.error("--total-env-steps must exceed the saved step count")
        if args.scene_config:
            cli.error("Resume restores the saved scene; omit --scene-config")
        saved_hands = metadata.get("observe_hand_feet", False)
        if args.observe_hand_feet is not None and args.observe_hand_feet != saved_hands:
            cli.error("Resume must preserve the saved hand/foot observation schema; use a fresh run to change it")
        args.observe_hand_feet = saved_hands
        if args.episode_length is None:
            args.episode_length = metadata["episode_length_s"]
        args.n_step = metadata["agent"]["n_step"]
        args.updates_per_step = metadata["updates_per_step"]
        values = metadata.get("carry")
        if values is not None:
            carry = CarrySceneCfg(
                **{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
            )
    elif args.scene_config:
        if args.task != "G1-Carry-Box":
            cli.error("--scene-config applies to G1-Carry-Box")
        values = yaml.safe_load(args.scene_config.read_text())
        if not isinstance(values, dict):
            cli.error("--scene-config must contain a YAML mapping")
        carry = CarrySceneCfg(
            **{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
        )
    saved_mode = metadata.get("reward_mode", "task") if metadata else None
    args.reward_mode = (
        args.reward_mode
        or saved_mode
        or ("amp" if args.amp_config or args.amp_data else "task")
    )
    if saved_mode is not None and args.reward_mode != saved_mode:
        cli.error("Resume must preserve the checkpoint reward mode")
    args.n_step = args.n_step or (1 if args.reward_mode in ("amp", "add") else 3)
    args.amp = None
    if args.reward_mode in ("amp", "add"):
        if args.task != "G1-Carry-Box":
            cli.error("AMP/ADD currently support G1-Carry-Box")
        if args.n_step != 1:
            cli.error("AMP/ADD reward relabeling requires --n-step 1")
        if args.reward_mode == "add":
            from humanoid_rl.algorithm.add.config import config_from_values
        else:
            from humanoid_rl.algorithm.amp.config import config_from_values

        values = metadata[args.reward_mode]["config"] if metadata else merge_config(
            load_train_config(args.reward_mode)["imitation"], args.train_settings.get("imitation", {}))
        if args.amp_config:
            if metadata:
                cli.error("Resume restores imitation config; omit --imitation-config")
            override = yaml.safe_load(args.amp_config.read_text())
            if not isinstance(override, dict):
                cli.error("--amp-config must contain a YAML mapping")
            values = merge_config(values, override)
        try:
            args.amp = config_from_values(values, args.amp_data)
        except (TypeError, ValueError) as error:
            cli.error(str(error))
    elif args.amp_config or args.amp_data:
        cli.error("Imitation data/config requires --reward-mode amp or add")
    args.stage_model_cfg = None
    if args.resume:
        if args.stage_model or args.stage_reference:
            cli.error("Resume restores the frozen stage predictor; omit --stage-model/--stage-reference")
        if metadata.get("stage_model") is not None:
            from humanoid_rl.tasks.carry_box.stage.config import StageModelConfig
            args.stage_model_cfg = StageModelConfig.from_metadata(metadata["stage_model"], args.resume)
    elif args.stage_model:
        if args.task != "G1-Carry-Box":
            cli.error("--stage-model currently supports G1-Carry-Box")
        from humanoid_rl.tasks.carry_box.stage.config import StageModelConfig
        try:
            if args.stage_reference:
                args.stage_model_cfg = StageModelConfig.from_reference(args.stage_model, args.stage_reference)
            else:
                import pickle
                try:
                    args.stage_model_cfg = StageModelConfig.from_bundle(args.stage_model)
                except (pickle.UnpicklingError, ValueError):
                    if args.amp is None:
                        cli.error("Raw --stage-model needs --stage-reference or AMP data")
                    root = Path(args.amp.data_root)
                    relative = next(line.strip() for line in (root / "prior_train.txt").read_text().splitlines() if line.strip())
                    reference = (root / relative).parent
                    args.stage_model_cfg = StageModelConfig.from_reference(args.stage_model, reference)
        except (ValueError, FileNotFoundError, KeyError) as error:
            cli.error(str(error))
    elif args.stage_reference:
        cli.error("--stage-reference requires --stage-model")
    if args.observe_hand_feet is None:
        args.observe_hand_feet = True
    cfg = make_task_cfg(args.task, carry, observe_hand_feet=args.observe_hand_feet, stage_model=args.stage_model_cfg)
    run(args, cfg, metadata)


if __name__ == "__main__":
    main()
