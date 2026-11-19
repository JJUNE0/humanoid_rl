"""Small vectorized collector around the original FlashSAC agent and replay."""

import json
import math
import random
import shutil
import time
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from humanoid_rl.algorithm.flashsac.config import make_agent_config
from humanoid_rl.env.mjlab_adapter import MjlabAdapter
from humanoid_rl.algorithm.flashsac.upstream import enable_flashsac
from humanoid_rl.algorithm.flashsac.checkpoint import is_replay_io_error, replay_directory, save_replay_atomic, write_json_atomic


def run(args, env_cfg, resume_metadata=None):
    enable_flashsac()
    from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig
    from mjlab.envs import ManagerBasedRlEnv

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.auto_reset = False
    if args.episode_length is not None:
        env_cfg.episode_length_s = args.episode_length

    if args.reward_mode in ("amp", "add"):
        from humanoid_rl.env.imitation.amp import configure_amp_environment

        fixed_scene = None
        controller = None
        if args.resume and args.amp.fixed_initial_state:
            from humanoid_rl.tasks.carry_box.fixed_scene import load_scene_bundle
            fixed_scene = load_scene_bundle(args.resume)
        if args.resume and args.amp.controller_profile == "reference":
            from robots.humanoid.g1.control import load_controller_bundle
            controller = load_controller_bundle(args.resume)
            if controller["fingerprint"] != resume_metadata["reference_controller"]["fingerprint"]:
                raise ValueError("Checkpoint reference controller and metadata differ")
        if args.reward_mode == "add":
            from humanoid_rl.env.imitation.add import configure_add_environment
            configure_add_environment(env_cfg, args.amp, fixed_scene=fixed_scene, controller=controller)
        else:
            configure_amp_environment(env_cfg, args.amp.reference_init_probability,
                                      amp_config=args.amp, fixed_scene=fixed_scene, controller=controller)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    adapter = writer = evaluator = tracker = None
    completed = False
    try:
        amp = feature_fn = control_fn = None
        if args.reward_mode in ("amp", "add"):
            from humanoid_rl.algorithm.amp.dataset import MotionDataset
            from humanoid_rl.algorithm.amp.features import EnvironmentFeatures
            from humanoid_rl.algorithm.amp.trainer import AmpTrainer

            feature_fn = EnvironmentFeatures(env, args.amp.observe_object)
            dataset = MotionDataset(
                args.amp.data_root,
                feature_fn.joint_names,
                args.amp.cache_dir,
                args.amp.max_motions,
                observe_object=args.amp.observe_object,
                single_motion=args.amp.single_motion,
            )
            if args.reward_mode == "add":
                from humanoid_rl.algorithm.add.features import EnvironmentAddFeatures, tracking_dataset
                from humanoid_rl.algorithm.add.trainer import AddTrainer
                dataset = tracking_dataset(dataset)
                env_cfg.add_reference = dataset.train.states
                feature_fn = EnvironmentAddFeatures(env, dataset.train.states)
                amp = AddTrainer(args.amp, dataset, args.device)
            else:
                amp = AmpTrainer(args.amp, dataset, args.device)
            env.amp_dataset = dataset
            if args.amp.control is not None:
                from humanoid_rl.env.control import ControlTerms
                control_fn = ControlTerms(env, args.amp.control)
            if not math.isclose(env.step_dt, 0.02, abs_tol=1e-8):
                raise ValueError("Expert transitions require a 50 Hz environment")
        adapter = MjlabAdapter(env, feature_fn=feature_fn, control_fn=control_fn)
        # Includes both observations, action, reward, flags and n-step discount.
        replay_bytes = (
            args.buffer_size
            * (
                2 * adapter.observation_dim
                + adapter.action_dim
                + 4
                + (2 * amp.feature_dim if amp is not None else 0)
                + (6 if control_fn is not None else 0)
            )
            * 4
        )
        args.buffer_device = args.buffer_device or args.device
        if args.buffer_device.startswith("cuda"):
            free_bytes = torch.cuda.mem_get_info(args.buffer_device)[0]
            if replay_bytes > free_bytes * 0.8:
                raise ValueError(
                    f"Replay needs {replay_bytes / 1e9:.2f} GB; reduce --buffer-size "
                    "or select --buffer-device cpu"
                )
        print(
            f"Observation={adapter.observation_dim}, action={adapter.action_dim}, "
            f"replay={replay_bytes / 1e9:.2f} GB on {args.buffer_device}",
            flush=True,
        )
        cfg = make_agent_config(args)
        start_step = 0
        if resume_metadata is not None:
            # Keep the saved learning schedule/algorithm; permit runtime/storage changes.
            values = resume_metadata["agent"] | {
                key: getattr(cfg, key)
                for key in (
                    "device_type",
                    "buffer_device_type",
                    "buffer_max_length",
                    "buffer_min_length",
                    "sample_batch_size",
                    "use_compile",
                    "compile_mode",
                    "use_amp",
                )
            }
            cfg = FlashSACConfig(**values)
            start_step = resume_metadata["env_steps"]
        if amp is not None:
            from humanoid_rl.algorithm.amp.agent import AmpFlashSACAgent

            agent = AmpFlashSACAgent(
                adapter.observation_space, adapter.action_space, {}, cfg, amp
            )
        else:
            agent = FlashSACAgent(
                adapter.observation_space, adapter.action_space, {}, cfg
            )
        if args.resume:
            agent.load(str(args.resume))
            replay_path = replay_directory(args.resume)
            if replay_path is not None:
                agent.load_replay_buffer(str(replay_path))
            else:
                print(
                    "Replay not saved; refilling from fresh mjlab episodes.", flush=True
                )
            # Simulator episodes restart, so per-environment return traces must restart too.
            if agent.reward_normalizer is not None:
                agent.reward_normalizer.G_r = torch.zeros(
                    args.num_envs, device=args.device
                )
        log_dir = args.log_dir / args.task / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_dir.mkdir(parents=True)
        writer = SummaryWriter(str(log_dir))
        metadata = {
            "task": args.task,
            "agent": asdict(agent._cfg),
            "carry": asdict(env_cfg.carry) if hasattr(env_cfg, "carry") else None,
            "episode_length_s": env_cfg.episode_length_s,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "observation_dim": adapter.observation_dim,
            "observe_hand_feet": getattr(env_cfg, "observe_hand_feet", False),
            "stage_model": env_cfg.stage_model.metadata() if getattr(env_cfg, "stage_model", None) is not None else None,
            "action_dim": adapter.action_dim,
            "updates_per_step": args.updates_per_step,
            "reward_mode": args.reward_mode,
            "reference_controller": getattr(env_cfg, "reference_controller", None),
            "evaluation": {"every_vector_steps": args.eval_every, "num_envs": args.eval_num_envs, "steps": args.eval_steps},
            (args.reward_mode if amp is not None else "amp"): {
                "config": asdict(args.amp),
                "feature_dim": amp.feature_dim,
                "feature_version": amp.feature_version,
                "feature_groups": amp.feature_groups,
                "reference_initialization": "fixed_frame_0" if args.amp.fixed_initial_state else (
                    "robot_box_supports_goal" if args.amp.observe_object else "robot_only"),
                "joint_names": feature_fn.joint_names,
                "train_fingerprint": amp.dataset.train.fingerprint,
                "validation_fingerprint": amp.dataset.validation.fingerprint if amp.dataset.validation is not None else None,
                "train_clips": amp.dataset.train.paths,
                "validation_clips": amp.dataset.validation.paths if amp.dataset.validation is not None else [],
                "held_out_validation_enabled": amp.dataset.validation is not None,
            }
            if amp is not None
            else None,
            "config_reference": "wolf-quadruped/rl/wolf-stair-rl/scripts/flash_rl/train.py",
        }
        if getattr(env_cfg, "fixed_scene", None) is not None:
            from humanoid_rl.tasks.carry_box.fixed_scene import scene_metadata
            metadata["fixed_scene"] = scene_metadata(env_cfg.fixed_scene)
            metadata["evaluation"]["best_checkpoint_criteria"] = [
                "placement_condition_fraction", "bilateral_box_lift_5cm_fraction",
                "bilateral_hand_box_contact_fraction", "hand_box_contact_fraction",
                "negative_min_box_goal_error_m", "survival_horizon", "upright_time_s",
            ]
        if args.reward_mode == "add":
            metadata["add"].update({
                "discriminator_input": "consecutive time-aligned reference-minus-policy residuals",
                "positive_sample": "zero vector",
                "actor_reference_condition": "normalized reference phase (1D)",
                "discriminator_input_dim": 2*amp.feature_dim,
                "gradient_penalty_inputs": "policy residuals and zero residual",
                "reward": "softplus(logit), capped at -log(1e-4)",
                "reference_end": "true terminal; no bootstrap beyond final frame",
            })
            metadata["evaluation"]["best_checkpoint_criteria"].insert(0, "negative_reference_tracking_rms")
        if metadata["reference_controller"] is not None and "fixed_scene" in metadata:
            metadata["fixed_scene"]["robot_dynamics"] = "native mjlab G1 geometry; recorded joint physics and explicit PD/action mapping"
        if getattr(args, "wandb_project", None):
            from humanoid_rl.algorithm.flashsac.tracking import WandbLogger

            tracker = WandbLogger(metadata, log_dir, project=args.wandb_project,
                                  entity=args.wandb_entity, name=args.wandb_name, mode=args.wandb_mode)
            metadata["wandb"] = tracker.metadata()
        (log_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Logs: {log_dir}", flush=True)

        def log_metrics(metrics, step):
            for key, value in metrics.items():
                writer.add_scalar(key, value, step)
            writer.flush()
            if tracker is not None:
                tracker.log(metrics, step)

        last_save_step = None

        def save(step, *, best=False):
            nonlocal last_save_step
            path = log_dir / (f"best_step_{step}" if best else f"step_{step}")
            agent.save(str(path))
            if hasattr(env, "carry_stage_observer"):
                env.carry_stage_observer.predictor.save_bundle(path / "stage_model.pt")
            if getattr(env_cfg, "fixed_scene", None) is not None:
                from humanoid_rl.tasks.carry_box.fixed_scene import BUNDLE_NAME
                torch.save(env_cfg.fixed_scene, path / BUNDLE_NAME)
            if getattr(env_cfg, "reference_controller", None) is not None:
                from robots.humanoid.g1.control import BUNDLE_NAME as CONTROLLER_BUNDLE
                torch.save(env_cfg.reference_controller, path / CONTROLLER_BUNDLE)
            checkpoint_metadata = metadata | {"env_steps": step, "replay_save": {"status": "not_requested"}}
            # A replay export failure must not hide already saved models or
            # prevent the final policy evaluation and metrics from being logged.
            write_json_atomic(path / "run.json", checkpoint_metadata)
            if args.save_replay and not best:
                try:
                    if shutil.disk_usage(path).free < replay_bytes + 64 * 2**20:
                        raise OSError("Insufficient free disk for replay export")
                    save_replay_atomic(agent, path)
                    checkpoint_metadata["replay_save"] = {"status": "complete", "directory": "replay"}
                except (OSError, RuntimeError) as error:
                    if not is_replay_io_error(error):
                        raise
                    checkpoint_metadata["replay_save"] = {"status": "failed", "error": str(error)}
                    print(f"Replay export failed; models remain usable at {path}: {error}", flush=True)
                    log_metrics({"checkpoint/replay_save_failed": 1.0}, step)
                write_json_atomic(path / "run.json", checkpoint_metadata)
            if best:
                link = log_dir / "best.tmp"
                link.symlink_to(path.name)
                link.replace(log_dir / "best")
            else:
                last_save_step = step

        best_score = None

        def evaluate(step):
            nonlocal best_score
            metrics = evaluator.evaluate(agent)
            log_metrics(metrics, step)
            with (log_dir / "evaluation.jsonl").open("a") as stream:
                stream.write(json.dumps({"env_steps": step} | metrics) + "\n")
            print(f"evaluation steps={step} {json.dumps(metrics)}", flush=True)
            if getattr(env_cfg, "fixed_scene", None) is not None:
                score = tuple(metrics[f"eval/{key}"] for key in (
                    "placement_condition_fraction", "bilateral_box_lift_5cm_fraction",
                    "bilateral_hand_box_contact_fraction", "hand_box_contact_fraction",
                )) + (-metrics["eval/min_box_goal_error_m"], metrics["eval/survival_horizon"], metrics["eval/upright_time_s"])
            else:
                score = (metrics.get("eval/survival_5s", metrics["eval/survival_horizon"]), metrics["eval/upright_time_s"])
            if args.reward_mode == "add":
                score = (-metrics["eval/reference_tracking_rms"],)+score
            if best_score is None or score > best_score:
                best_score = score
                save(step, best=True)

        if args.eval_every:
            from humanoid_rl.algorithm.flashsac.evaluation import PolicyEvaluator
            evaluator = PolicyEvaluator(env_cfg, args.eval_num_envs, args.eval_steps, args.seed, args.device)
            evaluate(start_step)

        episode_return = torch.zeros(args.num_envs, device=args.device)
        episode_length = torch.zeros_like(episode_return)
        reason_counts = {}
        completed_returns, completed_lengths = [], []
        update_credit = 0.0
        env_steps = start_step
        started = time.monotonic()
        next_save = (start_step // args.save_every + 1) * args.save_every
        updates = {}
        for vector_step in range(
            1, math.ceil((args.total_env_steps - start_step) / args.num_envs) + 1
        ):
            if agent.can_start_training():
                actions = agent.sample_actions(
                    vector_step,
                    {"next_observation": adapter.observations},
                    training=True,
                    return_tensor=True,
                )
            else:
                actions = adapter.random_actions()
            transition, reasons = adapter.step(actions)
            if amp is not None:
                pairs = torch.cat(
                    (transition["amp_observation"], transition["amp_next_observation"]),
                    -1,
                )
                transition["reward"] = amp.reward(pairs)
            if (
                not torch.isfinite(transition["next_observation"]).all()
                or not torch.isfinite(transition["reward"]).all()
            ):
                raise FloatingPointError("Nonfinite mjlab observation or reward")
            agent.process_transition(transition)
            env_steps += args.num_envs
            episode_return += transition["reward"]
            episode_length += 1
            done = transition["terminated"] | transition["truncated"]
            if done.any():
                completed_returns.extend(episode_return[done].tolist())
                completed_lengths.extend(episode_length[done].tolist())
                episode_return[done] = 0
                episode_length[done] = 0
            for name, mask in reasons.items():
                reason_counts[name] = reason_counts.get(name, 0) + int(mask.sum())
            if agent.can_start_training():
                update_credit += args.updates_per_step
                while update_credit >= 1:
                    updates.update(agent.update())
                    if not all(math.isfinite(value) for value in updates.values()):
                        raise FloatingPointError(
                            f"Nonfinite FlashSAC metrics: {updates}"
                        )
                    update_credit -= 1
            if vector_step % args.log_every == 0 or env_steps >= args.total_env_steps:
                fps = (env_steps - start_step) / max(time.monotonic() - started, 1e-6)
                metrics = updates | {
                    "train/fps": fps,
                    "train/updates": agent._update_step,
                    "train/reward_mean": float(transition["reward"].mean()),
                    "train/vector_steps": math.ceil(env_steps / args.num_envs),
                }
                if hasattr(env, "carry_stage_observer"):
                    metrics.update(env.carry_stage_observer.metrics())
                if completed_returns:
                    metrics["episode/return"] = float(np.mean(completed_returns))
                    metrics["episode/length"] = float(np.mean(completed_lengths))
                    if "success" in reason_counts:
                        metrics["episode/success_rate"] = reason_counts[
                            "success"
                        ] / len(completed_returns)
                metrics.update(
                    {f"termination/{k}": v for k, v in reason_counts.items()}
                )
                log_metrics(metrics, env_steps)
                print(
                    f"steps={env_steps} updates={agent._update_step} fps={fps:.0f} "
                    f"reward={metrics['train/reward_mean']:.4f}",
                    flush=True,
                )
                completed_returns.clear()
                completed_lengths.clear()
                reason_counts.clear()
            if env_steps >= next_save:
                save(env_steps)
                next_save = (env_steps // args.save_every + 1) * args.save_every
            if evaluator is not None and (vector_step % args.eval_every == 0 or env_steps >= args.total_env_steps):
                evaluate(env_steps)
        if last_save_step != env_steps:
            save(env_steps)
        completed = True
        return log_dir
    finally:
        if writer is not None:
            writer.close()
        if tracker is not None:
            tracker.finish(exit_code=0 if completed else 1)
        if evaluator is not None:
            evaluator.close()
        env.close()
