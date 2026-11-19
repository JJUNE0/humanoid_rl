"""Online FB-CPR-Aux training using original agents and replay on mjlab H1."""

from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import tempfile
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from humanoid_rl.algorithm.bfm_zero.config import config_from_args, make_agent_config, resolve_agent_config, validate_config
from humanoid_rl.algorithm.bfm_zero.dataset import MotionDataset
from humanoid_rl.algorithm.bfm_zero.upstream import BFM_ROOT, enable_bfm_zero
from humanoid_rl.env.bfm_adapter import BfmMjlabAdapter
from humanoid_rl.tasks.motion_prior.env_cfg import MotionPriorEnvConfig, make_env_cfg
from humanoid_rl.tasks.registry import MOTION_PRIOR_TASKS
from humanoid_retarget.datasets.paths import relocate_motion
from humanoid_rl.launchers.source import provenance


def run(args):
    if args.task not in MOTION_PRIOR_TASKS:
        raise ValueError(f"BFM-Zero requires a motion-prior task: {MOTION_PRIOR_TASKS}")
    values = config_from_args(args)
    resumed = json.loads((args.resume/"run.json").read_text()) if args.resume else None
    if resumed:
        if resumed.get("algorithm") != "bfm_zero" or resumed["task"] != args.task:
            raise ValueError("Resume checkpoint must match the BFM-Zero robot/task")
        if args.train_config:
            raise ValueError("Resume restores train.json; omit --train-config")
        restored = resumed["train_config"]
        runtime = ("num_envs", "total_env_steps", "device", "buffer_device", "save_every", "log_every", "log_dir", "save_replay")
        for key in runtime:
            if f"--{key.replace('_','-')}" in args.explicit_options:
                restored["training"][key] = values["training"][key]
        restored["motion"] = relocate_motion(restored["motion"], root=getattr(args, "motion_data", None),
                                             manifest=getattr(args, "motion_manifest", None),
                                             cache=getattr(args, "motion_cache", None))
        values = restored
    agent_cfg = resolve_agent_config(values, args.task, legacy=resumed is not None)
    validate_config(values)
    training = values["training"]
    if args.print_config:
        print(json.dumps(values, indent=2))
        return
    enable_bfm_zero()
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent
    from humanoidverse.agents.buffers.transition import DictBuffer
    from mjlab.envs import ManagerBasedRlEnv
    start_step = resumed["env_steps"] if resumed else 0
    if training["total_env_steps"] <= start_step:
        raise ValueError("total_env_steps must exceed the saved environment step")
    random.seed(training["seed"])
    np.random.seed(training["seed"])
    torch.manual_seed(training["seed"])
    torch.set_float32_matmul_precision("high")
    if training["device"].startswith("cuda"):
        torch.cuda.set_device(training["device"])
    if agent_cfg.robot == "g1":
        from robots.humanoid.g1.bfm_zero import load_control
    elif agent_cfg.robot == "humanoid_light":
        from robots.humanoid.humanoid_light.mjlab import load_control
    else:
        from robots.humanoid.h1.mjlab import load_control
    control = resumed["robot_control"] if resumed else load_control()
    env_settings = MotionPriorEnvConfig(**values["environment"])
    dataset = MotionDataset(values["motion"], env_settings, control, values["model"]["sequence_length"],
                            expert_device=training["device"],agent_cfg=agent_cfg)
    if resumed and dataset.fingerprint != resumed["expert_data"]["fingerprint"]:
        raise ValueError("Resume expert motions/default joint pose/feature schema differ")
    variable_rollout = values["motion"].get("variable_expert_rollout", False)
    if values["agent"]["rollout_expert_trajectories"] and not variable_rollout:
        length = values["agent"]["rollout_expert_trajectories_length"]
        if max(clip["frames"] for clip in dataset.clips) <= length:
            raise ValueError("Expert tracking rollouts need a clip longer than rollout_expert_trajectories_length")
    cfg = make_env_cfg(env_settings, control, agent_cfg=agent_cfg)
    cfg.scene.num_envs, cfg.seed = training["num_envs"], training["seed"]
    env = ManagerBasedRlEnv(cfg, device=training["device"])
    writer = tracker = None
    try:
        adapter = BfmMjlabAdapter(env, dataset)
        bytes_per_transition = 4*(2*sum(value.shape[-1] for value in adapter.observations.values())
                                  + adapter.action_dim+values["model"]["z_dim"]+len(values["auxiliary_rewards"]))+2
        replay_bytes = training["buffer_size"]*bytes_per_transition
        if training["buffer_device"].startswith("cuda"):
            available = torch.cuda.mem_get_info(training["buffer_device"])[0]
        else:
            available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        if replay_bytes > available-8*2**30:
            raise ValueError(f"Replay requires {replay_bytes/2**30:.2f} GiB; reduce buffer_size or select another buffer_device")
        print(f"BFM observations={dict((k,v.shape[-1]) for k,v in adapter.observations.items())}, "
              f"actions={adapter.action_dim}, replay={replay_bytes/2**30:.3f} GiB on {training['buffer_device']}", flush=True)
        if resumed:
            agent = FBcprAuxAgent.load(str(args.resume), device=torch.device(training["device"]).type)
        else:
            agent = make_agent_config(values).build(adapter.observation_space, adapter.action_dim)
        replay = DictBuffer(capacity=training["buffer_size"], device=training["buffer_device"])
        if resumed and (args.resume/"replay").is_dir():
            replay = DictBuffer.load(str(args.resume/"replay"), device=training["buffer_device"])
            # Upstream's load() currently accepts device without applying it.
            from torch.utils._pytree import tree_map
            replay.storage = tree_map(lambda value:value.to(training["buffer_device"]), replay.storage)
            replay.device = training["buffer_device"]
        elif resumed:
            print("Replay not saved; refilling from fresh mjlab episodes.", flush=True)
        buffers = {"train": replay, "expert_slicer": dataset.expert_slicer}
        context = agent.maybe_update_rollout_context
        if variable_rollout:
            from humanoid_rl.algorithm.bfm_zero.rollout import VariableExpertRollout
            context = VariableExpertRollout(dataset, agent, env.num_envs, env.device)
        run_dir = Path(training["log_dir"])/args.task/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True)
        writer = SummaryWriter(str(run_dir))
        metadata = {"algorithm": "bfm_zero", "task": args.task, "train_config": values,
                    "robot_control": control, "expert_data": dataset.metadata(),
                    "observation_dims": {key:value.shape[-1] for key,value in adapter.observations.items()},
                    "action_dim": adapter.action_dim, "episode_length_s": env_settings.episode_length_s,
                    "num_envs": env.num_envs, "seed": training["seed"],
                    "source_revision": provenance(BFM_ROOT.parents[1]).get("repositories", {}).get("third_party/BFM-Zero", {}).get("revision"),
                    "replay_bytes_estimate": replay_bytes, "updates": resumed.get("updates",0) if resumed else 0}
        (run_dir/"train.json").write_text(json.dumps(values, indent=2)+"\n")
        (run_dir/"run.json").write_text(json.dumps(metadata | {"env_steps":start_step},indent=2)+"\n")
        if args.wandb_project:
            import wandb
            tracker = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                                 name=args.wandb_name or run_dir.name, mode=args.wandb_mode,
                                 dir=str(run_dir), config=metadata, tags=["bfm-zero","fb-cpr-aux","mjlab",agent_cfg.robot],
                                 settings=wandb.Settings(disable_git=True, save_code=False))
        env_steps, updates = start_step, metadata["updates"]
        z = None
        started = time.monotonic()
        last_checkpoint = start_step

        def save_checkpoint():
            nonlocal last_checkpoint
            checkpoints = run_dir/"checkpoints"
            checkpoints.mkdir(exist_ok=True)
            destination = checkpoints/f"step_{env_steps}"
            if destination.exists():
                return destination
            temporary = Path(tempfile.mkdtemp(prefix=".writing-", dir=checkpoints))
            agent.save(str(temporary))
            if training["save_replay"]:
                replay.save(temporary/"replay")
            saved = metadata | {"env_steps":env_steps, "updates":updates}
            (temporary/"run.json").write_text(json.dumps(saved, indent=2)+"\n")
            temporary.replace(destination)
            (run_dir/"run.json").write_text(json.dumps(saved | {"latest_checkpoint":str(destination.resolve())}, indent=2)+"\n")
            last_checkpoint = env_steps
            print(f"BFM checkpoint: {destination}", flush=True)
            keep = training.get("keep_checkpoints")
            if keep is not None:
                saved_paths = sorted(checkpoints.glob("step_*"),key=lambda path:int(path.name[5:]))
                for old in saved_paths[:-keep]:
                    shutil.rmtree(old)
            return destination

        while env_steps < training["total_env_steps"]:
            with torch.no_grad():
                z = context(z, adapter.step_count, buffers)
                if env_steps-start_step < training["seed_steps"]:
                    action = torch.empty(env.num_envs, adapter.action_dim, device=env.device).uniform_(-1,1)
                else:
                    action = agent.act(adapter.observations, z, mean=False)
            transition, metrics = adapter.step(action)
            transition["z"] = z.detach().clone()
            replay.extend(transition)
            env_steps += env.num_envs
            if len(replay) >= training["min_buffer"] and env_steps-start_step >= training["seed_steps"]:
                accumulated = {}
                for _ in range(training["updates_per_step"]):
                    output = agent.update(buffers, updates)
                    for key,value in output.items():
                        value = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
                        if not np.isfinite(value):
                            raise FloatingPointError(f"Nonfinite BFM update metric: {key}={value}")
                        accumulated[key] = accumulated.get(key,0)+value
                    updates += 1
                metrics.update({f"bfm/{key}":value/training["updates_per_step"] for key,value in accumulated.items()})
            vector_steps = (env_steps-start_step)//env.num_envs
            if vector_steps % training["log_every"] == 0 or env_steps >= training["total_env_steps"]:
                metrics.update(env_steps=env_steps, updates=updates, replay_size=len(replay),
                               transitions_per_second=(env_steps-start_step)/max(time.monotonic()-started,1e-6))
                for key,value in metrics.items():
                    writer.add_scalar(key, value, env_steps)
                if tracker is not None:
                    tracker.log(metrics)
                with (run_dir/"metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics)+"\n")
                print(json.dumps(metrics), flush=True)
            if env_steps-last_checkpoint >= training["save_every"]:
                save_checkpoint()
        save_checkpoint()
        return run_dir
    finally:
        if writer is not None:
            writer.close()
        if tracker is not None:
            tracker.finish()
        env.close()
