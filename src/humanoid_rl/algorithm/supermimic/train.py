"""Online Light tracking using the official SuperMimic WM and BPTT learners.

Owns local/Colab paths, screened-motion preparation, diagnostics and resumable
checkpoints. Learning equations live in the pinned upstream implementation.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time


def parse_tasks_cfg(raw_tasks):
    """Deserialize optional JSON runner.tasks dict into SuperMimic TasksCfg."""
    from supermimic.tasks.task_spec import TasksCfg, VelocityTaskCfg

    if not raw_tasks:
        return TasksCfg()
    unknown = set(raw_tasks) - {"velocity"}
    if unknown:
        raise ValueError(f"Unknown runner.tasks keys: {sorted(unknown)}")
    vel_raw = dict(raw_tasks.get("velocity", {}))
    vel_kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in vel_raw.items()}
    return TasksCfg(velocity=VelocityTaskCfg(**vel_kwargs))


def build_locomotion_frame_sampler(packed_dir, motion, allowlist_path, device):
    """Build speed-stratified (stand/walk/run) frame indices & sampling weights for velocity slots."""
    import torch

    motion_names = json.loads((Path(packed_dir) / "motion_names.json").read_text())
    allow_data = json.loads(Path(allowlist_path).read_text())
    allow_prefixes = set(allow_data["sequence_prefixes"])
    ends = motion.motion_ends.detach().cpu().tolist()
    frame_chunks = []
    for c_idx, name in enumerate(motion_names):
        prefix = "/".join(name.split("/")[:2])
        if prefix in allow_prefixes:
            s = 0 if c_idx == 0 else int(ends[c_idx - 1])
            e = int(ends[c_idx])
            stop = max(s + 1, e - 10)
            frame_chunks.append(torch.arange(s, stop, dtype=torch.long, device=device))
    if not frame_chunks:
        raise ValueError(f"No packed clips matched velocity_allowlist: {allowlist_path}")
    frame_ids = torch.cat(frame_chunks, dim=0)
    v_xy = torch.linalg.vector_norm(motion.root_lin_vel_w[frame_ids, :2], dim=-1)
    stand_mask = v_xy < 0.10
    walk_mask = (v_xy >= 0.10) & (v_xy < 1.60)
    run_mask = v_xy >= 1.60
    probs = torch.zeros_like(v_xy, dtype=torch.float32)
    for mask, target_mass in ((stand_mask, 0.25), (walk_mask, 0.50), (run_mask, 0.25)):
        cnt = int(mask.sum().item())
        if cnt > 0:
            probs[mask] = target_mass / cnt
    total = float(probs.sum().item())
    if total <= 0.0:
        probs.fill_(1.0 / frame_ids.numel())
    else:
        probs /= total
    return frame_ids, probs


def check_resume_settings(prior, current):
    """Batch sizes and update frequency may change; model and task stay fixed."""
    before, after = deepcopy(prior["runner"]), deepcopy(current["runner"])
    changes = {}
    for learner in ("wm", "policy"):
        old, new = before[learner].pop("batch_size"), after[learner].pop("batch_size")
        if old != new:
            changes[f"runner.{learner}.batch_size"] = [old, new]
    if before != after:
        raise ValueError("Resume runner changed beyond batch sizes")
    if prior["environment"] != current["environment"]:
        raise ValueError("Resume environment changed")
    for key in ("seed", "warmup_steps", "buffer_size"):
        if prior["training"][key] != current["training"][key]:
            raise ValueError(f"Resume training.{key} changed")
    for key in ("num_envs", "updates_per_step", "compile", "cuda_graphs", "muon_dtype"):
        old = prior["training"].get(key, prior["training"].get("compile") if key == "cuda_graphs" else "bfloat16" if key == "muon_dtype" else None)
        new = current["training"].get(key, current["training"].get("compile") if key == "cuda_graphs" else "bfloat16" if key == "muon_dtype" else None)
        if old != new:
            changes[f"training.{key}"] = [old, new]
    return changes


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--resume", type=Path)
    cli.add_argument("--benchmark", action="store_true", help="Measure steady throughput without saving a training checkpoint")
    args = cli.parse_args()
    settings = json.loads(args.config.read_text())
    if not settings["training"]["compile"]:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
    else:
        os.environ.pop("TORCHDYNAMO_DISABLE", None)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    if settings.get("tuning") and not args.benchmark:
        from humanoid_rl.algorithm.supermimic.tune import run
        run(settings, args.output, args.resume)
    else:
        train(settings, args.output, args.resume, benchmark=args.benchmark)


def train(settings, output, resume=None, *, benchmark=False):
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg
    from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
    from supermimic.models import Dynamics
    from supermimic.representations import GenCoordRepresentation
    from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
    from supermimic.tasks.task_spec import TasksCfg, VelocityTaskCfg
    from supermimic.utils.tracking_reward import tracking_reward

    from humanoid_rl.algorithm.supermimic.motions import prepare
    from humanoid_rl.algorithm.supermimic.muon import configure as configure_muon
    from humanoid_rl.algorithm.supermimic.performance import first_nonfinite
    from humanoid_rl.env.actions import joint_range_targets
    from humanoid_rl.launchers.config import write_json
    from humanoid_rl.tasks.motion_tracking.light_supermimic import (
        configure_light_representation,
        make_cfg,
    )
    from robots.humanoid.humanoid_light.config import ASSET_DIR

    task = "Humanoid-Light-Motion-Tracking"
    training = settings["training"]
    configure_muon(training.get("muon_dtype", "bfloat16"))
    for key in ("num_envs", "total_env_steps", "warmup_steps", "updates_per_step", "save_every", "log_every", "buffer_size"):
        if not isinstance(training[key], int) or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    device = training["device"]
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise ValueError("Light SuperMimic training requires a CUDA GPU runtime")
    torch.cuda.set_device(device)
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.set_float32_matmul_precision("high")
    random.seed(training["seed"]); np.random.seed(training["seed"]); torch.manual_seed(training["seed"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    packed, dataset = prepare(settings["motion"])
    write_json(output / "motion_data.json", dataset)
    config = dict(settings["runner"])
    policy = ActorCriticPolicyLearnerCfg(**config.pop("policy"))
    wm_cfg = WorldModelLearnerCfg(**config.pop("wm"))
    tasks_cfg = parse_tasks_cfg(config.pop("tasks", None))
    vel_allowlist = config.pop("velocity_allowlist", None)
    cfg = SuperMimicRunnerCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in config.items()},
                             tasks=tasks_cfg, wm=wm_cfg, policy=policy, buffer_size=training["buffer_size"])
    if training["buffer_size"] < max(wm_cfg.horizon, cfg.history_length):
        raise ValueError("Replay must cover the world-model window and policy history")
    if training["warmup_steps"] < max(wm_cfg.horizon, cfg.history_length):
        raise ValueError("Warmup must cover the world-model window and policy history")
    if cfg.sampling_mode not in ("uniform", "return", "adaptive", "start") or cfg.actor_obs not in ("rep_state", "proprio"):
        raise ValueError("Unsupported Light SuperMimic sampling_mode or actor_obs")
    environment_cfg = make_cfg(packed, settings["environment"], training["num_envs"], training["seed"],
                               sampling_mode=cfg.sampling_mode)

    class CheckedEnv(RslRlVecEnvWrapper):
        def step(self, actions):
            self.check(actions, "action")
            result = super().step(actions)
            fields = {"observation": result[0]["full_state"], "reward": result[1]}
            fields.update({name: getattr(self.unwrapped.sim.data, name) for name in ("qpos", "qvel", "qacc")})
            self.check_many(fields)
            return result

        def check(self, value, field):
            self.check_many({field: value})

        def check_many(self, fields):
            field = first_nonfinite(fields)
            if field is not None:
                value = fields[field]
                data = self.unwrapped.sim.data
                torch.save({"field": field, "value": value.cpu(), "qpos": data.qpos.cpu(),
                            "qvel": data.qvel.cpu(), "qacc": data.qacc.cpu(),
                            "action": self.unwrapped.action_manager.action.cpu(),
                            "env_origins": self.unwrapped.scene.env_origins.cpu(),
                            "environment": settings["environment"],
                            "motion_frame": self.unwrapped.command_manager.get_term("motion").time_steps.cpu()},
                           output / "nonfinite_physics.pt")
                raise FloatingPointError(f"Nonfinite {field}; captured nonfinite_physics.pt before any learner update")

    class LightDynamics(Dynamics):
        def __init__(self, old, representation, sim_ctx_dim, limits):
            super().__init__(representation, 26, sim_ctx_dim,
                             old.action_scale, old.action_offset, hidden_dims=cfg.dynamics_hidden_dims)
            self.load_state_dict(old.state_dict())
            self.register_buffer("joint_lower", limits[:, 0].clone())
            self.register_buffer("joint_upper", limits[:, 1].clone())

        def proc(self, action):
            limits = torch.stack((self.joint_lower, self.joint_upper), -1)
            return joint_range_targets(action, self.action_offset, limits)

    class LightRunner(SuperMimicRunner):
        def _init_common(self):
            super()._init_common()
            configure_light_representation(
                self._representation,
                self._sim_context,
                settings["environment"],
                self.motion_cmd.cfg.body_names,
                device,
            )
            if settings["environment"].get("action_mode", "joint_range") == "joint_range":
                limits = self.env.unwrapped.scene["robot"].data.joint_pos_limits[0]
                dynamics = LightDynamics(self._dynamics, self._representation, self._sim_context.output_dim, limits).to(device)
                self._wm.dynamics = self._dynamics = dynamics
                self.world_model_learner = WorldModelLearner(dynamics, self._representation, cfg.wm, device=device)

        def _init_policy(self):
            super()._init_policy()
            self._sim_vel_lo = None
            if vel_allowlist and self.cfg.slot_mix.get("velocity", 0.0) > 0.0:
                if self.cfg.history_length != 1:
                    raise ValueError("velocity_allowlist currently requires history_length == 1")
                sim_slots = self.policy_learner.actor.slot_ranges(self.env.num_envs)
                wm_slots = self.policy_learner.actor.slot_ranges(self.cfg.policy.batch_size)
                sim_vel_lo, sim_vel_hi = sim_slots[1]
                wm_vel_lo, wm_vel_hi = wm_slots[1]
                if sim_vel_hi > sim_vel_lo and wm_vel_hi > wm_vel_lo:
                    self._sim_vel_lo = sim_vel_lo
                    self._loco_frame_ids, self._loco_frame_probs = build_locomotion_frame_sampler(
                        packed, self.motion, vel_allowlist, device
                    )
                    print(
                        f"[INFO] Velocity locomotion allowlist active: {self._loco_frame_ids.numel()} frames "
                        f"(sim slots [{sim_vel_lo}, {sim_vel_hi}), wm slots [{wm_vel_lo}, {wm_vel_hi}))",
                        flush=True,
                    )
                    rb = self.replay_buffer
                    wm_batch = self.cfg.policy.batch_size

                    def stratified_sample_initial_states():
                        n = rb.num_transitions
                        if n < 1:
                            raise RuntimeError("sample_initial_states on empty buffer")
                        base = rb._step if n == rb.buffer_size else 0
                        mot_env = torch.randint(0, sim_vel_lo, (wm_vel_lo,), device=rb.device)
                        vel_env = torch.randint(sim_vel_lo, rb.num_envs, (wm_batch - wm_vel_lo,), device=rb.device)
                        env_idx = torch.cat((mot_env, vel_env), dim=0)
                        slot = (base + torch.randint(0, n, (wm_batch,), device=rb.device)) % rb.buffer_size
                        return rb._storage[env_idx, slot]

                    self.wm_env.sample_initial_states = stratified_sample_initial_states

        def _reset_velocity_envs(self, env_ids):
            if env_ids.numel() == 0:
                return
            pick = torch.multinomial(self._loco_frame_probs, env_ids.numel(), replacement=True)
            frames = self._loco_frame_ids[pick]
            base_env = self.env.unwrapped
            self.motion_cmd.set_time_steps(env_ids, frames)
            self.motion_cmd._write_reference_state_to_sim(
                env_ids,
                self.motion.root_pos_w[frames] + base_env.scene.env_origins[env_ids],
                self.motion.root_quat_w[frames],
                self.motion.root_lin_vel_w[frames],
                self.motion.root_ang_vel_w[frames],
                self.motion_cmd.joint_pos[env_ids],
                self.motion_cmd.joint_vel[env_ids],
            )
            base_env.scene.write_data_to_sim()
            base_env.sim.forward()
            self.motion_cmd.update_relative_body_poses()
            base_env.sim.sense()
            base_env.observation_manager.compute(update_history=True)

        def _step_and_store(self, raw_actions, curr_states, actor_td=None, extra_fields=None):
            next_obs, rewards, dones, extras = super()._step_and_store(
                raw_actions, curr_states, actor_td=actor_td, extra_fields=extra_fields
            )
            if self._sim_vel_lo is not None:
                vel_dones = dones[self._sim_vel_lo:].nonzero(as_tuple=False).squeeze(-1) + self._sim_vel_lo
                if vel_dones.numel() > 0:
                    self._reset_velocity_envs(vel_dones)
                    next_obs = self.env.get_observations()
            return next_obs, rewards, dones, extras

    env = CheckedEnv(ManagerBasedRlEnv(environment_cfg, device=device))
    try:
        runner = LightRunner(env, cfg, task_id=task, device=device)
        if not training.get("cuda_graphs", training["compile"]):
            # Upstream's optional graph wrapper is bypassed for eager smoke runs;
            # the exact same learner update functions and optimizers are used.
            wm_learner, policy_learner = runner.world_model_learner, runner.policy_learner
            wm_learner._captured_update = wm_learner._update_from
            policy_learner._captured_update = policy_learner._update_from
        write_json(output / "resolved.json", dict(settings=settings, runner=asdict(cfg)))
        control_hash = hashlib.sha256((ASSET_DIR / "control.json").read_bytes()).hexdigest()
        progress = dict(env_steps=0, wm_updates=0, policy_updates=0)
        learners = {"wm": runner.world_model_learner.optimizers,
                    "actor": runner.policy_learner.actor_optimizers,
                    "critic": runner.policy_learner.critic_optimizers}
        resume_changes = {}
        if resume:
            saved = torch.load(Path(resume) / "last.pt", map_location=device, weights_only=False)
            prior = saved["settings"]
            if saved["fingerprint"] != dataset["fingerprint"] or saved["control_hash"] != control_hash:
                raise ValueError("Resume robot control or screened motions differ")
            resume_changes = check_resume_settings(prior, settings)
            runner._wm.load_bundle(saved["world_model"])
            runner.policy_learner.actor.load_state_dict(saved["actor"])
            runner.policy_learner.critic.load_state_dict(saved["critic"])
            for key, optimizers in learners.items():
                for optimizer, state in zip(optimizers, saved["optimizers"][key], strict=True):
                    optimizer.load_state_dict(state)
            progress = saved["progress"]
            torch.set_rng_state(saved["torch_rng"].cpu())
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
            random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        elif training.get("init_from"):
            init_path = Path(training["init_from"])
            ckpt_file = init_path / "last.pt" if init_path.is_dir() else init_path
            saved = torch.load(ckpt_file, map_location=device, weights_only=False)
            if saved["control_hash"] != control_hash:
                raise ValueError("Pretrained checkpoint robot control hash differs")
            prior_env = saved["settings"]["environment"]
            for env_key in ("action_mode", "action_scale", "key_bodies_only"):
                if prior_env.get(env_key) != settings["environment"].get(env_key):
                    raise ValueError(f"Pretrained checkpoint environment.{env_key} differs")
            runner._wm.load_bundle(saved["world_model"])
            runner.policy_learner.actor.load_state_dict(saved["actor"])
            runner.policy_learner.critic.load_state_dict(saved["critic"])
        if training["total_env_steps"] <= progress["env_steps"]:
            raise ValueError("Training budget must exceed the checkpoint step")

        def checkpoint():
            destination = output / "checkpoints" / f"step_{progress['env_steps']}"
            if destination.exists():
                return
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".writing-", dir=destination.parent) as folder:
                directory = Path(folder) / "checkpoint"
                directory.mkdir()
                torch.save(dict(settings=settings, fingerprint=dataset["fingerprint"], control_hash=control_hash,
                                progress=dict(progress), world_model=runner._wm.bundle(),
                                actor=runner.policy_learner.actor.state_dict(), critic=runner.policy_learner.critic.state_dict(),
                                optimizers={key: [o.state_dict() for o in optimizers] for key, optimizers in learners.items()},
                                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device),
                                python_rng=random.getstate(), numpy_rng=np.random.get_state()), directory / "last.pt")
                write_json(directory / "run.json", dict(method="supermimic", task=task, settings=settings,
                           progress=progress, fingerprint=dataset["fingerprint"], control_hash=control_hash,
                           resume_resource_changes=resume_changes,
                           replay_saved=False, resume="Refill replay from fresh episodes; restore weights/optimizers/RNG"))
                directory.rename(destination)
            write_json(output / "latest.json", {"checkpoint": str(destination.relative_to(output)), **progress})

        env.reset()
        if runner._sim_vel_lo is not None:
            runner._reset_velocity_envs(torch.arange(runner._sim_vel_lo, env.num_envs, device=device))
        runner._seed_sim_history()
        last_save, collected = progress["env_steps"], 0
        initial_steps = progress["env_steps"]
        started = time.perf_counter()
        measure_after = training["warmup_steps"] + 8
        measured_start = None
        measured_initial = None
        sim_seconds = learn_seconds = 0.
        memory_free = memory_total = None
        while progress["env_steps"] < training["total_env_steps"]:
            step_started = time.perf_counter()
            runner.step()
            sim_finished = time.perf_counter()
            collected += 1
            progress["env_steps"] += env.num_envs
            metrics = {}
            if collected >= training["warmup_steps"]:
                for _ in range(training["updates_per_step"]):
                    batches = (runner.replay_buffer.sample(cfg.wm.horizon, cfg.wm.batch_size) for _ in range(cfg.wm.num_batches))
                    metrics.update(runner.world_model_learner.update(batches))
                    progress["wm_updates"] += 1
                    metrics.update(runner.policy_learner.update())
                    progress["policy_updates"] += 1
                    if first_nonfinite({key: value for key, value in metrics.items() if torch.is_tensor(value)}) is not None:
                        raise FloatingPointError("Nonfinite SuperMimic update metric")
            if measured_start is not None:
                sim_seconds += sim_finished - step_started
                learn_seconds += time.perf_counter() - sim_finished
            if collected == measure_after:
                torch.cuda.synchronize(device)
                measured_start = time.perf_counter()
                measured_initial = dict(progress)
                torch.cuda.reset_peak_memory_stats(device)
            if collected % training["log_every"] == 0 or progress["env_steps"] >= training["total_env_steps"]:
                row = dict(progress) | {key: float(value.detach()) if torch.is_tensor(value) else value for key, value in metrics.items()}
                if len(runner.logger.rewbuffer) > 0:
                    row["Train/mean_reward"] = sum(runner.logger.rewbuffer) / len(runner.logger.rewbuffer)
                    row["Train/mean_episode_length"] = sum(runner.logger.lenbuffer) / len(runner.logger.lenbuffer)
                runner.logger.ep_extras.clear()
                elapsed = time.perf_counter() - started
                memory_free, memory_total = torch.cuda.mem_get_info(device)
                row.update(elapsed_seconds=elapsed, env_steps_per_second=(progress["env_steps"] - initial_steps) / elapsed,
                           gpu_free_mib=memory_free / 1048576, gpu_total_mib=memory_total / 1048576,
                           gpu_peak_torch_mib=torch.cuda.max_memory_allocated(device) / 1048576)
                if measured_start is not None and progress["env_steps"] > measured_initial["env_steps"]:
                    row["steady_env_steps_per_second"] = (progress["env_steps"] - measured_initial["env_steps"]) / (time.perf_counter() - measured_start)
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
                write_json(output / "progress.json", row)
            if not benchmark and progress["env_steps"] - last_save >= training["save_every"]:
                checkpoint(); last_save = progress["env_steps"]
        if measured_start is not None and progress["env_steps"] > measured_initial["env_steps"]:
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - measured_start
            frames = progress["env_steps"] - measured_initial["env_steps"]
            updates = progress["policy_updates"] - measured_initial["policy_updates"]
            memory_free, memory_total = torch.cuda.mem_get_info(device)
            write_json(output / "performance.json", dict(num_envs=env.num_envs, compile=training["compile"],
                       cuda_graphs=training.get("cuda_graphs", training["compile"]),
                       muon_dtype=training.get("muon_dtype", "bfloat16"),
                       native_bf16_supported=torch.cuda.is_bf16_supported(including_emulation=False),
                       wm_batch_size=cfg.wm.batch_size, policy_batch_size=cfg.policy.batch_size,
                       updates_per_step=training["updates_per_step"], measured_transitions=frames,
                       measured_seconds=seconds, env_steps_per_second=frames / seconds,
                       policy_samples_per_second=updates * cfg.policy.batch_size / seconds,
                       simulator_seconds=sim_seconds, learner_seconds=learn_seconds,
                       gpu_free_mib=memory_free / 1048576, gpu_total_mib=memory_total / 1048576,
                       gpu_peak_torch_mib=torch.cuda.max_memory_allocated(device) / 1048576,
                       cold_start_and_warmup_seconds=measured_start - started))
        if not benchmark:
            checkpoint()
        write_json(output / "complete.json", progress)
    finally:
        env.close()


if __name__ == "__main__":
    main()
