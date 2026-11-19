"""Interactive Viser (and headless validation) playback for Humanoid Light SuperMimic checkpoints.

Usage (from repo root):
  .venv_supermimic_light/bin/python src/humanoid_rl/sim2sim/play_light_supermimic_viser.py
  .venv_supermimic_light/bin/python src/humanoid_rl/sim2sim/play_light_supermimic_viser.py --port 8080
  .venv_supermimic_light/bin/python src/humanoid_rl/sim2sim/play_light_supermimic_viser.py --headless --steps 100
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "runs/light_supermimic_t4_long/free_t4_fp16_20261117/training/checkpoints/step_2050048"
)


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint directory (containing last.pt), training directory (containing latest.json), or last.pt file.",
    )
    cli.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional override for screening motion manifest JSON.",
    )
    cli.add_argument(
        "--viewer",
        choices=("viser", "native"),
        default="viser",
        help="Interactive viewer backend (default: viser).",
    )
    cli.add_argument("--host", default="0.0.0.0", help="Viser server bind host.")
    cli.add_argument("--port", type=int, default=8080, help="Viser server bind port.")
    cli.add_argument("--device", default="cuda:0", help="Torch/MuJoCo device.")
    cli.add_argument("--seed", type=int, default=42, help="Environment RNG seed.")
    cli.add_argument(
        "--initial-task",
        choices=("motion", "velocity"),
        default="motion",
        help="Initial task slot selected in the viewer.",
    )
    cli.add_argument(
        "--num-steps-rollout",
        type=int,
        default=1_000_000,
        help="Closed-loop world-model horizon before re-syncing to ground truth.",
    )
    cli.add_argument(
        "--episode-length",
        type=float,
        default=1e9,
        help="Episode timeout in seconds during playback (default: 1e9 so clips run until motion_end).",
    )
    cli.add_argument(
        "--no-ref-ghost",
        action="store_true",
        help="Disable the green reference-motion ghost overlay.",
    )
    cli.add_argument(
        "--paused",
        action="store_true",
        help="Open viewer paused on the initial frame.",
    )
    cli.add_argument(
        "--headless",
        action="store_true",
        help="Run deterministic rollout steps without starting a viewer (requires --steps).",
    )
    cli.add_argument(
        "--playlist",
        action="store_true",
        help="Enable sequential auto-advance across clips in manifest order.",
    )
    cli.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of control steps to execute (required with --headless; optional cap with viewer).",
    )
    return cli.parse_args()


def resolve_checkpoint_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if (path / "last.pt").is_file():
        return path / "last.pt"
    if (path / "latest.json").is_file():
        latest = json.loads((path / "latest.json").read_text())
        candidate = path / latest["checkpoint"] / "last.pt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find last.pt or latest.json under checkpoint path: {path}"
    )


def resolve_local_motion_settings(
    motion_settings: dict, manifest_override: Path | None
) -> dict:
    motion = deepcopy(motion_settings)
    orig_root = Path(motion["root"])
    orig_manifest = Path(motion["manifest"])

    if not orig_root.exists():
        motion["root"] = str(REPO_ROOT)

    if manifest_override is not None:
        motion["manifest"] = str(manifest_override.expanduser().resolve())
    elif not orig_manifest.exists():
        try:
            rel_manifest = orig_manifest.relative_to(orig_root)
        except ValueError:
            rel_manifest = Path("src/humanoid_retarget/configs/motions/light_supermimic_t4_trial.json")
        local_manifest = REPO_ROOT / rel_manifest
        if not local_manifest.is_file():
            alt = REPO_ROOT / "src/humanoid_retarget" / rel_manifest
            if alt.is_file():
                local_manifest = alt
        if not local_manifest.is_file():
            raise FileNotFoundError(
                f"Local manifest not found at {local_manifest} (from {orig_manifest})"
            )
        motion["manifest"] = str(local_manifest)

    cache_dir = Path(motion["cache_dir"])
    if not cache_dir.exists() and str(cache_dir).startswith("/content/"):
        motion["cache_dir"] = str(REPO_ROOT / "data/cache/supermimic/humanoid_light")

    return motion


def main() -> None:
    args = parse_args()
    if args.headless and (args.steps is None or args.steps < 1):
        raise SystemExit("--headless requires --steps >= 1")

    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    if args.headless or args.viewer == "viser":
        os.environ.setdefault("MUJOCO_GL", "egl")

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.viewer import VerbosityLevel
    from supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg
    from supermimic.learners.wm_learner import WorldModelLearnerCfg
    from supermimic.models import Dynamics
    from supermimic.representations import extract_actor_goal
    from supermimic.runners.supermimic_runner import (
        SuperMimicRunnerCfg,
        build_actor,
        build_world_model,
    )
    from supermimic.scripts.viewers import (
        SuperMimicNativeViewer,
        SuperMimicViserViewer,
        create_wm_vis_model,
    )
    from supermimic.tasks.task_spec import build_actor_obs_td

    from humanoid_rl.algorithm.supermimic.motions import prepare
    from humanoid_rl.env.actions import joint_range_targets
    from humanoid_rl.tasks.motion_tracking.light_supermimic import (
        configure_light_representation,
        make_cfg,
    )
    from robots.humanoid.humanoid_light.config import ASSET_DIR

    configure_torch_backends()
    device = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    ckpt_file = resolve_checkpoint_file(args.checkpoint)
    ckpt_sha256 = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()
    print(f"[INFO] Loading checkpoint: {ckpt_file}", flush=True)
    print(f"[INFO] Checkpoint SHA256: {ckpt_sha256}", flush=True)

    saved = torch.load(ckpt_file, map_location=device, weights_only=False)
    settings = saved["settings"]
    progress = saved.get("progress", {})
    print(
        f"[INFO] Checkpoint progress: env_steps={progress.get('env_steps')} "
        f"wm_updates={progress.get('wm_updates')} policy_updates={progress.get('policy_updates')}",
        flush=True,
    )

    control_hash = hashlib.sha256((ASSET_DIR / "control.json").read_bytes()).hexdigest()
    if saved.get("control_hash") and saved["control_hash"] != control_hash:
        raise ValueError(
            f"Robot control.json hash mismatch: checkpoint={saved['control_hash']} local={control_hash}"
        )

    motion_settings = resolve_local_motion_settings(settings["motion"], args.manifest)
    packed, dataset = prepare(motion_settings)
    if args.manifest is None and saved.get("fingerprint") != dataset["fingerprint"]:
        raise ValueError(
            f"Packed motion fingerprint mismatch: checkpoint={saved.get('fingerprint')} local={dataset['fingerprint']}"
        )
    motion_names = json.loads((packed / "motion_names.json").read_text())
    print(
        f"[INFO] Packed motions ready: {packed} ({dataset['clips']} clips, {dataset['frames']} frames)",
        flush=True,
    )
    raw_manifest = json.loads(Path(motion_settings["manifest"]).read_text())
    raw_train_recs = raw_manifest.get("splits", {}).get("train", [])
    has_showcase_labels = any("showcase_label" in r for r in raw_train_recs)
    name_to_packed_indices: dict[str, list[int]] = {}
    for p_idx, m_name in enumerate(motion_names):
        name_to_packed_indices.setdefault(m_name, []).append(p_idx)

    playlist_items: list[tuple[str, int]] = []
    if has_showcase_labels:
        chunk_frames = int(motion_settings.get("chunk_frames", 501))
        for rec in raw_train_recs:
            stem = Path(rec["path"]).stem
            start, stop = int(rec["start_frame"]), int(rec["stop_frame"])
            label = rec.get("showcase_label", "")
            for first in range(start, stop, chunk_frames):
                last = min(first + chunk_frames, stop)
                if last - first >= 2:
                    key = f"{rec['source']}/{stem}/{first}-{last}"
                    if key in name_to_packed_indices and name_to_packed_indices[key]:
                        p_idx = name_to_packed_indices[key].pop(0)
                        pos = len(playlist_items)
                        title = f"{pos:02d}: {label} [{key}]" if label else f"{pos:02d}: {key}"
                        playlist_items.append((title, p_idx))
    if not playlist_items:
        playlist_items = [(f"{i}: {n}", i) for i, n in enumerate(motion_names)]

    auto_playlist_default = bool(args.playlist or has_showcase_labels)

    from humanoid_rl.algorithm.supermimic.train import parse_tasks_cfg

    runner_dict = deepcopy(settings["runner"])
    policy_cfg = ActorCriticPolicyLearnerCfg(**runner_dict.pop("policy"))
    wm_cfg = WorldModelLearnerCfg(**runner_dict.pop("wm"))
    tasks_cfg = parse_tasks_cfg(runner_dict.pop("tasks", None))
    runner_dict.pop("velocity_allowlist", None)
    runner_cfg = SuperMimicRunnerCfg(
        **{
            k: tuple(v) if isinstance(v, list) else v
            for k, v in runner_dict.items()
        },
        tasks=tasks_cfg,
        wm=wm_cfg,
        policy=policy_cfg,
        buffer_size=settings["training"]["buffer_size"],
    )

    env_cfg = make_cfg(
        packed, settings["environment"], num_envs=1, seed=args.seed, play=True
    )
    env_cfg.episode_length_s = args.episode_length
    env_cfg.commands["motion"].debug_vis = not args.no_ref_ghost
    env_cfg.commands["motion"].sampling_mode = "start"

    class LightDynamics(Dynamics):
        def __init__(self, old, representation, sim_ctx_dim, limits):
            super().__init__(
                representation,
                26,
                sim_ctx_dim,
                old.action_scale,
                old.action_offset,
                hidden_dims=runner_cfg.dynamics_hidden_dims,
            )
            self.load_state_dict(old.state_dict())
            self.register_buffer("joint_lower", limits[:, 0].clone())
            self.register_buffer("joint_upper", limits[:, 1].clone())

        def proc(self, action):
            limits = torch.stack((self.joint_lower, self.joint_upper), -1)
            return joint_range_targets(action, self.action_offset, limits)

    from humanoid_rl.sim2sim.video_retarget_viser_bridge import VideoDanceStudioPanel

    class LightSuperMimicViserViewer(SuperMimicViserViewer):
        def __init__(
            self,
            *v_args,
            ordered_playlist: list[tuple[str, int]],
            auto_advance_playlist: bool = False,
            start_from_clip_begin: bool = True,
            **v_kwargs,
        ):
            self._ordered_playlist = ordered_playlist
            self._playlist_pos = 0
            self._auto_advance_playlist = auto_advance_playlist
            self._start_from_clip_begin = start_from_clip_begin
            self._suppress_dd_cb = False
            self._clip_dropdown = None
            self._video_studio: VideoDanceStudioPanel | None = None
            super().__init__(*v_args, **v_kwargs)
            self._selected_clip_idx = self._ordered_playlist[0][1]

        def setup(self) -> None:
            super().setup()
            self._video_studio = VideoDanceStudioPanel(self, self._server)
            if self._supports_clip_selection():
                with self._sim_lock:
                    self._apply_selected_clip()

        def _setup_motion_clip_widget(self) -> None:
            if not self._supports_clip_selection():
                return
            options = [item[0] for item in self._ordered_playlist]
            self._clip_dropdown = self._server.gui.add_dropdown(
                "Showcase Clip",
                options=options,
                initial_value=options[0],
                order=0,
            )
            prev_btn = self._server.gui.add_button("Prev Clip", order=0.1)
            next_btn = self._server.gui.add_button("Next Clip", order=0.2)
            auto_cb = self._server.gui.add_checkbox(
                "Auto-Advance Playlist",
                initial_value=self._auto_advance_playlist,
                order=0.3,
            )
            start_cb = self._server.gui.add_checkbox(
                "Start Clip at Frame 0",
                initial_value=self._start_from_clip_begin,
                order=0.4,
            )

            @self._clip_dropdown.on_update
            def _(event) -> None:
                if self._suppress_dd_cb:
                    return
                titles = [it[0] for it in self._ordered_playlist]
                if event.target.value in titles:
                    self._playlist_pos = titles.index(event.target.value)
                    self._selected_clip_idx = self._ordered_playlist[self._playlist_pos][1]
                    self.request_reset()

            def step_playlist(delta: int) -> None:
                self._playlist_pos = (self._playlist_pos + delta) % len(self._ordered_playlist)
                self._selected_clip_idx = self._ordered_playlist[self._playlist_pos][1]
                if self._clip_dropdown is not None:
                    self._suppress_dd_cb = True
                    self._clip_dropdown.value = self._ordered_playlist[self._playlist_pos][0]
                    self._suppress_dd_cb = False
                self.request_reset()

            @prev_btn.on_click
            def _(_) -> None:
                step_playlist(-1)

            @next_btn.on_click
            def _(_) -> None:
                step_playlist(1)

            @auto_cb.on_update
            def _(event) -> None:
                self._auto_advance_playlist = bool(event.target.value)

            @start_cb.on_update
            def _(event) -> None:
                self._start_from_clip_begin = bool(event.target.value)

        def step_and_predict(self, *, force: bool = False) -> bool:
            was_running = self.step_in_rollout > 0
            ok = super().step_and_predict(force=force)
            if ok and was_running and self.step_in_rollout == 0 and self._supports_clip_selection():
                if self._auto_advance_playlist:
                    self._playlist_pos = (self._playlist_pos + 1) % len(self._ordered_playlist)
                    self._selected_clip_idx = self._ordered_playlist[self._playlist_pos][1]
                    if self._clip_dropdown is not None:
                        self._suppress_dd_cb = True
                        self._clip_dropdown.value = self._ordered_playlist[self._playlist_pos][0]
                        self._suppress_dd_cb = False
                with self._sim_lock:
                    self._apply_selected_clip()
            elif ok and self._video_studio is not None:
                self._video_studio.update_synced_video_frame()
            return ok

        def _apply_selected_clip(self) -> None:
            motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
            motion = motion_cmd.motion
            idx = self._selected_clip_idx
            clip_start = 0 if idx == 0 else int(motion.motion_ends[idx - 1].item())
            clip_end = int(motion.motion_ends[idx].item())
            if self._start_from_clip_begin:
                frame = clip_start
            else:
                frame = int(torch.randint(clip_start, clip_end, (1,)).item())
            motion_cmd.reset_to_frame(torch.tensor([0], device=self.env.device), frame)
            base_env = self.env.unwrapped
            base_env.scene.write_data_to_sim()
            base_env.sim.forward()
            motion_cmd.update_relative_body_poses()
            base_env.sim.sense()
            base_env.observation_manager.compute(update_history=True)
            if self._video_studio is not None:
                self._video_studio.update_synced_video_frame()

    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))
    try:
        wm = build_world_model(runner_cfg, env, device)
        configure_light_representation(
            wm.representation,
            wm.sim_context,
            settings["environment"],
            env.unwrapped.command_manager.get_term("motion").cfg.body_names,
            device,
        )
        if settings["environment"].get("action_mode", "joint_range") == "joint_range":
            limits = env.unwrapped.scene["robot"].data.joint_pos_limits[0]
            wm.dynamics = LightDynamics(
                wm.dynamics, wm.representation, wm.sim_context.output_dim, limits
            ).to(device)
        wm.load_bundle(saved["world_model"])
        wm.dynamics.eval()

        actor = build_actor(runner_cfg, wm, device)
        actor.load_state_dict(saved["actor"])
        actor.eval()

        if args.headless:
            actor.set_slot_mix({args.initial_task: 1.0})
            obs, _ = env.reset()
            sim_ctx = wm.sim_context.read(env.unwrapped)
            curr_wm_aux = wm.representation.extract_aux_state(obs["full_state"])
            curr_pred_rep = wm.representation.represent(obs["full_state"])
            total_reward = 0.0
            dones_count = 0
            with torch.no_grad():
                for step_idx in range(args.steps):
                    obs = env.get_observations()
                    gt_states, goals = obs["full_state"], obs["goal"]
                    gt_rep = wm.representation.represent(gt_states)
                    sensor = wm.representation.compute_actor_sensor(gt_rep, sim_ctx)
                    actor_goal = extract_actor_goal(goals)
                    actor_obs = build_actor_obs_td(
                        {args.initial_task: actor_goal}, sensor.unsqueeze(1)
                    )
                    actions = actor(actor_obs)
                    if not torch.isfinite(actions).all():
                        raise FloatingPointError(f"Nonfinite actor action at step {step_idx}")
                    next_obs, rewards, dones, _ = env.step(actions)
                    if not torch.isfinite(next_obs["full_state"]).all():
                        raise FloatingPointError(f"Nonfinite GT state at step {step_idx}")
                    next_aux, next_rep = wm.dynamics(
                        curr_pred_rep, actions, curr_wm_aux, sim_ctx
                    )
                    if not (torch.isfinite(next_aux).all() and torch.isfinite(next_rep).all()):
                        raise FloatingPointError(f"Nonfinite WM prediction at step {step_idx}")
                    curr_wm_aux, curr_pred_rep = next_aux, next_rep
                    total_reward += float(rewards[0].item())
                    if dones.any():
                        dones_count += 1
                        curr_wm_aux = wm.representation.extract_aux_state(
                            next_obs["full_state"]
                        )
                        curr_pred_rep = wm.representation.represent(
                            next_obs["full_state"]
                        )
            print(
                f"[PASS] Headless playback completed: steps={args.steps} "
                f"mean_step_reward={total_reward / args.steps:.4f} resets={dones_count}",
                flush=True,
            )
            return

        wm_vis_model = create_wm_vis_model(env.unwrapped.sim.mj_model)
        if args.viewer == "native":
            viewer = SuperMimicNativeViewer(
                env=env,
                actor=actor,
                dynamics=wm.dynamics,
                representation=wm.representation,
                wm_vis_model=wm_vis_model,
                num_steps_rollout=args.num_steps_rollout,
                sim_context=wm.sim_context,
                task_specs=actor.task_specs,
                initial_task=args.initial_task,
                verbosity=VerbosityLevel.INFO,
            )
        else:
            import viser

            server = viser.ViserServer(
                host=args.host, port=args.port, label="supermimic-light"
            )
            print(
                f"[INFO] Viser server listening at http://localhost:{server.get_port()}",
                flush=True,
            )
            viewer = LightSuperMimicViserViewer(
                env=env,
                actor=actor,
                dynamics=wm.dynamics,
                representation=wm.representation,
                wm_vis_model=wm_vis_model,
                num_steps_rollout=args.num_steps_rollout,
                motion_names=motion_names,
                ordered_playlist=playlist_items,
                auto_advance_playlist=auto_playlist_default,
                sim_context=wm.sim_context,
                task_specs=actor.task_specs,
                initial_task=args.initial_task,
                viser_server=server,
                verbosity=VerbosityLevel.INFO,
            )
        if args.paused:
            viewer.pause()
        viewer.run(num_steps=args.steps)
        if viewer.get_status().last_error:
            raise RuntimeError(viewer.get_status().last_error)
    finally:
        env.close()


if __name__ == "__main__":
    main()
