"""Evaluate a saved motion-prior policy against held-out CMU/LAFAN references.

Example: python -m humanoid_rl.algorithm.bfm_zero.evaluate --checkpoint DIR
Reference initialization is explicit; this measures tracking from a supplied
state, not autonomous takeoff, recovery, or dynamic feasibility of the dataset.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def evaluate(checkpoint, *, split="validation", clips_per_source=2, steps=250, device="cuda:0", output=None,
             motion_data=None, motion_manifest=None, motion_cache=None):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.lab_api.math import quat_apply
    from humanoid_rl.algorithm.bfm_zero.config import resolve_agent_config
    from humanoid_rl.algorithm.bfm_zero.dataset import MotionDataset
    from humanoid_rl.algorithm.bfm_zero.upstream import ROOT, enable_bfm_zero
    from humanoid_rl.env.bfm_adapter import BfmMjlabAdapter
    from humanoid_rl.tasks.motion_prior.env_cfg import MotionPriorEnvConfig, make_env_cfg
    from humanoid_retarget.datasets.paths import relocate_motion

    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint/"run.json").read_text())
    if metadata.get("algorithm") != "bfm_zero":
        raise ValueError("Expected a BFM-Zero checkpoint")
    values = metadata["train_config"]
    profile = resolve_agent_config(values, metadata["task"])
    settings = dict(values["motion"], split=split, limit=None, files=None, max_frames=None,
                    source_weights=None, expert_buffer_device="cpu")
    settings = relocate_motion(settings, root=motion_data, manifest=motion_manifest, cache=motion_cache)
    manifest_path = Path(settings["manifest"])
    manifest_path = manifest_path if manifest_path.is_absolute() else ROOT/manifest_path
    records = json.loads(manifest_path.read_text())["splits"][split]
    chosen = []
    for source in sorted({record["source"] for record in records}):
        files = sorted({record["path"] for record in records if record["source"] == source})
        chosen.extend(files[:clips_per_source])
    if not chosen:
        raise ValueError(f"No expert motions in {split!r} split")
    settings["files"] = chosen
    env_settings = MotionPriorEnvConfig(**(values["environment"] | {"lie_down_probability": 0.,
                                        "observation_noise": False, "domain_randomization": False}))
    dataset = MotionDataset(settings, env_settings, metadata["robot_control"], values["model"]["sequence_length"],
                            expert_device=device, agent_cfg=profile)
    enable_bfm_zero()
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModel
    model = FBcprAuxModel.load(str(checkpoint/"model"), device=torch.device(device).type, strict=False)
    cfg = make_env_cfg(env_settings, metadata["robot_control"], agent_cfg=profile)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device=device)
    results = []
    try:
        adapter = BfmMjlabAdapter(env, dataset=None)
        robot = env.scene["robot"]
        dimensions = {key: value.shape[-1] for key, value in adapter.observations.items()}
        if dimensions != metadata["observation_dims"] or adapter.action_dim != metadata["action_dim"]:
            raise ValueError("Evaluation observation/action schema differs from checkpoint")
        # One longest accepted chunk per selected source file, without joining gaps.
        selected = []
        for path in chosen:
            candidates = [i for i, clip in enumerate(dataset.clips) if clip.get("path") == path]
            if candidates:
                selected.append(max(candidates, key=lambda index: dataset.clips[index]["frames"]))
        with torch.no_grad():
            for clip_id in selected:
                start = int(dataset.clip_starts[clip_id])
                length = min(steps, int(dataset.clip_lengths[clip_id])-1)
                q = dataset.qpos[start:start+length+1].to(device)
                qv = dataset.qvel[start:start+length+1].to(device)
                obs = {key: value[start+1:start+length+1].to(device)
                       for key, value in dataset.expert_slicer.storage["observation"].items()}
                z = model.tracking_inference(obs)
                env.reset()
                root = torch.cat((q[:1, :3]+env.scene.env_origins, q[:1, 3:7], qv[:1, :3],
                                  quat_apply(q[:1, 3:7], qv[:1, 3:6])), -1)
                robot.write_root_state_to_sim(root)
                robot.write_joint_state_to_sim(q[:1, 7:], qv[:1, 6:])
                env.scene.write_data_to_sim(); env.sim.forward()
                adapter.history.reset(slice(None)); adapter.last_action.zero_(); adapter.step_count.zero_()
                adapter.observations = adapter.observe()
                joint_errors, heights, gravity_errors, saturated = [], [], [], []
                limits = torch.tensor([metadata["robot_control"]["effort_limit"][name]
                                       for name in robot.actuator_names], device=device)
                for frame in range(length):
                    action = model.act(adapter.observations, z[frame:frame+1], mean=True)
                    transition, _ = adapter.step(action)
                    if bool(transition["next"]["terminated"].any() | transition["next"]["truncated"].any()):
                        raise RuntimeError("Evaluation episode ended before the reference; shorten --steps")
                    joint_errors.append((robot.data.joint_pos[0]-q[frame+1, 7:]).square().mean().item())
                    heights.append((robot.data.root_link_pos_w[0, 2]-env.scene.env_origins[0, 2]-q[frame+1, 2]).square().item())
                    joint_count = adapter.action_dim
                    actual_g = transition["next"]["observation"]["state"][0, 2*joint_count:2*joint_count+3]
                    reference_g = obs["state"][frame, 2*joint_count:2*joint_count+3]
                    gravity_errors.append(torch.acos((actual_g*reference_g).sum().clamp(-1, 1)).item())
                    saturated.append((robot.data.actuator_force[0].abs() >= limits*.99).float().mean().item())
                row = dict(source=dataset.clips[clip_id].get("dataset"), motion=dataset.clips[clip_id]["source"],
                           source_start_frame=dataset.clips[clip_id]["start_frame"], steps=length,
                           joint_rmse_rad=float(np.sqrt(np.mean(joint_errors))),
                           root_height_rmse_m=float(np.sqrt(np.mean(heights))),
                           mean_root_tilt_error_deg=float(np.rad2deg(np.mean(gravity_errors))),
                           mean_torque_saturated_fraction=float(np.mean(saturated)))
                results.append(row)
                print(json.dumps(row), flush=True)
    finally:
        env.close()
    result = dict(checkpoint=str(checkpoint.resolve()), split=split, reference_initialized=True,
                  env_steps=metadata["env_steps"], motions=results)
    output = Path(output) if output else checkpoint/"tracking_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--split", choices=("train", "validation"), default="validation")
    cli.add_argument("--clips-per-source", type=int, default=2)
    cli.add_argument("--steps", type=int, default=250)
    cli.add_argument("--device", default="cuda:0")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--motion-data", type=Path)
    cli.add_argument("--motion-manifest", type=Path)
    cli.add_argument("--motion-cache", type=Path)
    args = cli.parse_args()
    if min(args.clips_per_source, args.steps) < 1:
        cli.error("clips-per-source and steps must be positive")
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_num_threads(2)
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
