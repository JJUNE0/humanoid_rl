"""Capture first invalid Light physics/observations without running a learner.

Reconstruct the training reset sampler and clips from the screened manifest.
This is a diagnostic reproduction, not a replay of the lost training policy.
Run in a separate resource-limited systemd service on a shared GPU.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

import mujoco
import numpy as np
import torch

from humanoid_rl.algorithm.bfm_zero.config import resolve_agent_config
from humanoid_rl.algorithm.bfm_zero.dataset import MotionDataset
from humanoid_rl.algorithm.configuration import load_train_config
from humanoid_rl.env.bfm_adapter import BfmMjlabAdapter
from humanoid_rl.tasks.motion_prior.env_cfg import MotionPriorEnvConfig, make_env_cfg
from robots.humanoid.humanoid_light.mjlab import load_control


class ResetDataset:
    """Same clip ordering/weights/frame draws, without unused expert features."""

    sample_states = MotionDataset.sample_states

    def __init__(self, settings, split="train"):
        self.robot = "humanoid_light"
        self.manifest = json.loads((ROOT/settings["manifest"]).read_text())
        by_path = {}
        for record in self.manifest["splits"][split]:
            by_path.setdefault(record["path"], []).append(record)
        positions, velocities, self.clips = [], [], []
        for path in sorted(by_path):
            with np.load(ROOT/path, allow_pickle=False) as source:
                qpos, qvel = source["qpos"], source["qvel"]
                assert float(source["frequency"]) == 50.
            for record in by_path[path]:
                for start in range(record["start_frame"], record["stop_frame"]-1, 500):
                    stop = min(start+501, record["stop_frame"])
                    if stop-start <= 8:
                        continue
                    positions.append(qpos[start:stop])
                    velocities.append(qvel[start:stop])
                    self.clips.append(dict(path=path, dataset=record["source"],
                                           start_frame=start, frames=stop-start))
        self.qpos = torch.from_numpy(np.concatenate(positions))
        self.qvel = torch.from_numpy(np.concatenate(velocities))
        self.clip_starts = torch.tensor(np.cumsum([0]+[r["frames"] for r in self.clips[:-1]]))
        self.clip_lengths = torch.tensor([r["frames"] for r in self.clips])
        counts = {s: sum(r["dataset"] == s for r in self.clips) for s in ("cmu", "lafan")}
        self.clip_probabilities = torch.tensor([settings["source_weights"][r["dataset"]]/counts[r["dataset"]]
                                                for r in self.clips])
        self.clip_probabilities /= self.clip_probabilities.sum()
        self.last_ids = None

    def tracked_sample(self, count, device):
        # Reproduce sample_states' draws, recording their provenance.
        clips = torch.multinomial(self.clip_probabilities, count, replacement=True)
        offsets = (torch.rand(count)*(self.clip_lengths[clips]-1)).long()
        self.last_ids = (clips, offsets)
        indices = self.clip_starts[clips]+offsets
        return self.qpos[indices].to(device).clone(), self.qvel[indices].to(device).clone()


def geometry(model, pose, velocity):
    if not np.isfinite(pose).all() or not np.isfinite(velocity).all():
        return None
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = pose, velocity
    mujoco.mj_forward(model, data)
    lowest = (float("inf"), None)
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start+count]
        height = float((vertices@data.geom_xmat[gid].reshape(3, 3)[2]+data.geom_xpos[gid, 2]).min())
        if height < lowest[0]:
            lowest = (height, model.body(model.geom_bodyid[gid]).name)
    contacts = sorted([(float(c.dist), model.geom(int(c.geom1)).name,
                        model.geom(int(c.geom2)).name) for c in data.contact], key=lambda c:c[0])
    return dict(lowest_mesh_height_m=lowest[0], lowest_body=lowest[1],
                minimum_contact_distance_m=contacts[0][0] if contacts else None,
                contacts=contacts[:8])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--steps", type=int, default=200)
    cli.add_argument("--num-envs", type=int, default=1024)
    cli.add_argument("--seed", type=int, default=4728)
    cli.add_argument("--lying-probability", type=float, default=.3)
    cli.add_argument("--mode", choices=("random", "zero"), default="random")
    cli.add_argument("--physics-dt", type=float, default=.005)
    args = cli.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.10, 0)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    values = load_train_config("bfm_zero", task="Humanoid-Light-Motion-Prior")
    profile = resolve_agent_config(values, "Humanoid-Light-Motion-Prior")
    settings = MotionPriorEnvConfig(**(values["environment"] | dict(
        lie_down_probability=args.lying_probability, physics_dt=args.physics_dt,
        decimation=round(.02/args.physics_dt))))
    assert np.isclose(settings.physics_dt*settings.decimation, .02)
    dataset = ResetDataset(values["motion"])
    print(json.dumps(dict(reset_frames=len(dataset.qpos), clips=len(dataset.clips))), flush=True)
    dataset.sample_states = dataset.tracked_sample
    cfg = make_env_cfg(settings, load_control(), agent_cfg=profile)
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    reset = cfg.events["reference_reset"].func
    traces = {}

    def tracked_reset(env, ids):
        reset(env, ids)
        if not hasattr(env, "motion_dataset"):
            return
        ids = torch.arange(env.num_envs, device=env.device) if ids is None else ids
        clips, offsets = dataset.last_ids
        q = env.sim.data.qpos[ids].detach().cpu().numpy()
        v = env.sim.data.qvel[ids].detach().cpu().numpy()
        for i, eid in enumerate(ids.tolist()):
            record = dataset.clips[int(clips[i])]
            source_index = int(dataset.clip_starts[clips[i]]+offsets[i])
            traces[eid] = dict(path=record["path"], source=record["dataset"],
                               source_frame=record["start_frame"]+int(offsets[i]),
                               reference_qpos=dataset.qpos[source_index].numpy().copy(),
                               reference_qvel=dataset.qvel[source_index].numpy().copy(),
                               reset_qpos=q[i].copy(), reset_qvel=v[i].copy())

    cfg.events["reference_reset"].func = tracked_reset
    from mjlab.envs import ManagerBasedRlEnv
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    recent = deque(maxlen=16)
    actions = []
    physics_step = 0
    vector_step = 0
    failure = None
    report = dict(started=datetime.now().isoformat(), configuration=vars(args) | {"output":str(out)},
                  note="Physics-only reproduction; original trained weights/actions were not saved.")
    try:
        adapter = BfmMjlabAdapter(env, dataset)
        np.savez_compressed(out/"initial_states.npz", qpos=env.sim.data.qpos.cpu().numpy(),
                            qvel=env.sim.data.qvel.cpu().numpy())
        step = env.sim.step

        def checked_step():
            nonlocal physics_step, failure
            fields = ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "qfrc_actuator", "qfrc_constraint")
            state = {k:getattr(env.sim.data, k).detach().cpu().numpy().copy() for k in fields}
            state["physics_step"] = physics_step
            recent.append(state)
            step()
            physics_step += 1
            bad = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
            invalid = {}
            for k in fields:
                tensor = getattr(env.sim.data, k)
                mask = ~torch.isfinite(tensor).all(-1)
                if mask.any():
                    invalid[k] = mask.nonzero().flatten().cpu().tolist()
                    bad |= mask
            if bad.any():
                failure = dict(kind="raw_physics", fields=invalid, env_ids=bad.nonzero().flatten().cpu().tolist())
                raise FloatingPointError("First invalid physics substep captured")

        env.sim.step = checked_step
        for vector_step in range(args.steps):
            action = (torch.empty(args.num_envs, 26, device=env.device).uniform_(-1, 1)
                      if args.mode == "random" else torch.zeros(args.num_envs, 26, device=env.device))
            actions.append(action.cpu().numpy().copy())
            try:
                adapter.step(action)
            except FloatingPointError:
                if failure is None:
                    bad = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
                    invalid = {}
                    for k, tensor in adapter.observe().items():
                        mask = ~torch.isfinite(tensor).all(-1)
                        if mask.any():
                            invalid[k] = mask.nonzero().flatten().cpu().tolist(); bad |= mask
                    failure = dict(kind="observations", fields=invalid,
                                   env_ids=bad.nonzero().flatten().cpu().tolist())
                break
            if vector_step % 20 == 0:
                print(json.dumps(dict(vector_step=vector_step, max_joint_speed=float(
                    env.scene["robot"].data.joint_vel.abs().max()), physics_step=physics_step)), flush=True)
        report.update(failure=failure, completed_vector_steps=vector_step+(failure is None), physics_step=physics_step)
        if failure:
            env_ids = failure["env_ids"]
            arrays = {f"pre_{k}":np.stack([state[k][env_ids] for state in recent])
                      for k in recent[-1] if k != "physics_step"}
            arrays.update(action=np.stack(actions)[:, env_ids], env_ids=np.array(env_ids),
                          history_physics_steps=np.array([s["physics_step"] for s in recent]))
            for k in ("qpos", "qvel", "qacc", "ctrl", "qfrc_actuator", "qfrc_constraint"):
                arrays[f"post_{k}"] = getattr(env.sim.data,k)[env_ids].cpu().numpy()
            selected = []
            for eid in env_ids:
                record = traces[eid]
                item = {k:v for k,v in record.items() if not isinstance(v,np.ndarray)}
                item["env_id"] = eid
                for k in ("reference_qpos", "reference_qvel", "reset_qpos", "reset_qvel"):
                    arrays[f"env_{eid}_{k}"] = record[k]
                item["reset_geometry"] = geometry(env.sim.mj_model, record["reset_qpos"], record["reset_qvel"])
                item["pre_failure_max_qvel"] = float(np.abs(recent[-1]["qvel"][eid]).max())
                item["pre_failure_max_qacc"] = float(np.abs(recent[-1]["qacc"][eid]).max())
                selected.append(item)
            report["affected_resets"] = selected
            np.savez_compressed(out/"failure.npz", **arrays)
            mujoco.mj_saveModel(env.sim.mj_model, str(out/"model.mjb"), None)
        report["finished"] = datetime.now().isoformat()
        (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
