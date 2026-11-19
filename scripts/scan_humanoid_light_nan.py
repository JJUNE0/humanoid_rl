"""Scan accepted CMU/LAFAN reset windows for conditional physics instability.

These are finite-horizon physical stress tests, not a data-quality certificate
or a learned policy evaluation. Sources and the training manifest are read-only.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

from diagnose_humanoid_light_nan import ROOT, ResetDataset
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply
from humanoid_rl.algorithm.configuration import load_train_config
from humanoid_rl.algorithm.bfm_zero.config import resolve_agent_config
from humanoid_rl.tasks.motion_prior.env_cfg import MotionPriorEnvConfig, make_env_cfg
from robots.humanoid.humanoid_light.mjlab import load_control


GIB = 2**30


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def check_headroom():
    available = next(int(s.split()[1])*1024 for s in Path("/proc/meminfo").read_text().splitlines()
                     if s.startswith("MemAvailable:"))
    free = int(subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free",
                                       "--format=csv,noheader,nounits"], text=True).strip())*2**20
    if available < 12*GIB or free < 3*GIB:
        raise RuntimeError(f"Diagnostic stopped for headroom: available RAM={available/GIB:.2f} GiB, "
                           f"free GPU={free/GIB:.2f} GiB")
    return dict(available_ram_gib=available/GIB, free_gpu_gib=free/GIB)


def make_trials(dataset, split, horizon):
    trials = []
    for clip_id, clip in enumerate(dataset.clips):
        last = max(0, clip["frames"]-horizon-1)
        for offset in sorted({0, last//2, last}):
            for mode in ("reference", "random"):
                key = f"{split}:{clip_id}:{offset}:{mode}"
                trials.append(dict(key=key, split=split, clip_id=clip_id, offset=offset,
                                   path=clip["path"], source=clip["dataset"],
                                   frame=clip["start_frame"]+offset, mode=mode,
                                   horizon=min(horizon, clip["frames"]-offset-1) if mode == "reference" else horizon,
                                   action_seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)))
    return trials


def run():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--candidates", type=Path)
    cli.add_argument("--dt", type=float, default=.005)
    cli.add_argument("--horizon", type=int, default=100)
    cli.add_argument("--num-envs", type=int, default=512)
    cli.add_argument("--njmax", type=int, default=512)
    cli.add_argument("--record-keys", type=Path, help="JSON trial keys whose physics poses should be saved for video")
    cli.add_argument("--limit", type=int)
    args = cli.parse_args()
    record_keys=set(json.loads(args.record_keys.read_text())) if args.record_keys else set()
    if not np.isclose(round(.02/args.dt)*args.dt,.02):
        cli.error("The physics timestep must divide 20 ms")
    os.chdir(ROOT)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.12,0)
    torch.manual_seed(4728)
    values = load_train_config("bfm_zero",task="Humanoid-Light-Motion-Prior")
    profile = resolve_agent_config(values,"Humanoid-Light-Motion-Prior")
    datasets = {split:ResetDataset(values["motion"],split) for split in ("train","validation")}
    trials = (json.loads(args.candidates.read_text()) if args.candidates else
              [t for split, dataset in datasets.items() for t in make_trials(dataset,split,args.horizon)])
    if args.limit:
        trials = trials[:args.limit]
    if not trials:
        write_json(out/"summary.json",dict(tests=0,candidates=0,finished=datetime.now().isoformat()))
        write_json(out/"candidates.json",[])
        return
    write_json(out/"trials.json",trials)
    report = dict(state="running",started=datetime.now().isoformat(),tests_total=len(trials),
                  tests_completed=0,dt=args.dt,njmax=args.njmax,control_dt=.02,artificial_lying_resets=False,
                  definitions=dict(nan="Nonfinite qpos, qvel, qacc or applied control",
                                   divergence="Peak absolute joint velocity >= 1000 rad/s or qacc >= 1e6",
                                   pass_label="No recorded instability within the sampled finite horizon"),
                  limitations=["A stress-test failure depends on commands, timestep and solver; it is not proof of a corrupt retarget.",
                               "Samples are three windows per accepted ten-second chunk, not every possible reset frame.",
                               "Reference mode uses joint-position targets with no root tracking force or velocity feedforward.",
                               "Random-mode commands are deterministic per trial and independent of GPU batch order."],
                  files_total=len({t["path"] for t in trials}))
    write_json(out/"status.json",report)
    print(json.dumps(report),flush=True)
    settings = MotionPriorEnvConfig(**(values["environment"] | dict(lie_down_probability=0.,
                            physics_dt=args.dt,decimation=round(.02/args.dt))))
    control = load_control()
    cfg = make_env_cfg(settings,control,agent_cfg=profile)
    cfg.scene.num_envs,cfg.seed=args.num_envs,4728
    cfg.sim.njmax=args.njmax
    env=ManagerBasedRlEnv(cfg,device="cuda:0")
    if record_keys:
        import mujoco
        mujoco.mj_saveModel(env.sim.mj_model,str(out/"model.mjb"),None)
    robot=env.scene["robot"]
    base_step=env.sim.step
    candidate_trials=[]
    results=[]
    try:
        for batch_start in range(0,len(trials),args.num_envs):
            resources=check_headroom()
            batch=trials[batch_start:batch_start+args.num_envs]
            count=len(batch)
            env.reset()
            ids=torch.arange(count,device=env.device)
            q=np.stack([datasets[t["split"]].qpos[int(datasets[t["split"]].clip_starts[t["clip_id"]])+t["offset"]].numpy() for t in batch])
            v=np.stack([datasets[t["split"]].qvel[int(datasets[t["split"]].clip_starts[t["clip_id"]])+t["offset"]].numpy() for t in batch])
            q[:,:2]=0
            qp,qv=torch.from_numpy(q).to(env.device),torch.from_numpy(v).to(env.device)
            root=torch.cat((qp[:,:3]+env.scene.env_origins[ids],qp[:,3:7],qv[:,:3],quat_apply(qp[:,3:7],qv[:,3:6])),-1)
            # Preserve world coordinates for confirmation; floating-point contact
            # arithmetic can otherwise change with a different slot in the batch.
            for i,t in enumerate(batch):
                if "world_origin_xy" in t:
                    root[i,:2]=torch.tensor(t["world_origin_xy"],device=env.device)
            initial_origin=root[:,:2].cpu().tolist()
            robot.write_root_state_to_sim(root,env_ids=ids)
            robot.write_joint_state_to_sim(qp[:,7:],qv[:,6:],env_ids=ids)
            env.scene.write_data_to_sim();env.sim.forward();env.scene.update(dt=0.)
            horizon=max(t["horizon"] for t in batch)
            actions=np.zeros((horizon,args.num_envs,26),np.float32)
            lower=robot.data.joint_pos_limits[0,:,0].cpu().numpy()
            upper=robot.data.joint_pos_limits[0,:,1].cpu().numpy()
            default=robot.data.default_joint_pos[0].cpu().numpy()
            for i,t in enumerate(batch):
                if t["mode"]=="random":
                    actions[:t["horizon"],i]=np.random.default_rng(t["action_seed"]).uniform(-1,1,(t["horizon"],26))
                else:
                    ds=datasets[t["split"]]
                    start=int(ds.clip_starts[t["clip_id"]])+t["offset"]+1
                    targets=ds.qpos[start:start+t["horizon"],7:].numpy()
                    span=np.where(targets>=default,upper-default,default-lower)
                    actions[:t["horizon"],i]=np.clip((targets-default)/span,-1,1)
            gpu_actions=torch.from_numpy(actions).to(env.device)
            active=torch.zeros(args.num_envs,dtype=torch.bool,device=env.device);active[:count]=True
            peak_v=torch.zeros(args.num_envs,device=env.device)
            peak_a=torch.zeros_like(peak_v);peak_constraint=torch.zeros_like(peak_v)
            peak_nefc=torch.zeros(args.num_envs,dtype=torch.long,device=env.device)
            overflow_flags=torch.zeros_like(peak_nefc)
            failures={};substep=0
            snapshots={}
            recorded={i:dict(qpos=[env.sim.data.qpos[i].cpu().numpy().copy()],
                             qvel=[env.sim.data.qvel[i].cpu().numpy().copy()],time=[0.])
                      for i,t in enumerate(batch) if t["key"] in record_keys}

            def checked_step():
                nonlocal substep,peak_v,peak_a,peak_constraint,peak_nefc,overflow_flags
                base_step();substep+=1
                data=env.sim.data
                for i,trace in recorded.items():
                    if bool(active[i]):
                        trace["qpos"].append(data.qpos[i].cpu().numpy().copy())
                        trace["qvel"].append(data.qvel[i].cpu().numpy().copy())
                        trace["time"].append(substep*args.dt)
                peak_nefc=torch.maximum(peak_nefc,torch.where(active,data.nefc.to(torch.long),0))
                overflow_flags|=torch.where(active,data.overflow.to(torch.long),0)
                speed=data.qvel[:,6:].abs().amax(-1)
                acc=data.qacc.abs().amax(-1)
                constraint=data.qfrc_constraint.abs().amax(-1)
                peak_v=torch.maximum(peak_v,torch.where(active,torch.nan_to_num(speed,nan=0.,posinf=0.),0.))
                peak_a=torch.maximum(peak_a,torch.where(active,torch.nan_to_num(acc,nan=0.,posinf=0.),0.))
                peak_constraint=torch.maximum(peak_constraint,torch.where(active,torch.nan_to_num(constraint,nan=0.,posinf=0.),0.))
                invalid=(~torch.isfinite(data.qpos).all(-1)|~torch.isfinite(data.qvel).all(-1)
                         |~torch.isfinite(data.qacc).all(-1)|~torch.isfinite(data.ctrl).all(-1))&active
                exploding=(speed>1e8)&active
                bad=(invalid|exploding).nonzero().flatten()
                if len(bad):
                    for i in bad.tolist():
                        failures[i]=dict(outcome="nan" if bool(invalid[i]) else "divergence",
                                         failure_time_s=substep*args.dt,physics_substep=substep)
                        snapshots[i]={k:getattr(data,k)[i].cpu().numpy().copy() for k in ("qpos","qvel","qacc","ctrl","qfrc_constraint")}
                    active[bad]=False
                    # Retire only failed worlds before their invalid state reaches
                    # kinematic observations. Other worlds keep their rollout.
                    env.reset(env_ids=bad)

            env.sim.step=checked_step
            for t in range(horizon):
                finished=torch.tensor([i for i,b in enumerate(batch) if b["horizon"]==t],device=env.device,dtype=torch.long)
                if len(finished):
                    active[finished]=False
                    env.reset(env_ids=finished)
                action=gpu_actions[t].clone();action[~active]=0
                env.step(action)
                if not active.any():
                    break
            env.sim.step=base_step
            peaks=torch.stack((peak_v,peak_a,peak_constraint),-1).cpu().numpy()
            counts=peak_nefc.cpu().tolist();overflows=overflow_flags.cpu().tolist()
            for i,t in enumerate(batch):
                row={k:v for k,v in t.items() if k!="world_origin_xy"}
                row.update(dt=args.dt,njmax=args.njmax,max_nefc=counts[i],overflow_flags=overflows[i],
                           batch=batch_start//args.num_envs,slot=i,
                           max_joint_speed_rad_s=float(peaks[i,0]),max_qacc=float(peaks[i,1]),
                           max_constraint_force=float(peaks[i,2]),outcome="finite")
                row.update(failures.get(i,{}))
                if row["outcome"]=="finite" and (peaks[i,0]>=1000 or peaks[i,1]>=1e6):
                    row["outcome"]="divergence"
                if row["outcome"]=="finite" and overflows[i]:
                    row["outcome"]="capacity_overflow"
                if row["outcome"]!="finite":
                    candidate=dict(t,world_origin_xy=initial_origin[i])
                    candidate_trials.append(candidate)
                    if i in snapshots:
                        np.savez_compressed(out/f"capture_{t['key'].replace(':','_')}.npz",**snapshots[i],
                                            reset_qpos=q[i],reset_qvel=v[i],actions=actions[:,i])
                results.append(row)
                if i in recorded:
                    trace=recorded[i]
                    np.savez_compressed(out/f"trace_{t['key'].replace(':','_')}.npz",
                                        qpos=np.stack(trace["qpos"]),qvel=np.stack(trace["qvel"]),
                                        time=np.array(trace["time"]),actions=actions[:,i])
            with (out/"results.jsonl").open("a") as stream:
                for row in results[batch_start:]:stream.write(json.dumps(row,allow_nan=False)+"\n")
            report.update(tests_completed=len(results),candidates=len(candidate_trials),
                          nan=sum(r["outcome"]=="nan" for r in results),
                          divergence=sum(r["outcome"]=="divergence" for r in results),
                          capacity_overflow=sum(bool(r.get("overflow_flags")) for r in results),
                          files_completed=len({t["path"] for t in trials[:len(results)]}),
                          resources=resources,updated=datetime.now().isoformat())
            write_json(out/"status.json",report);write_json(out/"candidates.json",candidate_trials)
            print(json.dumps(report),flush=True)
        report.update(state="complete",finished=datetime.now().isoformat())
        write_json(out/"summary.json",report);write_json(out/"status.json",report)
        columns=list(dict.fromkeys(k for r in results for k in r))
        with (out/"results.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(results)
    except BaseException as error:
        report.update(state="stopped",error=str(error),finished=datetime.now().isoformat())
        write_json(out/"status.json",report)
        raise
    finally:
        env.sim.step=base_step
        env.close()


if __name__=="__main__":
    run()
