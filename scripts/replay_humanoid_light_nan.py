"""Replay captured reset/actions with CPU MuJoCo and the unchanged PD law."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

import mujoco
import numpy as np

from robots.humanoid.humanoid_light.mjlab import coupling_matrix, load_control


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--capture", type=Path, required=True)
    args = cli.parse_args()
    p = args.capture.resolve()
    archive = np.load(p/"failure.npz")
    eid = int(archive["env_ids"][0])
    control = load_control()
    results = []
    for dt, contacts in ((.005, True), (.0025, True), (.001, True), (.005, False)):
        model = mujoco.MjModel.from_binary_path(str(p/"model.mjb"))
        model.opt.timestep = dt
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        if not contacts:
            model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        data = mujoco.MjData(model)
        data.qpos[:] = archive[f"env_{eid}_reset_qpos"]
        data.qvel[:] = archive[f"env_{eid}_reset_qvel"]
        mujoco.mj_forward(model, data)
        names = [model.joint(int(jid)).name.split("/")[-1] for jid in model.actuator_trnid[:, 0]]
        jid = model.actuator_trnid[:, 0]
        qidx, vidx = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
        low, high = model.jnt_range[jid].T
        kp = np.array([control["stiffness"][name] for name in names])
        kd = np.array([control["damping"][name] for name in names])
        limit = np.array([control["effort_limit"][name] for name in names])
        default = np.array([control["default_joint_pos"][name] for name in names])
        joint_names = [model.joint(i).name.split("/")[-1] for i in range(1, model.njnt)]
        action_columns = [joint_names.index(name) for name in names]
        coupled = []
        for pair, group in (("left_ankle", "ankles"), ("right_ankle", "ankles"), ("torso", "torso_pitch_roll")):
            cfg = control["groups"][group]
            indices = [names.index(f"{pair}_{suffix}_joint") for suffix in ("pitch", "roll")]
            coupled.append((indices, np.array(coupling_matrix(pair, cfg["gear_ratio_1"], cfg["gear_ratio_2"])), cfg["gamma"]))
        worst_speed = 0.
        worst_acceleration = 0.
        worst_constraint = 0.
        failure = None
        steps = 0
        for vector, action in enumerate(archive["action"][:, 0]):
            a = action[action_columns]
            target = default+np.where(a >= 0, a*(high-default), a*(default-low))
            for substep in range(round(.02/dt)):
                error = target-data.qpos[qidx]
                velocity = data.qvel[vidx]
                torque = kp*error-kd*velocity
                for indices, matrix, gamma in coupled:
                    motor = kp[indices]*(matrix@error[indices])-kd[indices]*(matrix@velocity[indices])
                    torque[indices] = gamma*(matrix.T@motor)
                data.ctrl[:] = np.clip(torque, -limit, limit)
                mujoco.mj_step(model, data)
                steps += 1
                worst_speed = max(worst_speed,float(np.abs(data.qvel).max()))
                worst_acceleration = max(worst_acceleration,float(np.abs(data.qacc).max()))
                worst_constraint = max(worst_constraint,float(np.abs(data.qfrc_constraint).max()))
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or worst_speed > 1e6:
                    failure = dict(vector_step=vector, substep=substep, simulation_time=steps*dt)
                    break
            if failure:
                break
        result = dict(engine="CPU MuJoCo float64", dt=dt, contacts=contacts,
                      failure=failure, max_speed=worst_speed, max_acceleration=worst_acceleration,
                      max_constraint_force=worst_constraint, simulation_time=steps*dt)
        print(json.dumps(result), flush=True)
        results.append(result)
    (p/"cpu_replay.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
