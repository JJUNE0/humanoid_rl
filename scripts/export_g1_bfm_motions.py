"""Export original 30->50 Hz MotionLibRobot outputs without changing its math.

Use the upstream audit interpreter (MuJoCo dependencies), not the training venv.
The resulting numeric NPZ files need only NumPy/Torch in the mjlab trainer.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    upstream = ROOT/"third_party/BFM-Zero"
    sys.path.insert(0, str(upstream))
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
    from humanoidverse.envs.legged_robot_motions.legged_robot_motions import compute_humanoid_observations_max
    from humanoidverse.utils.helpers import get_reference_proprioception
    from humanoidverse.utils.torch_utils import quat_rotate_inverse

    data_root = ROOT/"src/humanoid_retarget/datasets/lafan1/retargeted/g1"
    source = data_root/"lafan_29dof_10s-clipped.pkl"
    target = data_root/"bfm_zero_50hz"
    target.mkdir(exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with hydra.initialize_config_dir(config_dir=str(upstream/"humanoidverse/config"), version_base=None):
        cfg = hydra.compose(config_name="exp/bfm_zero/bfm_zero", overrides=["robot=g1/g1_29dof_hard_waist"])
    OmegaConf.set_struct(cfg,False)
    cfg.robot.motion.asset.assetRoot = str(upstream/"humanoidverse/data/robots/g1")
    cfg.robot.motion.motion_file = str(source)
    cfg.robot.motion.step_dt = .02
    lib = MotionLibRobot(cfg.robot.motion, num_envs=1024, device="cpu")
    lib.multi_thread = False
    lib.load_motions(random_sample=False, num_motions_to_load=args.limit or lib._num_unique_motions)
    names = tuple(cfg.robot.dof_names)
    defaults = torch.tensor([[cfg.robot.init_state.default_joint_angles[name] for name in names]])
    fake_env = SimpleNamespace(default_dof_pos=defaults, gravity_vec=torch.tensor([[0.,0.,-1.]]), config=cfg)
    entries = []
    for index, key in enumerate(lib.curr_motion_keys):
        length = int(np.ceil(float(lib._motion_lengths[index])/.02))
        time = torch.arange(length)*.02
        state = lib.get_motion_state(torch.full((length,),index,dtype=torch.long),time)
        fields = get_reference_proprioception(fake_env,state["dof_pos"],state["dof_vel"],
                                              state["rb_rot"][:,0],state["body_ang_vel"][:,0])
        actor_state = torch.cat(list(fields.values()),-1)
        privileged = torch.cat(list(compute_humanoid_observations_max(
            state["rg_pos_t"],state["rg_rot_t"],state["body_vel_t"],state["body_ang_vel_t"],
            local_root_obs=True,root_height_obs=True).values()),-1)
        rotation = state["root_rot"]
        qpos = torch.cat((state["root_pos"],rotation[:,[3,0,1,2]],state["dof_pos"]),-1)
        angular = quat_rotate_inverse(rotation,state["root_ang_vel"],w_last=True)
        qvel = torch.cat((state["root_vel"],angular,state["dof_vel"]),-1)
        filename = str(key)+".npz"
        np.savez(target/filename,qpos=qpos.numpy(),qvel=qvel.numpy(),frequency=np.asarray(50.),
                 joint_names=np.asarray(("root",)+names),split_points=np.asarray([0,length]),
                 expert_state=actor_state.numpy(),expert_privileged_state=privileged.numpy(),
                 bfm_source_sha256=np.asarray(source_hash))
        entries.append({"file":filename,"frames":length})
        if index % 50 == 0:
            print(f"Exported {index+1}/{len(lib.curr_motion_keys)} G1 clips",flush=True)
    manifest = {"source":str(source.relative_to(ROOT)),"source_sha256":source_hash,
                "method":"original MotionLibRobot FK, Gaussian velocities and get_motion_state interpolation",
                "frequency":50,"state_dim":64,"privileged_dim":463,"clips":entries}
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"G1 original-motion export complete: {len(entries)} clips",flush=True)


if __name__ == "__main__":
    main()
