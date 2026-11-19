"""Light control mapping, split-safe mixed experts, and GPU feature parity."""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import hashlib
import json
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace as NS
import unittest

import mujoco
import numpy as np
import torch

from humanoid_rl.algorithm.configuration import load_train_config
from humanoid_rl.algorithm.bfm_zero.config import resolve_agent_config
from humanoid_rl.algorithm.bfm_zero.dataset import MotionDataset, encode_expert
from humanoid_rl.tasks.motion_prior.env_cfg import MotionPriorEnvConfig, make_env_cfg
from robots.humanoid.humanoid_light.config import JOINT_NAMES, get_spec
from robots.humanoid.humanoid_light.mjlab import coupling_matrix, coupled_pd_torque, get_robot_spec, load_control

ROOT = Path(__file__).resolve().parents[1]


class LightControllerTests(unittest.TestCase):
    def test_source_coupling_torques_and_power_are_preserved(self):
        # Independent source equations, including asymmetric left/right signs.
        q = torch.tensor([[.2, -.1], [-.3, .15]])
        velocity = torch.tensor([[.4, -.8], [.7, -.2]])
        target = torch.tensor([[.25, -.05], [-.2, .2]])
        for pair, g1, g2 in (("left_ankle", -2., -2.), ("right_ankle", -2., -2.), ("torso", 1., 1.)):
            matrix = torch.tensor(coupling_matrix(pair, g1, g2))
            pitch, roll = (target-q).unbind(-1)
            vp, vr = velocity.unbind(-1)
            if pair == "left_ankle":
                m1, m2 = 20*g1*(roll-pitch)-.25*g1*(vr-vp), 20*g2*(roll+pitch)-.25*g2*(vr+vp)
                expected = torch.stack((-g1*m1+g2*m2, g1*m1+g2*m2), -1)
            elif pair == "right_ankle":
                m1, m2 = 20*g1*(roll+pitch)-.25*g1*(vr+vp), 20*g2*(roll-pitch)-.25*g2*(vr-vp)
                expected = torch.stack((g1*m1-g2*m2, g1*m1+g2*m2), -1)
            else:
                m1, m2 = 20*g1*(roll-pitch)-.25*g1*(vr-vp), -20*g2*(roll+pitch)+.25*g2*(vr+vp)
                expected = torch.stack((-g1*m1-g2*m2, g1*m1-g2*m2), -1)
            actual = coupled_pd_torque(matrix, 20., .25, target, q, velocity, 28.)
            torch.testing.assert_close(actual, expected.clamp(-28, 28))
            motor_torque = torch.tensor([[7., -11.], [3., 5.]])
            torch.testing.assert_close(((motor_torque@matrix)*velocity).sum(-1),
                                       (motor_torque*(velocity@matrix.T)).sum(-1))

    def test_task_defaults_retain_original_network_and_updates(self):
        original = load_train_config("bfm_zero")
        light = load_train_config("bfm_zero", task="Humanoid-Light-Motion-Prior")
        self.assertEqual(light["model"], original["model"])
        for key in ("batch_size", "updates_per_step", "num_envs"):
            self.assertEqual(light["training"][key], original["training"][key])
        self.assertEqual(resolve_agent_config(light, "Humanoid-Light-Motion-Prior").robot, "humanoid_light")
        model = get_robot_spec(load_control()).compile()
        colliders = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        self.assertTrue(bool((model.geom_contype[colliders] == 1).all()))
        self.assertTrue(bool((model.geom_conaffinity[colliders] == 1).all()))
        self.assertEqual(model.nexclude, 31)


class MixedExpertTests(unittest.TestCase):
    def test_short_expert_rollouts_use_only_real_states_and_resample_at_end(self):
        from humanoid_rl.algorithm.bfm_zero.rollout import VariableExpertRollout
        class Model:
            cfg = NS(seq_length=8)
            def sample_z(self, count, device):
                return torch.full((count, 2), -1., device=device)
            def backward_map(self, obs):
                self.last_input = obs["state"].clone()
                return obs["state"]
            def project_z(self, value):
                return value
        model = Model()
        states = torch.arange(20, 40, dtype=torch.float32).reshape(10, 2)
        dataset = NS(clip_probabilities=torch.ones(1), clip_starts=torch.zeros(1, dtype=torch.long),
                     clip_lengths=torch.tensor([10]), expert_slicer=NS(storage={"observation": {"state": states}}))
        train = NS(rollout_expert_trajectories_length=250, rollout_expert_trajectories_percentage=1.,
                   rollout_expert_trajectories=True, update_z_every_step=100, use_mix_rollout=False)
        agent = NS(cfg=NS(train=train, model=NS(archi=NS(z_dim=2))), _model=model)
        rollout = VariableExpertRollout(dataset, agent, 1, "cpu")
        z = None
        for frame in range(10):
            z = rollout(z, torch.tensor([[frame]]), {})
            self.assertTrue(bool((z >= 20).all()))
        torch.testing.assert_close(model.last_input, states[1:])
        self.assertEqual(int(rollout.length[0]), 9)
        self.assertEqual(int(rollout.position[0]), 1)

    def test_rejected_gaps_validation_subjects_and_source_weights(self):
        (ROOT/"outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/"outputs") as folder:
            directory = Path(folder)
            model = get_spec().compile()
            records = []
            for source, frames, intervals in (("cmu", 90, [(0, 30), (45, 90)]), ("lafan", 60, [(0, 60)])):
                path = directory/f"{source}.npz"
                q = np.repeat(model.qpos0[None], frames, axis=0).astype(np.float32)
                q[:, 2] = 1.+np.arange(frames)*.001
                np.savez_compressed(path, qpos=q, qvel=np.zeros((frames, model.nv), np.float32),
                                    frequency=np.array(50.), joint_names=np.array(("root",)+JOINT_NAMES),
                                    split_points=np.array([0, frames]))
                for start, stop in intervals:
                    records.append(dict(path=str(path.relative_to(ROOT)), source=source, subject=1,
                                        name=source, start_frame=start, stop_frame=stop,
                                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            manifest = dict(version=1, robot="humanoid_light", splits={"train": records, "validation": []})
            manifest_path = directory/"manifest.json"; manifest_path.write_text(json.dumps(manifest))
            settings = dict(root=".", manifest=str(manifest_path), cache_dir=str(directory/"cache"),
                            clip_seconds=10., source_weights={"cmu": .7, "lafan": .3}, expert_buffer_device="cpu")
            cfg = load_train_config("bfm_zero", task="Humanoid-Light-Motion-Prior")
            profile = resolve_agent_config(cfg, "Humanoid-Light-Motion-Prior")
            dataset = MotionDataset(settings, MotionPriorEnvConfig(), load_control(), 8, agent_cfg=profile)
            self.assertEqual(len(dataset.qpos), 135)
            np.testing.assert_allclose(dataset.clip_probabilities.numpy(), [.35, .35, .3])
            cached = MotionDataset(settings, MotionPriorEnvConfig(), load_control(), 8, agent_cfg=profile)
            self.assertEqual(dataset.fingerprint, cached.fingerprint)
            # A Colab checkout can store the same manifest/motions under a
            # different root without invalidating a saved expert fingerprint.
            with tempfile.TemporaryDirectory() as moved:
                moved = Path(moved)
                for record in records:
                    destination = moved/record["path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT/record["path"], destination)
                moved_manifest = moved/"manifest.json"
                moved_manifest.write_bytes(manifest_path.read_bytes())
                relocated = dict(settings, manifest=str(moved_manifest), manifest_root=str(moved),
                                 cache_dir=str(moved/"cache"))
                transferred = MotionDataset(relocated, MotionPriorEnvConfig(), load_control(), 8, agent_cfg=profile)
                self.assertEqual(dataset.fingerprint, transferred.fingerprint)
                self.assertEqual(dataset.clips, transferred.clips)
                torch.testing.assert_close(dataset.qpos, transferred.qpos, rtol=0, atol=0)
            torch.testing.assert_close(dataset.qpos, cached.qpos, rtol=0, atol=0)
            torch.testing.assert_close(dataset.qvel, cached.qvel, rtol=0, atol=0)
            for key in dataset.expert_slicer.storage["observation"]:
                torch.testing.assert_close(dataset.expert_slicer.storage["observation"][key],
                                           cached.expert_slicer.storage["observation"][key], rtol=0, atol=0)
            slicer = dataset.expert_slicer
            slicer.output_key_t = ["motion_id", "truncated"]
            batch = slicer.sample(800)
            ids = batch["motion_id"].reshape(-1, 8)
            self.assertTrue(bool((ids == ids[:, :1]).all()))
            self.assertFalse(bool(batch["truncated"].any()))
            self.assertEqual(dataset.clips[1]["start_frame"], 45)


@unittest.skipUnless(torch.cuda.is_available(), "Native mjlab needs CUDA")
class LightGpuTests(unittest.TestCase):
    def test_expert_online_parity_and_coupled_pd_execution(self):
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.utils.lab_api.math import quat_apply
        from humanoid_rl.env.bfm_features import from_environment
        cfg = load_train_config("bfm_zero", task="Humanoid-Light-Motion-Prior")
        profile = resolve_agent_config(cfg, "Humanoid-Light-Motion-Prior")
        control = load_control()
        env_cfg = make_env_cfg(MotionPriorEnvConfig(lie_down_probability=0), control, agent_cfg=profile)
        env_cfg.scene.num_envs = 4
        env = ManagerBasedRlEnv(env_cfg, device="cuda:0")
        try:
            env.reset()
            model = get_spec().compile()
            q = np.repeat(model.qpos0[None], 4, axis=0).astype(np.float32)
            q[:, 2] = np.array([.9, 1., 1.1, 1.2])
            rng = np.random.default_rng(3)
            quaternion = rng.normal(size=(4, 4))
            q[:, 3:7] = quaternion/np.linalg.norm(quaternion, axis=1, keepdims=True)
            q[:, 7:] = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1], (4, 26))
            qd = rng.normal(0, .1, (4, 32)).astype(np.float32)
            expected = encode_expert(model, q, qd, np.zeros(26, np.float32))
            qp, qv = torch.from_numpy(q).to(env.device), torch.from_numpy(qd).to(env.device)
            root = torch.cat((qp[:, :3]+env.scene.env_origins, qp[:, 3:7], qv[:, :3], quat_apply(qp[:, 3:7], qv[:, 3:6])), -1)
            robot = env.scene["robot"]
            self.assertEqual(tuple(robot.joint_names), JOINT_NAMES)
            action_term = env.action_manager.get_term("joint_pos")
            for action, bound in ((1., 1), (-1., 0)):
                env.action_manager.process_action(torch.full((4, 26), action, device=env.device))
                torch.testing.assert_close(action_term._processed_actions,
                                           robot.data.joint_pos_limits[..., bound])
            env.action_manager.process_action(torch.zeros((4, 26), device=env.device))
            torch.testing.assert_close(action_term._processed_actions, robot.data.default_joint_pos)
            robot.write_root_state_to_sim(root)
            robot.write_joint_state_to_sim(qp[:, 7:], qv[:, 6:])
            env.scene.write_data_to_sim(); env.sim.forward()
            actual = from_environment(env)
            for name in expected:
                torch.testing.assert_close(actual[name].cpu(), expected[name], atol=4e-5, rtol=4e-5)
            env.reset()
            for _ in range(20):
                env.step(torch.zeros((4, 26), device=env.device))
            self.assertTrue(bool(torch.isfinite(robot.data.joint_pos).all()))
            self.assertTrue(bool(torch.isfinite(robot.data.actuator_force).all()))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
