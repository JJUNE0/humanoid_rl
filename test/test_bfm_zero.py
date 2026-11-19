"""H1 expert/online parity, original FB-CPR-Aux updates and JSON configuration."""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from humanoid_rl.algorithm.bfm_zero.config import config_from_args, make_agent_config, validate_config
from humanoid_rl.algorithm.configuration import load_train_config
from humanoid_rl.train import parser
from humanoid_retarget.resample import resample_robot_motion


SMOKE = Path(__file__).resolve().parents[1]/"src/humanoid_rl/algorithm/bfm_zero/configs/smoke.json"
DATA = Path(__file__).resolve().parents[1]/"src/humanoid_retarget/datasets/lafan1/retargeted/h1/walk1_subject1.npz"
EXTRAS = all(importlib.util.find_spec(name) is not None for name in ("safetensors", "h5py", "tensordict"))


class ConfigurationTests(unittest.TestCase):
    def test_json_defaults_and_explicit_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory)/"train.json"
            custom.write_text(json.dumps({"training":{"num_envs":8},"model":{"hidden_dim":96}}))
            argv = ["--algorithm","bfm_zero","--task","H1-Motion-Prior","--train-config",str(custom),"--num-envs","16"]
            args = parser(argv).parse_args(argv)
            values = config_from_args(args)
            self.assertEqual(values["training"]["num_envs"], 16)
            self.assertEqual(values["model"]["hidden_dim"], 96)
            self.assertEqual(values["agent"]["grad_penalty_discriminator"], 10)
            self.assertEqual(load_train_config("bfm_zero")["training"]["num_envs"], 1024)

    def test_imitation_json_and_legacy_flashsac_defaults(self):
        args = parser([]).parse_args([])
        self.assertEqual((args.algorithm, args.num_envs, args.batch_size), ("flashsac",1024,2048))
        for mode in ("amp","add"):
            argv = ["--reward-mode",mode]
            args = parser(argv).parse_args(argv)
            self.assertEqual(args.n_step,1)
            self.assertEqual(args.train_settings["imitation"]["gradient_penalty"],10)
            self.assertEqual(args.train_settings["training"]["reward_mode"],mode)

    def test_invalid_sequences_and_unknown_training_keys_are_rejected(self):
        values = load_train_config("bfm_zero",SMOKE)
        values["training"]["batch_size"] = 9
        with self.assertRaisesRegex(ValueError,"divisible"):
            validate_config(values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"train.json"
            path.write_text('{"training":{"batch_szie":16}}')
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser(["--train-config",str(path)])

    def test_40_to_50_hz_slerp_preserves_time_and_quaternion_sign_invariance(self):
        time = np.arange(41)/40
        qpos = np.zeros((41,26),np.float32)
        qpos[:,0], qpos[:,7] = time, 2*time
        angle = time*np.pi/2
        qpos[:,3],qpos[:,6] = np.cos(angle/2),np.sin(angle/2)
        qpos[1::2,3:7] *= -1
        q, qd = resample_robot_motion(qpos,np.ones((41,25),np.float32),40,50)
        self.assertEqual(q.shape,(51,26))
        np.testing.assert_allclose(q[:,0],np.arange(51)/50,atol=1e-6)
        np.testing.assert_allclose(q[:,7],2*np.arange(51)/50,atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(q[:,3:7],axis=-1),1,atol=1e-6)
        expected = np.zeros((51,4)); expected[:,0]=np.cos(np.arange(51)/50*np.pi/4); expected[:,3]=np.sin(np.arange(51)/50*np.pi/4)
        np.testing.assert_allclose(np.abs((q[:,3:7]*expected).sum(-1)),1,atol=1e-6)
        np.testing.assert_allclose(qd,1)


@unittest.skipUnless(torch.cuda.is_available() and EXTRAS and DATA.is_file(), "BFM extras, CUDA and local H1 motions required")
class BfmEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.envs import ManagerBasedRlEnv
        from humanoid_rl.algorithm.bfm_zero.dataset import MotionDataset
        from humanoid_rl.env.bfm_adapter import BfmMjlabAdapter
        from humanoid_rl.tasks.motion_prior.env_cfg import MotionPriorEnvConfig, make_env_cfg
        from robots.humanoid.h1.mjlab import load_control
        cls.values = load_train_config("bfm_zero",SMOKE)
        cls.values["environment"]["lie_down_probability"] = 0
        settings = MotionPriorEnvConfig(**cls.values["environment"])
        cls.dataset = MotionDataset(cls.values["motion"], settings, load_control(),8,expert_device="cuda:0")
        cfg = make_env_cfg(settings); cfg.scene.num_envs=4; cfg.seed=7
        cls.env = ManagerBasedRlEnv(cfg,device="cuda:0")
        cls.adapter = BfmMjlabAdapter(cls.env,cls.dataset)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset()
        self.adapter.history.reset(slice(None))
        self.adapter.last_action.zero_(); self.adapter.step_count.zero_()
        self.adapter.observations = self.adapter.observe()

    def test_expert_and_online_observations_match_after_heading_and_translation(self):
        from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul
        from humanoid_rl.env.bfm_features import from_environment
        indices = torch.tensor([0,200,350,499])
        qpos = self.dataset.qpos[indices].to("cuda:0")
        qvel = self.dataset.qvel[indices].to("cuda:0")
        yaw = quat_from_euler_xyz(torch.zeros(4,device="cuda:0"),torch.zeros(4,device="cuda:0"),torch.full((4,),.7,device="cuda:0"))
        rotation = quat_mul(yaw,qpos[:,3:7])
        position = quat_apply(yaw,qpos[:,:3])+self.env.scene.env_origins
        position[:,:2] += torch.tensor([2.,-1.],device="cuda:0")
        root = torch.cat((position,rotation,quat_apply(yaw,qvel[:,:3]),quat_apply(rotation,qvel[:,3:6])),dim=-1)
        self.env.scene["robot"].write_root_state_to_sim(root)
        self.env.scene["robot"].write_joint_state_to_sim(qpos[:,7:],qvel[:,6:])
        self.env.scene.write_data_to_sim(); self.env.sim.forward()
        actual = from_environment(self.env)
        expected = self.dataset.expert_slicer.storage["observation"]
        for name in ("state","privileged_state"):
            torch.testing.assert_close(actual[name],expected[name][indices.to("cuda:0")],atol=3e-5,rtol=3e-5)
        self.assertEqual(actual["state"].shape,(4,44))
        self.assertEqual(actual["privileged_state"].shape,(4,298))

    def test_expert_sequences_never_cross_clip_boundaries(self):
        slicer = self.dataset.expert_slicer
        previous = slicer.output_key_t
        slicer.output_key_t = ["observation","motion_id","truncated"]
        try:
            batch=slicer.sample(128)
        finally:
            slicer.output_key_t=previous
        ids=batch["motion_id"].view(-1,8)
        self.assertTrue(bool((ids==ids[:,:1]).all()))
        self.assertFalse(bool(batch["truncated"].any()))

    def test_actor_action_units_and_history_alignment_match_original(self):
        previous_state = self.adapter.observations["state"].clone()
        first_action = torch.full((4,19),.1,device="cuda:0")
        transition,_ = self.adapter.step(first_action)
        torch.testing.assert_close(transition["next"]["observation"]["last_action"], first_action*5)
        torch.testing.assert_close(transition["next"]["observation"]["history_actor"][:,:19], torch.zeros_like(first_action))
        torch.testing.assert_close(self.adapter.history.states[:,0], previous_state)
        second_action = torch.full((4,19),.2,device="cuda:0")
        transition,_ = self.adapter.step(second_action)
        torch.testing.assert_close(transition["next"]["observation"]["last_action"], second_action*5)
        torch.testing.assert_close(transition["next"]["observation"]["history_actor"][:,:19], first_action*5)

    def test_reference_lying_reset_keeps_motion_joints_and_velocities(self):
        from mjlab.utils.lab_api.math import quat_conjugate, quat_mul
        qpos = self.dataset.qpos[:4].to("cuda:0").clone()
        qvel = self.dataset.qvel[:4].to("cuda:0").clone()
        probability = self.env.cfg.motion_prior.lie_down_probability
        self.env.cfg.motion_prior.lie_down_probability = 1.0
        try:
            with patch.object(self.dataset, "sample_states", return_value=(qpos.clone(), qvel.clone())):
                self.env.reset()
            data = self.env.scene["robot"].data
            torch.testing.assert_close(data.joint_pos, qpos[:,7:])
            torch.testing.assert_close(data.joint_vel, qvel[:,6:])
            torch.testing.assert_close(data.root_link_lin_vel_w, qvel[:,:3])
            torch.testing.assert_close(data.root_link_pos_w[:,2]-self.env.scene.env_origins[:,2],
                                       torch.full((4,),.5,device="cuda:0"))
            rotation = quat_mul(data.root_link_quat_w, quat_conjugate(qpos[:,3:7]))
            expected = torch.tensor([2**-.5,2**-.5,0.,0.],device="cuda:0").expand(4,4)
            torch.testing.assert_close(rotation.abs(), expected, atol=1e-5, rtol=1e-5)
        finally:
            self.env.cfg.motion_prior.lie_down_probability = probability

    def test_timeout_preserves_terminal_history_and_bootstrap_then_resets_selected_rows(self):
        self.env.episode_length_buf[0] = self.env.max_episode_length-1
        action = torch.full((4,19),.2,device="cuda:0")
        transition,_ = self.adapter.step(action)
        self.assertTrue(bool(transition["next"]["truncated"][0]))
        self.assertFalse(bool(transition["next"]["terminated"].any()))
        torch.testing.assert_close(transition["next"]["observation"]["last_action"],action*5)
        torch.testing.assert_close(self.adapter.observations["last_action"][0],torch.zeros(19,device="cuda:0"))
        torch.testing.assert_close(self.adapter.observations["last_action"][1:],action[1:]*5)
        self.assertGreater(float(transition["next"]["observation"]["history_actor"][0].abs().sum()),0)
        self.assertEqual(float(self.adapter.observations["history_actor"][0].abs().sum()),0)

    def test_original_update_changes_all_six_networks_and_checkpoint_roundtrips(self):
        from humanoidverse.agents.buffers.transition import DictBuffer
        from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent
        agent = make_agent_config(self.values).build(self.adapter.observation_space,19)
        replay = DictBuffer(capacity=64,device="cpu")
        for _ in range(4):
            action=torch.empty(4,19,device="cuda:0").uniform_(-.2,.2)
            transition,_ = self.adapter.step(action)
            transition["z"] = agent._model.sample_z(4,device="cuda:0")
            replay.extend(transition)
        names=("_actor","_forward_map","_backward_map","_discriminator","_critic","_aux_critic")
        before={name:[value.detach().clone() for value in getattr(agent._model,name).parameters()] for name in names}
        metrics=agent.update({"train":replay,"expert_slicer":self.dataset.expert_slicer},0)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in metrics.values()))
        self.assertIn("disc_wgan_gp_loss",metrics)
        for name in names:
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before[name],getattr(agent._model,name).parameters())),name)
        observation=self.adapter.observations
        z=agent._model.sample_z(4,device="cuda:0")
        with torch.no_grad(): expected=agent.act(observation,z,mean=True)
        with tempfile.TemporaryDirectory() as directory:
            agent.save(directory)
            restored=FBcprAuxAgent.load(directory,device="cuda")
            with torch.no_grad(): actual=restored.act(observation,z,mean=True)
        torch.testing.assert_close(actual,expected)


if __name__ == "__main__":
    unittest.main()
