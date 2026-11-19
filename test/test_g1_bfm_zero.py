"""G1 native mjlab wiring, original feature parity and enabled randomization."""

import ast
from collections import OrderedDict
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from humanoid_rl.algorithm.bfm_zero.config import resolve_agent_config
from humanoid_rl.algorithm.bfm_zero.upstream import BFM_ROOT, enable_bfm_zero
from humanoid_rl.algorithm.configuration import load_train_config
from humanoid_rl.train import parser

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT/"src/humanoid_rl/algorithm/bfm_zero/configs/g1_smoke.json"
DATA = ROOT/"src/humanoid_retarget/datasets/lafan1/retargeted/g1/bfm_zero_50hz/manifest.json"


class G1SettingsTests(unittest.TestCase):
    def test_task_defaults_and_original_joint_control_limits(self):
        from robots.humanoid.g1.bfm_zero import get_robot_spec,load_control
        args = parser(["--algorithm","bfm_zero","--task","G1-Motion-Prior"]).parse_args(
            ["--algorithm","bfm_zero","--task","G1-Motion-Prior"])
        self.assertEqual(args.train_settings["training"]["total_env_steps"],384000000)
        self.assertTrue(args.train_settings["environment"]["domain_randomization"])
        control = load_control()
        model = get_robot_spec(control).compile()
        self.assertEqual((model.nq,model.nv,model.nbody),(36,35,31))
        expected = [control["effort_limit"][name] for name in control["joint_names"]]
        np.testing.assert_allclose(model.jnt_actfrcrange[1:,1],expected)
        np.testing.assert_allclose(model.dof_armature[6:],list(control["armature"].values()))


@unittest.skipUnless(torch.cuda.is_available() and DATA.is_file(),"CUDA and original G1 motion exports required")
class G1EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.envs import ManagerBasedRlEnv
        from humanoid_rl.algorithm.bfm_zero.dataset import MotionDataset
        from humanoid_rl.env.bfm_adapter import BfmMjlabAdapter
        from humanoid_rl.tasks.motion_prior.env_cfg import MotionPriorEnvConfig, make_env_cfg
        from robots.humanoid.g1.bfm_zero import load_control
        values = load_train_config("bfm_zero",SMOKE,task="G1-Motion-Prior")
        profile = resolve_agent_config(values,"G1-Motion-Prior")
        settings = MotionPriorEnvConfig(**values["environment"])
        cls.dataset = MotionDataset(values["motion"],settings,load_control(),8,agent_cfg=profile)
        cfg = make_env_cfg(settings,agent_cfg=profile); cfg.scene.num_envs=4
        cls.env = ManagerBasedRlEnv(cfg,device="cuda:0")
        cls.adapter = BfmMjlabAdapter(cls.env,cls.dataset)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_noise_free_features_match_original_with_virtual_head_and_offset(self):
        from humanoid_rl.env.bfm_features import from_environment
        enable_bfm_zero()
        from humanoidverse.utils import torch_utils as original
        source = BFM_ROOT/"humanoidverse/envs/legged_robot_motions/legged_robot_motions.py"
        node = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node,ast.FunctionDef) and node.name == "compute_humanoid_observations_max")
        node.decorator_list=[]
        namespace = vars(original).copy() | {"OrderedDict":OrderedDict}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),namespace)
        robot = self.env.scene["robot"]
        data = robot.data
        parent = robot.body_names.index("torso_link")
        rot = data.body_link_quat_w[:,:,[1,2,3,0]]
        offset = original.my_quat_rotate(rot[:,parent],torch.tensor([0.,0.,.35],device=self.env.device).expand(4,3))
        pos = data.body_link_pos_w-self.env.scene.env_origins[:,None]
        vel,omega = data.body_link_lin_vel_w,data.body_link_ang_vel_w
        expected = namespace["compute_humanoid_observations_max"](
            torch.cat((pos,(pos[:,parent]+offset)[:,None]),1),torch.cat((rot,rot[:,parent:parent+1]),1),
            torch.cat((vel,(vel[:,parent]+torch.cross(omega[:,parent],offset,dim=-1))[:,None]),1),
            torch.cat((omega,omega[:,parent:parent+1]),1),True,True)
        enabled = self.env.cfg.motion_prior.observation_noise
        self.env.cfg.motion_prior.observation_noise=False
        try: actual=from_environment(self.env)
        finally: self.env.cfg.motion_prior.observation_noise=enabled
        torch.testing.assert_close(actual["privileged_state"],torch.cat(list(expected.values()),-1),atol=3e-5,rtol=3e-5)
        torch.testing.assert_close(actual["state"][:,:29],data.joint_pos+data.encoder_bias-data.default_joint_pos)
        self.assertEqual(actual["privileged_state"].shape,(4,463))

    def test_randomized_mass_inertia_and_strict_timeout(self):
        env = self.env
        bodies = env.scene["robot"].data.indexing.body_ids
        mass_ratio = env.sim.model.body_mass[:,bodies]/env.sim.get_default_field("body_mass")[bodies]
        inertia_ratio = env.sim.model.body_inertia[:,bodies]/env.sim.get_default_field("body_inertia")[bodies]
        self.assertTrue(bool(((mass_ratio>=.95)&(mass_ratio<=1.05)).all()))
        torch.testing.assert_close(inertia_ratio,mass_ratio[...,None].expand_as(inertia_ratio))
        env.episode_length_buf[:]=env.max_episode_length-1
        action=torch.zeros(4,29,device=env.device)
        first,_=self.adapter.step(action)
        self.assertFalse(bool(first["next"]["truncated"].any()))
        second,_=self.adapter.step(action)
        self.assertTrue(bool(second["next"]["truncated"].all()))
        self.assertFalse(bool(second["next"]["terminated"].any()))


if __name__ == "__main__":
    unittest.main()
