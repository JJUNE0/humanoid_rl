"""Box-frame parity, coherent RSI and versioned object-AMP replay rewards."""

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul

from humanoid_rl.algorithm.amp.config import AmpConfig, config_from_values
from humanoid_rl.algorithm.amp.dataset import MotionDataset, MotionSplit
from humanoid_retarget.datasets.carry_box import scene_description
from humanoid_rl.env.imitation.amp import configure_amp_environment, reset_reference_scene
from humanoid_rl.algorithm.amp.features import EnvironmentFeatures, ExpertFeatures, object_features
from humanoid_rl.algorithm.amp.trainer import AmpTrainer
from humanoid_rl.env.mjlab_adapter import MjlabAdapter
from humanoid_rl.tasks.carry_box.env_cfg import make_env_cfg
from test_amp import make_motion


DATA = Path("/home/cocel/Desktop/Research/humanoid/I-BFM-Code/ibfm/tasks/carry_box/rollouts/carry_20_40_success500_v1")
BOX_YAML = "geom_pos: [0.01, -0.02, 0.03]\ngeom_quat_wxyz: [0.70710678, 0, 0, 0.70710678]\n"
ROBOT_XML = '<mujoco><worldbody><geom name="initial_box_support" type="box" pos="1 0 .39" size=".25 .25 .01"/><geom name="target_box_support" type="box" pos="2 0 .39" size=".25 .25 .01"/></worldbody></mujoco>'


def object_motion(names):
    motion = make_motion(names)
    length = len(motion["time"])
    motion.update(object_pos=np.tile([1., .1, .55], (length, 1)).astype(np.float32),
                  object_lin_vel=np.tile([.1, .2, -.1], (length, 1)).astype(np.float32),
                  object_ang_vel_local=np.tile([.3, -.2, .1], (length, 1)).astype(np.float32),
                  box_size_xyz_m=np.asarray([.3, .25, .35], np.float32),
                  box_mass_kg=np.asarray(.65, np.float32),
                  goal=np.asarray([2., 0, .55], np.float32))
    return motion


class ObjectAmpUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = get_spec().compile()
        cls.names = tuple(model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0)

    def test_geometry_centre_velocity_conversion_and_transform_invariance(self):
        motion = object_motion(self.names)
        scene = scene_description(BOX_YAML, ROBOT_XML)
        features, raw = ExpertFeatures(self.names, True).encode(motion, scene)
        self.assertEqual(features.shape, (4, 104))
        self.assertEqual(raw.shape, (4, 97))
        # Identity source body rotation means world geometry offset == local offset.
        np.testing.assert_allclose(raw[0, 71:74], motion["object_pos"][0] + scene["geom_position"], atol=1e-7)
        expected_velocity = motion["object_lin_vel"][0] + np.cross(motion["object_ang_vel_local"][0], scene["geom_position"])
        np.testing.assert_allclose(raw[0, 78:81], expected_velocity, atol=1e-7)
        root, quat = raw[:, :3], raw[:, 3:7]
        linear, angular = raw[:, 7:10], quat_apply(quat, raw[:, 10:13])
        # Recover native wrist positions from the body feature's heading-local XYZs.
        from mjlab.utils.lab_api.math import yaw_quat
        hand = root[:, None] + quat_apply(yaw_quat(quat)[:, None].expand(-1, 2, -1), features[:, 71:77].reshape(-1, 2, 3))
        original = object_features(root, quat, linear, angular, raw[:, 71:74], raw[:, 74:78], raw[:, 78:81], raw[:, 81:84], hand)
        yaw = quat_from_euler_xyz(torch.zeros(4), torch.zeros(4), torch.full((4,), .9))
        shift = torch.tensor([3., -2., .4])
        moved = object_features(quat_apply(yaw, root) + shift, quat_mul(yaw, quat), quat_apply(yaw, linear), quat_apply(yaw, angular),
                                quat_apply(yaw, raw[:, 71:74]) + shift, quat_mul(yaw, raw[:, 74:78]),
                                quat_apply(yaw, raw[:, 78:81]), quat_apply(yaw, raw[:, 81:84]),
                                quat_apply(yaw[:, None].expand(-1, 2, -1), hand) + shift)
        torch.testing.assert_close(original, moved, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(original, features[:, 83:])
        perturbed = dict(motion, object_pos=motion["object_pos"] + .2)
        changed = ExpertFeatures(self.names, True).encode(perturbed, scene)[0]
        torch.testing.assert_close(changed[:, :83], features[:, :83])
        self.assertFalse(torch.equal(changed[:, 83:], features[:, 83:]))

    def test_object_cache_tracks_geometry_sidecars_and_never_crosses_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                episode = root / str(index)
                episode.mkdir()
                np.savez(episode / "motion.npz", **object_motion(self.names))
                (episode / "box_config.yaml").write_text(BOX_YAML)
                (episode / "robot.xml").write_text(ROBOT_XML)
            (root / "prior_train.txt").write_text("0/motion.npz\n1/motion.npz\n")
            (root / "prior_validation.txt").write_text("2/motion.npz\n")
            dataset = MotionDataset(root, self.names, root / "cache", observe_object=True)
            self.assertEqual(dataset.train.starts.tolist(), [0, 1, 2, 4, 5, 6])
            self.assertEqual(dataset.train.sample_pairs(8).shape, (8, 208))
            cached = MotionDataset(root, self.names, root / "cache", observe_object=True)
            self.assertEqual(cached.train.fingerprint, dataset.train.fingerprint)
            (root / "0/box_config.yaml").write_text(BOX_YAML.replace("0.01", "0.04"))
            changed = MotionDataset(root, self.names, root / "cache", observe_object=True)
            self.assertNotEqual(changed.train.fingerprint, dataset.train.fingerprint)
            self.assertFalse(torch.equal(changed.train.states, dataset.train.states))
            self.assertEqual(changed.validation.fingerprint, dataset.validation.fingerprint)

    def test_discriminator_object_inputs_update_relabel_and_restore(self):
        state = torch.randn(24, 104)
        split = MotionSplit(state, torch.zeros(24, 97), torch.arange(23), torch.zeros(23, dtype=torch.long), ["motion"], "fixture", ["fixture"])
        data = SimpleNamespace(train=split, validation=split, joint_names=self.names)
        cfg = AmpConfig("fixture", observe_object=True, hidden_dims=(32,), batch_size=8)
        trainer = AmpTrainer(cfg, data, "cpu")
        pairs = torch.randn(8, 208)
        metrics = trainer.update(pairs)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertEqual(trainer.discriminator.network[0].in_features, 208)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "amp.pt"
            trainer.save(path)
            other = AmpTrainer(cfg, data, "cpu")
            other.load(path)
            torch.testing.assert_close(trainer.reward(pairs), other.reward(pairs))
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                AmpTrainer(AmpConfig("fixture"), data, "cpu")
        with self.assertRaisesRegex(ValueError, "boolean"):
            config_from_values({"data_root": "fixture", "observe_object": "false"})


@unittest.skipUnless(DATA.exists() and torch.cuda.is_available(), "Object AMP parity requires reference data and CUDA")
class ObjectAmpEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.envs import ManagerBasedRlEnv

        cfg = make_env_cfg()
        cfg.scene.num_envs = 2
        config = AmpConfig(str(DATA), observe_object=True, terminate_on_fall=False)
        configure_amp_environment(cfg, 1., amp_config=config)
        cls.env = ManagerBasedRlEnv(cfg, device="cuda:0")
        cls.features = EnvironmentFeatures(cls.env, True)
        episode = DATA / "episode_00002"
        scene = scene_description((episode / "box_config.yaml").read_bytes(), (episode / "robot.xml").read_bytes())
        with np.load(episode / "motion.npz", allow_pickle=False) as motion:
            cls.expected, cls.raw = ExpertFeatures(cls.features.joint_names, True).encode(motion, scene)
            # A held transport frame exercises the box frame/velocity conversion,
            # including the non-identity teacher box body's quaternion.
            cls.frame = int(np.flatnonzero(motion["phase"] == "transport")[20])
        raw = cls.raw[cls.frame:cls.frame + 1]
        cls.env.amp_dataset = SimpleNamespace(train=SimpleNamespace(sample_references=lambda count, device: raw.expand(count, -1).to(device)))
        cls.adapter = MjlabAdapter(cls.env, cls.features)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_reference_scene_preserves_expert_robot_box_features_and_geometry(self):
        env = self.env
        env.cfg.episode_length_s = 30
        env.reset()
        expected = self.expected[self.frame].expand(2, -1)
        torch.testing.assert_close(self.features(env).cpu(), expected, atol=3e-4, rtol=3e-4)
        box = env.scene["box"]
        gid, bid = box.indexing.geom_ids[0], box.indexing.body_ids[0]
        half = self.raw[self.frame, 84:87].to(env.device) / 2
        torch.testing.assert_close(env.sim.model.geom_size[:, gid], half.expand(2, -1))
        torch.testing.assert_close(env.sim.model.geom_aabb[:, gid, 1], half.expand(2, -1))
        torch.testing.assert_close(env.sim.model.geom_rbound[:, gid], half.norm().expand(2))
        torch.testing.assert_close(env.sim.model.body_mass[:, bid], self.raw[self.frame, 87].to(env.device).expand(2))
        target = env.scene["target_support"]
        top = target.data.root_link_pos_w[:, 2] + env.sim.model.geom_size[:, target.indexing.geom_ids[0], 2]
        torch.testing.assert_close(env.carry_goal_w[:, 2] - top, half[2].expand(2), atol=2e-5, rtol=2e-5)

    def test_partial_reference_scene_reset_preserves_unselected_world(self):
        env = self.env
        env.reset()
        before = self.features(env)[0].clone()
        box = env.scene["box"].data.root_link_pose_w[0].clone()
        goal = env.carry_goal_w[0].clone()
        reset_reference_scene(env, torch.tensor([1], device=env.device), 1.)
        env.scene.write_data_to_sim()
        env.sim.forward()
        torch.testing.assert_close(self.features(env)[0], before, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(env.scene["box"].data.root_link_pose_w[0], box)
        torch.testing.assert_close(env.carry_goal_w[0], goal)

    def test_terminal_object_features_are_saved_before_the_scene_reset(self):
        env = self.env
        env.cfg.episode_length_s = env.step_dt
        self.adapter.observations = env.reset()[0]["actor"].clone()
        transition, _ = self.adapter.step(torch.zeros(2, 29, device=env.device))
        self.assertTrue(bool(transition["truncated"].all()))
        self.assertEqual(transition["amp_next_observation"].shape, (2, 104))
        # Both encoders use the box geometry frame, and capture the same terminal state.
        torch.testing.assert_close(transition["amp_next_observation"][:, 83:92], transition["next_observation"][:, 67:76], atol=2e-5, rtol=2e-5)
        self.assertFalse(torch.equal(transition["amp_next_observation"], self.features(env)))
        env.cfg.episode_length_s = 30


if __name__ == "__main__":
    unittest.main()
