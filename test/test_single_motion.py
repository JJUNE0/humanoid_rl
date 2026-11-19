"""Single-clip overfit isolation, exact scene resets and portable playback."""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

from humanoid_rl.algorithm.amp.config import AmpConfig, ControlConfig, config_from_values
from humanoid_rl.algorithm.amp.dataset import MotionDataset
from humanoid_rl.env.imitation.amp import configure_amp_environment
from humanoid_rl.tasks.carry_box.fixed_scene import BUNDLE_NAME, load_scene_bundle, reset_fixed_scene, scene_metadata
from humanoid_rl.algorithm.amp.features import EnvironmentFeatures
from humanoid_rl.algorithm.amp.trainer import AmpTrainer
from humanoid_rl.env.mjlab_adapter import MjlabAdapter
from humanoid_rl.play import configure_scene
from humanoid_rl.algorithm.flashsac.evaluation import PolicyEvaluator
from humanoid_rl.tasks.carry_box.env_cfg import make_env_cfg
from test_object_amp import object_motion, BOX_YAML, ROBOT_XML, DATA


MOTION = "episode_00002/motion.npz"


class SingleMotionUnitTests(unittest.TestCase):
    def test_single_motion_uses_only_selected_clip_and_no_fake_heldout_metrics(self):
        model = get_spec().compile()
        names = tuple(model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "chosen"
            episode.mkdir()
            np.savez(episode / "motion.npz", **object_motion(names))
            (episode / "box_config.yaml").write_text(BOX_YAML)
            (episode / "robot.xml").write_text(ROBOT_XML)
            # These deliberately refer to nonexistent, unrelated episodes.
            (root / "prior_train.txt").write_text("other/motion.npz\n")
            (root / "prior_validation.txt").write_text("heldout/motion.npz\n")
            data = MotionDataset(root, names, root / "cache", observe_object=True,
                                 single_motion="chosen/motion.npz")
            self.assertEqual(data.train.paths, ["chosen/motion.npz"])
            self.assertEqual(data.train.starts.tolist(), [0, 1, 2])
            self.assertIsNone(data.validation)
            cfg = AmpConfig(str(root), single_motion="chosen/motion.npz", observe_object=True,
                            hidden_dims=(16,), batch_size=8)
            trainer = AmpTrainer(cfg, data, "cpu")
            pairs = data.train.sample_pairs(8)
            metrics = trainer.update(pairs)
            self.assertEqual(metrics["amp/held_out_validation_enabled"], 0)
            self.assertIn("amp/train_reference_score", metrics)
            self.assertTrue(all(not key.startswith("amp/validation_") for key in metrics))
            path = root / "amp.pt"
            trainer.save(path)
            restored = AmpTrainer(cfg, data, "cpu")
            restored.load(path)
            torch.testing.assert_close(trainer.reward(pairs), restored.reward(pairs))
            with self.assertRaisesRegex(ValueError, "relative"):
                MotionDataset(root, names, root / "cache", single_motion="../outside/motion.npz")

    def test_fixed_start_rejects_incomplete_or_random_time_configuration(self):
        base = {"data_root": "fixture", "single_motion": MOTION,
                "fixed_initial_state": True, "observe_object": True,
                "reference_init_probability": 0}
        config_from_values(base)
        for override, message in (({"single_motion": None}, "requires"),
                                  ({"observe_object": False}, "requires"),
                                  ({"reference_init_probability": .5}, "reference_init_probability"),
                                  ({"single_motion": "/outside/motion.npz"}, "relative"),
                                  ({"max_motions": 1}, "omit max_motions")):
            with self.assertRaisesRegex(ValueError, message):
                config_from_values(base | override)


@unittest.skipUnless(DATA.exists() and torch.cuda.is_available(), "Fixed-scene parity requires data and CUDA")
class FixedSceneEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.envs import ManagerBasedRlEnv
        cls.amp = AmpConfig(str(DATA), single_motion=MOTION, fixed_initial_state=True,
            observe_object=True, reference_init_probability=0, terminate_on_fall=False,
            control=ControlConfig())
        cfg = make_env_cfg()
        cfg.scene.num_envs = 2
        configure_amp_environment(cfg, amp_config=cls.amp)
        cls.env = ManagerBasedRlEnv(cfg, device="cuda:0")
        cls.features = EnvironmentFeatures(cls.env, True)
        cls.dataset = MotionDataset(DATA, cls.features.joint_names, observe_object=True, single_motion=MOTION)
        cls.adapter = MjlabAdapter(cls.env, cls.features)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_exact_frame_zero_features_geometry_and_seed_independent_resets(self):
        env = self.env
        self.assertEqual(set(env.cfg.events), {"fixed_scene"})
        self.assertNotIn("fallen", env.termination_manager.active_terms)
        self.assertEqual(env.reward_manager.active_terms, [])
        expected = self.dataset.train.states[0].expand(2, -1)
        qpos = None
        for seed in (42, 1234):
            env.reset(seed=seed)
            torch.testing.assert_close(self.features(env).cpu(), expected, atol=3e-4, rtol=3e-4)
            if qpos is not None:
                torch.testing.assert_close(env.sim.data.qpos.clone(), qpos, atol=0, rtol=0)
            qpos = env.sim.data.qpos.clone()
            ref = self.dataset.train.references[0].to(env.device)
            torch.testing.assert_close(env.scene["robot"].data.root_link_pos_w - env.scene.env_origins,
                                       ref[:3].expand(2, -1), atol=2e-6, rtol=0)
            torch.testing.assert_close(env.carry_goal_w - env.scene.env_origins,
                                       ref[94:97].expand(2, -1), atol=2e-6, rtol=0)
            box = env.scene["box"]
            gid, bid = box.indexing.geom_ids[0], box.indexing.body_ids[0]
            torch.testing.assert_close(env.sim.model.geom_size[:, gid], (ref[84:87] / 2).expand(2, -1))
            torch.testing.assert_close(env.sim.model.body_mass[:, bid], ref[87].expand(2))
            for i, (name, top) in enumerate((("source_support", ref[88:91]), ("target_support", ref[91:94]))):
                support = env.scene[name]
                size = env.sim.model.geom_size[:, support.indexing.geom_ids[0]]
                torch.testing.assert_close(size, env.cfg.fixed_scene["support_half_sizes"][i].to(env.device).expand(2, -1))
                actual = support.data.root_link_pos_w - env.scene.env_origins
                actual = actual.clone()
                actual[:, 2] += size[:, 2]
                torch.testing.assert_close(actual, top.expand(2, -1), atol=2e-6, rtol=0)

    def test_feature_parity_in_each_phase_and_partial_reset_isolation(self):
        env = self.env
        with np.load(DATA / MOTION, allow_pickle=False) as motion:
            ids = [int(np.flatnonzero(motion["phase"] == name)[0])
                   for name in ("approach", "lift", "transport", "place", "complete")]
        for frame in ids:
            scene = env.cfg.fixed_scene | {"reference": self.dataset.train.references[frame]}
            reset_fixed_scene(env, None, scene)
            env.scene.write_data_to_sim()
            env.sim.forward()
            torch.testing.assert_close(self.features(env).cpu(), self.dataset.train.states[frame].expand(2, -1),
                                       atol=3e-4, rtol=3e-4)
        before = env.sim.data.qpos[1].clone()
        before_size = env.sim.model.geom_size[1].clone()
        env.reset(env_ids=torch.tensor([0], device=env.device))
        torch.testing.assert_close(env.sim.data.qpos[1], before, atol=0, rtol=0)
        torch.testing.assert_close(env.sim.model.geom_size[1], before_size, atol=0, rtol=0)
        torch.testing.assert_close(self.features(env)[0].cpu(), self.dataset.train.states[0], atol=3e-4, rtol=3e-4)

    def test_timeout_keeps_terminal_amp_state_then_restores_fixed_start(self):
        env = self.env
        env.cfg.episode_length_s = env.step_dt
        try:
            self.adapter.observations = env.reset()[0]["actor"].clone()
            transition, _ = self.adapter.step(torch.full((2, 29), .5, device=env.device))
            self.assertTrue(bool(transition["truncated"].all()))
            self.assertFalse(bool(transition["terminated"].any()))
            torch.testing.assert_close(self.features(env).cpu(), self.dataset.train.states[0].expand(2, -1), atol=3e-4, rtol=3e-4)
            self.assertFalse(torch.equal(transition["amp_next_observation"], self.features(env)))
            self.assertTrue(bool((env.action_manager.action == 0).all()))
        finally:
            env.cfg.episode_length_s = 30

    def test_playback_and_evaluator_restore_portable_scene_without_source_data(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)
            torch.save(self.env.cfg.fixed_scene, checkpoint / BUNDLE_NAME)
            bundle = load_scene_bundle(checkpoint)
            self.assertEqual(scene_metadata(bundle)["source_motion"], MOTION)
            metadata = {"carry": asdict(self.env.cfg.carry), "observation_dim": 120,
                "observe_hand_feet": True, "stage_model": None, "reward_mode": "amp",
                "amp": {"config": asdict(self.amp) | {"data_root": "/nonexistent/reference/data"}},
                "episode_length_s": 30}
            args = SimpleNamespace(task="G1-Carry-Box", scene_config=None,
                checkpoint=checkpoint, num_envs=2, seed=7, episode_length=None, env_index=0)
            cfg = configure_scene(args, metadata)
            evaluator = PolicyEvaluator(cfg, 2, 3, 8, "cuda:0")
            try:
                evaluator.env.reset()
                torch.testing.assert_close(self.features(evaluator.env).cpu(),
                    self.dataset.train.states[0].expand(2, -1), atol=3e-4, rtol=3e-4)
                self.assertEqual(set(evaluator.env.cfg.events), {"fixed_scene"})
                self.assertEqual(evaluator.env.termination_manager.active_terms, ["time_out"])
            finally:
                evaluator.close()


if __name__ == "__main__":
    unittest.main()
