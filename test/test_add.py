"""ADD reward stability, trajectory observability and reset/phase alignment."""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

from humanoid_rl.algorithm.add.config import AddConfig, config_from_values
from humanoid_rl.env.imitation.add import configure_add_environment, reference_finished
from humanoid_rl.algorithm.add.features import EnvironmentAddFeatures, FEATURE_DIM, tracking_dataset
from humanoid_rl.algorithm.add.trainer import AddTrainer, DifferentialNormalizer, differential_reward
from humanoid_rl.algorithm.amp.dataset import MotionDataset, MotionSplit
from humanoid_rl.tasks.carry_box.fixed_scene import reset_fixed_scene
from humanoid_rl.env.mjlab_adapter import MjlabAdapter
from test_object_amp import DATA


MOTION = "episode_00002/motion.npz"


class AddUnitTests(unittest.TestCase):
    def test_zero_target_stays_zero_when_normalizer_updates_and_restores(self):
        norm = DifferentialNormalizer(3)
        pairs = torch.tensor([[1., -2., 0., 3., 2., 0.]])
        norm.update(pairs)
        norm.update(2*pairs)
        self.assertEqual(norm.count.item(), 4)
        torch.testing.assert_close(norm(torch.zeros_like(pairs)), torch.zeros_like(pairs))
        self.assertTrue(torch.isfinite(norm(pairs)).all())
        restored = DifferentialNormalizer(3)
        restored.load_state_dict(norm.state_dict())
        torch.testing.assert_close(restored(pairs), norm(pairs))

    def test_reward_extreme_logits_do_not_overflow_or_get_negative(self):
        reward = differential_reward(torch.tensor([-10000., -2., 0., 2., 10000.]))
        self.assertTrue(torch.isfinite(reward).all())
        self.assertTrue(bool((reward >= 0).all()))
        self.assertTrue(bool((reward.diff() >= 0).all()))
        self.assertLess(reward[-1].item(), 10)

    def test_add_rejects_unaligned_reference_setup(self):
        base = {"data_root": "fixture", "single_motion": MOTION}
        config_from_values(base)
        for override in ({"single_motion": None}, {"fixed_initial_state": False},
                         {"observe_object": False}, {"reference_init_probability": .5}):
            with self.assertRaises(ValueError):
                config_from_values(base | override)

    def test_update_relabel_and_checkpoint_preserve_differential_schema(self):
        torch.manual_seed(9)
        states = torch.randn(8, FEATURE_DIM)
        split = MotionSplit(states, torch.zeros(8, 97), torch.arange(7),
                            torch.zeros(7, dtype=torch.long), ["clip"], "fixture", [MOTION])
        data = SimpleNamespace(train=split, validation=None, joint_names=("fixture",))
        cfg = AddConfig("fixture", single_motion=MOTION, hidden_dims=(32,), batch_size=8)
        trainer = AddTrainer(cfg, data, "cpu")
        pairs = torch.randn(8, 2*FEATURE_DIM)
        before = [p.clone() for p in trainer.discriminator.parameters()]
        metrics = trainer.update(pairs)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, trainer.discriminator.parameters())))
        self.assertGreater(metrics["add/gradient_penalty"], 0)
        batch = {"amp_observation": pairs[:, :FEATURE_DIM], "amp_next_observation": pairs[:, FEATURE_DIM:]}
        trainer.relabel(batch)
        torch.testing.assert_close(batch["reward"], trainer.reward(pairs))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"add.pt"
            trainer.save(path)
            other = AddTrainer(cfg, data, "cpu")
            other.load(path)
            torch.testing.assert_close(trainer.reward(pairs), other.reward(pairs))
            self.assertEqual(other.update_step, 1)
            saved = torch.load(path, weights_only=True)
            saved["feature_version"] = "g1-body-object104-v1"
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "schema"):
                other.load(path)


@unittest.skipUnless(DATA.exists() and torch.cuda.is_available(), "ADD parity requires data and CUDA")
class AddEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.envs import ManagerBasedRlEnv
        from humanoid_rl.tasks.carry_box.env_cfg import make_env_cfg

        cfg = make_env_cfg()
        cfg.scene.num_envs = 2
        config = AddConfig(str(DATA), single_motion=MOTION, terminate_on_fall=False)
        configure_add_environment(cfg, config)
        cls.env = ManagerBasedRlEnv(cfg, device="cuda:0")
        model = get_spec().compile()
        names = tuple(model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0)
        cls.dataset = tracking_dataset(MotionDataset(DATA, names, observe_object=True, single_motion=MOTION))
        cls.env.cfg.add_reference = cls.dataset.train.states
        cls.features = EnvironmentAddFeatures(cls.env, cls.dataset.train.states)
        cls.adapter = MjlabAdapter(cls.env, cls.features)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_matching_reference_frames_have_zero_residual_including_box(self):
        env = self.env
        for frame in (0, 67, 150, 350, 567):
            reset_fixed_scene(env, None, env.cfg.fixed_scene | {"reference": self.dataset.train.references[frame]})
            env.episode_length_buf[:] = frame
            env.scene.write_data_to_sim()
            env.sim.forward()
            torch.testing.assert_close(self.features(env), torch.zeros(2, FEATURE_DIM, device=env.device), atol=3e-5, rtol=0)
        env.reset()

    def test_root_and_box_world_displacement_are_visible(self):
        env = self.env
        env.reset()
        scene = env.cfg.fixed_scene.copy()
        ref = scene["reference"].clone()
        ref[0] += .2
        ref[71] += .2
        scene["reference"] = ref
        reset_fixed_scene(env, None, scene)
        env.scene.write_data_to_sim()
        env.sim.forward()
        difference = self.features(env)
        torch.testing.assert_close(difference[:, 0], torch.full((2,), -.2, device=env.device), atol=3e-6, rtol=0)
        torch.testing.assert_close(difference[:, 106], torch.full((2,), -.2, device=env.device), atol=3e-6, rtol=0)
        # Moving robot and box together leaves the relative box position unchanged.
        torch.testing.assert_close(difference[:, 85:88], torch.zeros(2, 3, device=env.device), atol=3e-6, rtol=0)

    def test_terminal_phase_and_residual_are_captured_before_partial_reset(self):
        env = self.env
        env.cfg.episode_length_s = 2*env.step_dt
        self.adapter.observations = env.reset()[0]["actor"].clone()
        first, _ = self.adapter.step(torch.zeros(2, 29, device=env.device))
        frames = len(self.dataset.train.states)
        self.assertEqual(first["observation"].shape[-1], 121)
        torch.testing.assert_close(first["next_observation"][:, -1], torch.full((2,), 1/(frames-1), device=env.device))
        second, _ = self.adapter.step(torch.zeros(2, 29, device=env.device))
        self.assertTrue(bool(second["truncated"].all()))
        torch.testing.assert_close(second["next_observation"][:, -1], torch.full((2,), 2/(frames-1), device=env.device))
        self.assertTrue(bool((self.adapter.observations[:, -1] == 0).all()))
        self.assertFalse(torch.equal(second["amp_next_observation"], self.features(env)))
        env.episode_length_buf[:] = torch.tensor([12, 23], device=env.device)
        env.reset(env_ids=torch.tensor([1], device=env.device))
        self.assertEqual(env.episode_length_buf.tolist(), [12, 0])
        env.cfg.episode_length_s = (frames-1)*env.step_dt

    def test_reference_completion_is_true_terminal_without_bootstrapping(self):
        env = self.env
        self.adapter.observations = env.reset()[0]["actor"].clone()
        frames = len(self.dataset.train.states)
        env.episode_length_buf[:] = frames-2
        self.assertFalse(bool(reference_finished(env, frames).any()))
        transition, reasons = self.adapter.step(torch.zeros(2, 29, device=env.device))
        self.assertTrue(bool(transition["terminated"].all()))
        self.assertTrue(bool(reasons["reference_finished"].all()))
        torch.testing.assert_close(transition["next_observation"][:, -1], torch.ones(2, device=env.device))
        self.assertTrue(bool((self.adapter.observations[:, -1] == 0).all()))


if __name__ == "__main__":
    unittest.main()
