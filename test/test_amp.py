"""AMP schema/frame parity, episode boundaries, relabeling and checkpoint tests."""

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul

from humanoid_rl.algorithm.amp.config import AmpConfig
from humanoid_rl.algorithm.amp.dataset import MotionDataset, MotionSplit
from humanoid_rl.algorithm.amp.discriminator import style_reward
from humanoid_rl.env.imitation.amp import configure_amp_environment, reset_reference_pose
from humanoid_rl.algorithm.amp.features import EnvironmentFeatures, ExpertFeatures
from humanoid_rl.algorithm.amp.replay import AmpReplay
from humanoid_rl.algorithm.amp.trainer import AmpTrainer
from humanoid_rl.env.mjlab_adapter import MjlabAdapter
from humanoid_rl.algorithm.flashsac.upstream import enable_flashsac
from humanoid_rl.tasks.carry_box.env_cfg import make_env_cfg


def make_motion(names, length=4):
    angles = torch.tensor([0.12])
    quat = quat_from_euler_xyz(angles, -angles, angles * 3).numpy()
    return {
        "quaternion_order": np.asarray("wxyz"),
        "fps": np.asarray(50.0),
        "time": np.arange(length) * 0.02,
        "root_pos": np.tile([0.1, 0.2, 0.8], (length, 1)).astype(np.float32),
        "root_quat": np.tile(quat, (length, 1)),
        "object_quat": np.tile([1.0, 0.0, 0.0, 0.0], (length, 1)),
        "dof_pos": np.tile(np.linspace(-0.1, 0.1, 29), (length, 1)).astype(np.float32),
        "dof_vel": np.tile(np.linspace(-0.2, 0.2, 29), (length, 1)).astype(np.float32),
        "root_lin_vel": np.tile([0.2, -0.1, 0.05], (length, 1)).astype(np.float32),
        "root_ang_vel_local": np.tile([0.3, 0.2, -0.15], (length, 1)).astype(
            np.float32
        ),
        "joint_names": np.asarray(names),
        "phase": np.asarray(["approach"] * length),
    }


class AmpUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = get_spec().compile()
        cls.names = tuple(
            model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0
        )

    def test_dataset_never_crosses_clips_and_reorders_named_joints(self):
        motion = make_motion(self.names)
        encoder = ExpertFeatures(self.names)
        original, _ = encoder.encode(motion)
        permutation = np.arange(29)[::-1]
        shuffled = dict(
            motion,
            joint_names=motion["joint_names"][permutation],
            dof_pos=motion["dof_pos"][:, permutation],
            dof_vel=motion["dof_vel"][:, permutation],
        )
        torch.testing.assert_close(encoder.encode(shuffled)[0], original)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(3):
                folder = root / str(i)
                folder.mkdir()
                np.savez(folder / "motion.npz", **motion)
            (root / "prior_train.txt").write_text("0/motion.npz\n1/motion.npz\n")
            (root / "prior_validation.txt").write_text("2/motion.npz\n")
            dataset = MotionDataset(root, self.names, root / "cache")
            self.assertEqual(dataset.train.starts.tolist(), [0, 1, 2, 4, 5, 6])
            self.assertEqual(dataset.train.sample_pairs(8).shape, (8, 166))
            cached = MotionDataset(root, self.names, root / "cache")
            torch.testing.assert_close(cached.train.states, dataset.train.states)
            (root / "prior_validation.txt").write_text("0/motion.npz\n")
            with self.assertRaisesRegex(ValueError, "overlap"):
                MotionDataset(root, self.names, root / "cache")

    def test_reward_range_and_current_discriminator_relabeling(self):
        torch.testing.assert_close(
            style_reward(torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])),
            torch.tensor([0.0, 0.0, 0.75, 1.0, 0.0]),
        )
        dataset = self.synthetic_dataset()
        trainer = AmpTrainer(
            AmpConfig("fixture", hidden_dims=(16,), batch_size=8), dataset, "cpu"
        )
        for parameter in trainer.discriminator.parameters():
            parameter.data.zero_()
        batch = {
            "amp_observation": torch.randn(8, 83),
            "amp_next_observation": torch.randn(8, 83),
            "reward": torch.full((8,), 123.0),
        }
        trainer.discriminator.network[-1].bias.data.fill_(-1)
        first = trainer.relabel(dict(batch))["reward"]
        trainer.discriminator.network[-1].bias.data.fill_(1)
        second = trainer.relabel(dict(batch))["reward"]
        torch.testing.assert_close(first, torch.zeros(8))
        torch.testing.assert_close(second, torch.ones(8))
        self.assertFalse(second.requires_grad)

    @staticmethod
    def synthetic_dataset():
        generator = torch.Generator().manual_seed(3)
        states = torch.randn(40, 83, generator=generator)
        split = MotionSplit(
            states,
            torch.zeros(40, 71),
            torch.arange(39),
            torch.zeros(39, dtype=torch.long),
            ["motion"],
            "fixture",
            ["fixture"],
        )
        return SimpleNamespace(
            train=split, validation=split, joint_names=tuple(str(i) for i in range(29))
        )

    def test_discriminator_update_and_checkpoint_restore(self):
        dataset = self.synthetic_dataset()
        cfg = AmpConfig("fixture", hidden_dims=(32, 16), batch_size=16)
        trainer = AmpTrainer(cfg, dataset, "cpu")
        pairs = torch.randn(16, 166) + 2
        before = [p.detach().clone() for p in trainer.discriminator.parameters()]
        metrics = trainer.update(pairs)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertTrue(
            any(
                not torch.equal(p, old)
                for p, old in zip(trainer.discriminator.parameters(), before)
            )
        )
        expected = trainer.reward(pairs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "amp.pt"
            trainer.save(path)
            restored = AmpTrainer(cfg, dataset, "cpu")
            restored.load(path)
            torch.testing.assert_close(restored.reward(pairs), expected)
            self.assertEqual(restored.update_step, 1)

    def test_amp_replay_ring_alignment_and_restore(self):
        enable_flashsac()
        from flash_rl.buffers.torch_buffer import TorchUniformBuffer

        space = gym.spaces.Box(-np.inf, np.inf, (2, 1), np.float32)

        def replay(capacity=4):
            return AmpReplay(
                TorchUniformBuffer(space, space, 1, 0.99, capacity, 1, 2, "cpu"), 83
            )

        buffer = replay()
        for start in (0, 2, 4):
            values = torch.arange(start, start + 2).float()
            buffer.add(
                {
                    "observation": values[:, None],
                    "action": torch.zeros(2, 1),
                    "next_observation": (values + 1)[:, None],
                    "reward": torch.zeros(2),
                    "terminated": torch.zeros(2, dtype=torch.bool),
                    "truncated": torch.zeros(2, dtype=torch.bool),
                    "amp_observation": values[:, None].expand(-1, 83),
                    "amp_next_observation": (values + 1)[:, None].expand(-1, 83),
                }
            )
        batch = buffer.sample(torch.arange(4))
        torch.testing.assert_close(
            batch["observation"][:, 0], batch["amp_observation"][:, 0]
        )
        torch.testing.assert_close(
            batch["next_observation"][:, 0], batch["amp_next_observation"][:, 0]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay_buffer.pt"
            buffer.save(path)
            restored = replay()
            restored.load(path)
            torch.testing.assert_close(
                restored.sample(torch.arange(4))["amp_observation"],
                batch["amp_observation"],
            )
            grown = replay(8)
            grown.load(path)
            self.assertEqual(grown.base._current_idx, 4)


@unittest.skipUnless(
    torch.cuda.is_available(), "AMP/mjlab feature parity requires CUDA"
)
class AmpEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.envs import ManagerBasedRlEnv

        cfg = make_env_cfg()
        cfg.scene.num_envs = 2
        configure_amp_environment(cfg)
        cls.env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        cls.features = EnvironmentFeatures(cls.env)
        cls.adapter = MjlabAdapter(cls.env, cls.features)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_expert_and_simulation_features_agree_for_identical_physical_state(self):
        env = self.env
        motion = make_motion(self.features.joint_names)
        expected, reference = ExpertFeatures(self.features.joint_names).encode(motion)
        raw = reference[:1].to(env.device)
        position = raw[:, :3] + env.scene.env_origins[:1]
        angular = quat_apply(raw[:, 3:7], raw[:, 10:13])
        ids = torch.tensor([0], device=env.device)
        robot = env.scene["robot"]
        robot.write_root_state_to_sim(
            torch.cat((position, raw[:, 3:10], angular), -1), env_ids=ids
        )
        robot.write_joint_state_to_sim(raw[:, 13:42], raw[:, 42:71], env_ids=ids)
        env.scene.write_data_to_sim()
        env.sim.forward()
        torch.testing.assert_close(
            self.features(env)[0].cpu(), expected[0], atol=2e-4, rtol=2e-4
        )

    def test_policy_hand_feet_match_amp_frame_and_ignore_world_translation_heading(self):
        from humanoid_rl.tasks.carry_box.mdp.observations import hand_foot_relative_positions

        env = self.env
        observations, _ = env.reset()
        self.assertEqual(observations["actor"].shape, (2, 120))
        before = hand_foot_relative_positions(env).clone()
        torch.testing.assert_close(observations["actor"][:, -12:], self.features(env)[:, -12:])
        robot = env.scene["robot"]
        state = torch.cat((robot.data.root_link_pose_w, robot.data.root_link_vel_w), -1).clone()
        state[:, :3] += torch.tensor([2., -1., .3], device=env.device)
        angles = torch.full((2,), .7, device=env.device)
        rotation = quat_from_euler_xyz(torch.zeros_like(angles), torch.zeros_like(angles), angles)
        state[:, 3:7] = quat_mul(rotation, state[:, 3:7])
        robot.write_root_state_to_sim(state)
        env.scene.write_data_to_sim()
        env.sim.forward()
        torch.testing.assert_close(hand_foot_relative_positions(env), before, atol=2e-5, rtol=2e-5)

    def test_amp_only_terminal_pairs_do_not_cross_reset(self):
        env = self.env
        self.assertEqual(env.reward_manager.active_terms, [])
        self.assertNotIn("success", env.termination_manager.active_terms)
        env.cfg.episode_length_s = env.step_dt
        self.adapter.observations = env.reset()[0]["actor"].clone()
        transition, _ = self.adapter.step(torch.zeros(2, 29, device=env.device))
        self.assertTrue(bool(transition["truncated"].all()))
        self.assertTrue(bool((transition["reward"] == 0).all()))
        self.assertEqual(transition["amp_next_observation"].shape, (2, 83))
        self.assertFalse(
            torch.equal(transition["amp_next_observation"], self.features(env))
        )
        env.cfg.episode_length_s = 30

    def test_reference_reset_preserves_style_features_and_leaves_box_unchanged(self):
        env = self.env
        env.reset()
        motion = make_motion(self.features.joint_names)
        expected, references = ExpertFeatures(self.features.joint_names).encode(motion)
        split = MotionSplit(
            expected,
            references,
            torch.arange(3),
            torch.zeros(3, dtype=torch.long),
            ["motion"],
            "fixture",
            ["fixture"],
        )
        env.amp_dataset = SimpleNamespace(train=split)
        box_before = env.scene["box"].data.root_link_pose_w.clone()
        goal_before = env.carry_goal_w.clone()
        reset_reference_pose(env, torch.tensor([0, 1], device=env.device), 1.0)
        env.scene.write_data_to_sim()
        env.sim.forward()
        torch.testing.assert_close(
            self.features(env).cpu(), expected[:2], atol=2e-4, rtol=2e-4
        )
        torch.testing.assert_close(env.scene["box"].data.root_link_pose_w, box_before)
        torch.testing.assert_close(env.carry_goal_w, goal_before)


if __name__ == "__main__":
    unittest.main()
