"""GPU integration checks for randomized scenes and off-policy reset semantics."""

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import unittest

import torch
from mjlab.envs import ManagerBasedRlEnv

from humanoid_rl.env.mjlab_adapter import MjlabAdapter
from humanoid_rl.tasks.carry_box.env_cfg import make_env_cfg


@unittest.skipUnless(torch.cuda.is_available(), "mjlab integration requires CUDA")
class CarryEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = make_env_cfg()
        cfg.scene.num_envs = 8
        cfg.seed = 7
        cls.env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        cls.adapter = MjlabAdapter(cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_randomized_scene_and_partial_reset(self):
        env = self.env
        env.reset()
        box = env.scene["box"]
        half = env.sim.model.geom_size[:, box.indexing.geom_ids[0]]
        self.assertTrue(bool(((2 * half >= 0.2) & (2 * half <= 0.4)).all()))
        self.assertGreater(float(half.std()), 0.005)
        mass = env.sim.model.body_mass[:, box.indexing.body_ids[0]]
        self.assertTrue(bool(((mass >= 0.5) & (mass <= 0.9)).all()))
        expected_z = (
            env.scene["source_support"].data.root_link_pos_w[:, 2]
            + 0.01
            + half[:, 2]
            + 0.001
        )
        torch.testing.assert_close(box.data.root_link_pos_w[:, 2], expected_z)
        target = env.scene["target_support"].data.root_link_pos_w
        torch.testing.assert_close(
            env.carry_goal_w[:, 2], target[:, 2] + 0.01 + half[:, 2]
        )
        old_goal = env.carry_goal_w.clone()
        old_sizes = half.clone()
        old_position = box.data.root_link_pos_w.clone()
        env.reset(env_ids=torch.tensor([0, 2], device=env.device))
        untouched = torch.tensor([1, 3, 4, 5, 6, 7], device=env.device)
        torch.testing.assert_close(env.carry_goal_w[untouched], old_goal[untouched])
        torch.testing.assert_close(
            env.sim.model.geom_size[untouched, box.indexing.geom_ids[0]],
            old_sizes[untouched],
        )
        torch.testing.assert_close(
            box.data.root_link_pos_w[untouched], old_position[untouched]
        )
        self.assertFalse(torch.equal(env.carry_goal_w[0], old_goal[0]))

    def test_placement_contact_emits_one_success_bonus(self):
        env = self.env
        self.adapter.observations = env.reset()[0]["actor"].clone()
        env_ids = torch.tensor([0], device=env.device)
        position = env.carry_goal_w[:1].clone()
        position[:, 2] += 0.001
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
        env.scene["box"].write_root_state_to_sim(
            torch.cat((position, identity, torch.zeros(1, 6, device=env.device)), -1),
            env_ids=env_ids,
        )
        env.scene.write_data_to_sim()
        env.sim.forward()
        actions = torch.zeros(8, 29, device=env.device)
        for _ in range(8):
            transition, reasons = self.adapter.step(actions)
            if reasons["success"][0]:
                self.assertEqual(float(transition["reward"][0]), 1.0)
                self.assertTrue(bool(transition["terminated"][0]))
                break
        else:
            self.fail("Placing the box on the target did not trigger success")
        following, _ = self.adapter.step(actions)
        self.assertEqual(float(following["reward"][0]), 0.0)

    def test_timeout_transition_has_terminal_observation_before_reset(self):
        env = self.env
        env.cfg.episode_length_s = env.step_dt
        self.adapter.observations = env.reset()[0]["actor"].clone()
        transition, reasons = self.adapter.step(torch.zeros(8, 29, device=env.device))
        self.assertEqual(transition["next_observation"].shape, (8, 120))
        self.assertTrue(bool(transition["truncated"].all()))
        self.assertFalse(bool(transition["terminated"].any()))
        self.assertTrue(bool(reasons["time_out"].all()))
        self.assertTrue(bool(torch.isfinite(transition["next_observation"]).all()))
        self.assertFalse(
            torch.equal(transition["next_observation"], self.adapter.observations)
        )
        self.assertTrue(bool((env.episode_length_buf == 0).all()))
        env.cfg.episode_length_s = 30

    def test_playback_observation_refresh_after_viewer_resets(self):
        from humanoid_rl.play import make_playback_env

        playback = make_playback_env(self.env)
        before = playback.get_observations().clone()
        # Mirrors the Viser GUI: reset through the unwrapped mjlab environment.
        self.env.reset(env_ids=torch.tensor([0], device=self.env.device))
        after = playback.get_observations()
        self.assertFalse(torch.equal(before[0], after[0]))
        torch.testing.assert_close(before[1:], after[1:])


if __name__ == "__main__":
    unittest.main()
