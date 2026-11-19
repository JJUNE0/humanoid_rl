"""Control-critic gradients, replay alignment, reset semantics and action bounds."""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from humanoid_rl.algorithm.amp.config import AmpConfig, ControlConfig, config_from_values
from humanoid_rl.env.control import ControlTerms
from humanoid_rl.algorithm.amp.normalization import RewardEMA
from humanoid_rl.algorithm.amp.auxiliary import AuxiliaryCritic, td_target
from humanoid_rl.env.imitation.amp import configure_amp_environment
from humanoid_rl.algorithm.amp.replay import AmpReplay
from humanoid_rl.env.mjlab_adapter import MjlabAdapter
from humanoid_rl.tasks.carry_box.env_cfg import make_env_cfg
from flash_rl.agents.flashSAC.network import FlashSACActor
from flash_rl.agents.utils.network import Network
from flash_rl.buffers.torch_buffer import TorchUniformBuffer


class ControlUnitTests(unittest.TestCase):
    def test_scale_only_ema_preserves_sign_and_restores_statistics(self):
        normalizer = RewardEMA(0.99)
        raw = torch.tensor([-1., -3., -5., -7.])
        torch.testing.assert_close(normalizer(raw), raw / raw.std(correction=0))
        self.assertTrue(bool((normalizer(raw) < 0).all()))
        restored = RewardEMA(0.99)
        restored.load_state_dict(normalizer.state_dict())
        torch.testing.assert_close(restored(raw, update=False), normalizer(raw, update=False))

    def test_auxiliary_td_bootstraps_timeout_but_not_success(self):
        actual = td_target(torch.tensor([-2., -2.]), torch.tensor([-10., -10.]),
                           torch.tensor([True, False]), torch.tensor([.99, .99]))
        torch.testing.assert_close(actual, torch.tensor([-2., -11.9]))

    def test_auxiliary_update_actor_gradient_and_checkpoint(self):
        cfg = SimpleNamespace(
            critic_hidden_dim=16, critic_num_blocks=1, learning_rate_peak=3e-4,
            learning_rate_init=3e-4, learning_rate_end=1.5e-4,
            learning_rate_warmup_step=0, learning_rate_decay_step=100,
            critic_target_update_tau=.01, gamma=.99,
        )
        auxiliary = AuxiliaryCritic(5, 2, cfg, ControlConfig(), torch.device("cpu"))
        actor = Network(FlashSACActor(1, 5, 16, 2))
        batch = {"observation": torch.randn(16, 5), "next_observation": torch.randn(16, 5),
                 "action": torch.rand(16, 2), "control_terms": torch.rand(16, 6),
                 "terminated": torch.zeros(16, dtype=torch.bool)}
        before = {k: p.clone() for k, p in auxiliary.critic.network.named_parameters()}
        auxiliary.update(batch, actor)
        self.assertTrue(all(np.isfinite(value) for value in auxiliary.metrics.values()))
        self.assertTrue(any(not torch.equal(p, before[k]) for k, p in auxiliary.critic.network.named_parameters()))
        auxiliary.critic.optimizer.zero_grad(set_to_none=True)
        actions = torch.randn(16, 2, requires_grad=True)
        bonus = auxiliary.actor_bonus(torch.ones(16), batch["observation"], actions)
        bonus.mean().backward()
        self.assertGreater(actions.grad.norm().item(), 0)
        self.assertTrue(all(p.grad is None for p in auxiliary.critic.network.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            auxiliary.save(directory)
            restored = AuxiliaryCritic(5, 2, cfg, ControlConfig(), torch.device("cpu"))
            restored.load(directory)
            with torch.no_grad():
                torch.testing.assert_close(auxiliary.critic(batch["observation"], actions),
                                           restored.critic(batch["observation"], actions))
            torch.testing.assert_close(restored.normalizer.mass, auxiliary.normalizer.mass)

    def test_control_replay_ring_and_restore(self):
        space = gym.spaces.Box(-np.inf, np.inf, (2, 1), np.float32)
        def make():
            return AmpReplay(TorchUniformBuffer(space, space, 1, .99, 4, 1, 2, "cpu"), 83, 6)
        replay = make()
        for start in (0, 2, 4):
            values = torch.arange(start, start + 2).float()
            replay.add({"observation": values[:, None], "next_observation": (values + 1)[:, None],
                        "action": torch.zeros(2, 1), "reward": torch.zeros(2),
                        "terminated": torch.zeros(2, dtype=torch.bool), "truncated": torch.zeros(2, dtype=torch.bool),
                        "amp_observation": values[:, None].expand(-1, 83),
                        "amp_next_observation": (values + 1)[:, None].expand(-1, 83),
                        "control_terms": values[:, None].expand(-1, 6)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay_buffer.pt"
            replay.save(path)
            restored = make()
            restored.load(path)
            batch = restored.sample(torch.arange(4))
            torch.testing.assert_close(batch["control_terms"][:, 0], batch["observation"][:, 0])

    def test_old_config_preserves_old_termination_and_scale(self):
        amp = config_from_values({"data_root": "fixture"})
        self.assertIsNone(amp.control)
        cfg = make_env_cfg()
        configure_amp_environment(cfg, amp_config=amp)
        self.assertIn("fallen", cfg.terminations)
        self.assertNotIn("success", cfg.terminations)
        self.assertEqual(amp.action_scale_multiplier, 1)


@unittest.skipUnless(torch.cuda.is_available(), "mjlab integration requires CUDA")
class ControlEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.envs import ManagerBasedRlEnv
        cfg = make_env_cfg()
        cfg.scene.num_envs = 2
        cls.amp = AmpConfig("fixture", terminate_on_fall=False, terminate_on_success=True,
                            action_scale_multiplier=5, control=ControlConfig())
        configure_amp_environment(cfg, amp_config=cls.amp)
        cls.env = ManagerBasedRlEnv(cfg, device="cuda:0")
        cls.adapter = MjlabAdapter(cls.env, control_fn=ControlTerms(cls.env, cls.amp.control))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_enlarged_action_targets_reach_carry_poses_and_respect_limits(self):
        env = self.env
        action = env.action_manager.get_term("joint_pos")
        zero = torch.zeros(2, 29, device=env.device)
        action.process_actions(zero)
        torch.testing.assert_close(action._processed_actions, action.offset)
        names = action.target_names
        desired = {"left_shoulder_pitch_joint": -1.2, "left_elbow_joint": -.4, "left_knee_joint": 2.0}
        for name, target in desired.items():
            j = names.index(name)
            zero[:, j] = (target - action.offset[:, j]) / action.scale[:, j]
            self.assertTrue(bool((zero[:, j].abs() < 1).all()))
        action.process_actions(zero)
        for name, target in desired.items():
            torch.testing.assert_close(action._processed_actions[:, names.index(name)], torch.full((2,), target, device=env.device))
        limits = env.scene["robot"].data.joint_pos_limits[:, action.target_ids]
        for sign in (-1, 1):
            action.process_actions(torch.full_like(zero, sign))
            self.assertTrue(bool((action._processed_actions >= limits[..., 0]).all()))
            self.assertTrue(bool((action._processed_actions <= limits[..., 1]).all()))

    def test_fall_continues_and_control_terms_are_copied_before_timeout_reset(self):
        env = self.env
        self.assertNotIn("fallen", env.termination_manager.active_terms)
        self.assertIn("success", env.termination_manager.active_terms)
        self.assertEqual(env.reward_manager.active_terms, [])
        self.adapter.observations = env.reset()[0]["actor"].clone()
        robot = env.scene["robot"]
        state = torch.cat((robot.data.root_link_pose_w, robot.data.root_link_vel_w), -1).clone()
        state[:, 2] = env.scene.env_origins[:, 2] + .15
        robot.write_root_state_to_sim(state)
        env.scene.write_data_to_sim()
        env.sim.forward()
        transition, _ = self.adapter.step(torch.zeros(2, 29, device=env.device))
        self.assertFalse(bool(transition["terminated"].any()))
        self.assertTrue(bool((env.episode_length_buf == 1).all()))
        env.cfg.episode_length_s = env.step_dt
        self.adapter.observations = env.reset()[0]["actor"].clone()
        transition, _ = self.adapter.step(torch.full((2, 29), .2, device=env.device))
        self.assertTrue(bool(transition["truncated"].all()))
        torch.testing.assert_close(transition["control_terms"][:, 0], torch.full((2,), 29., device=env.device))
        self.assertTrue(bool(torch.isfinite(transition["control_terms"]).all()))
        self.assertTrue(bool((env.action_manager.action == 0).all()))
        env.cfg.episode_length_s = 30

    def test_evaluation_repeats_fixed_spawns_and_preserves_training_rng(self):
        from humanoid_rl.algorithm.flashsac.evaluation import PolicyEvaluator
        evaluator = PolicyEvaluator(self.env.cfg, 2, 100, 42, self.env.device)
        actor = Network(FlashSACActor(1, self.adapter.observation_dim, 16, 29).to(self.env.device))
        agent = SimpleNamespace(_actor=actor)
        starts = []
        reset = evaluator.env.reset

        def capture_reset(*args, **kwargs):
            result = reset(*args, **kwargs)
            starts.append((evaluator.env.sim.data.qpos.clone(), evaluator.env.sim.model.geom_size.clone()))
            return result

        evaluator.env.reset = capture_reset
        try:
            cpu_state = torch.get_rng_state().clone()
            gpu_state = torch.cuda.get_rng_state().clone()
            first = evaluator.evaluate(agent)
            first_goal = evaluator.env.carry_goal_w.clone()
            torch.testing.assert_close(torch.get_rng_state(), cpu_state)
            torch.testing.assert_close(torch.cuda.get_rng_state(), gpu_state)
            second = evaluator.evaluate(agent)
            # CUDA contact reductions need not reproduce continuous box motion
            # bit for bit. Assert exact initial scenes and the original upright
            # diagnostics; the new interaction diagnostics must remain finite.
            for initial, repeated in zip(starts[0], starts[1]):
                torch.testing.assert_close(initial, repeated, rtol=0, atol=0)
            for key in ("eval/upright_time_s", "eval/survival_horizon", "eval/horizon_s", "eval/survival_1s"):
                self.assertEqual(first[key], second[key])
            self.assertTrue(all(math.isfinite(value) for value in first.values()))
            self.assertTrue(all(math.isfinite(value) for value in second.values()))
            torch.testing.assert_close(evaluator.env.carry_goal_w, first_goal)
        finally:
            evaluator.close()


if __name__ == "__main__":
    unittest.main()
