"""Actor-only playback must reproduce deterministic FlashSAC inference."""

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from humanoid_rl.play import make_policy
from humanoid_rl.algorithm.flashsac.upstream import enable_flashsac


class PlaybackPolicyTests(unittest.TestCase):
    def test_playback_restores_legacy_and_new_observation_schemas(self):
        from humanoid_rl.play import configure_scene

        args = SimpleNamespace(task="G1-Carry-Box", scene_config=None, num_envs=1,
                               seed=42, episode_length=None, env_index=0)
        old = {"carry": None, "episode_length_s": 30, "observation_dim": 108}
        legacy = configure_scene(args, old)
        self.assertFalse(legacy.observe_hand_feet)
        self.assertNotIn("hand_foot_position", legacy.observations["actor"].terms)
        current = configure_scene(args, old | {"observe_hand_feet": True, "observation_dim": 120})
        self.assertTrue(current.observe_hand_feet)
        self.assertIn("hand_foot_position", current.observations["actor"].terms)

    def test_checkpoint_actor_matches_deterministic_inference(self):
        enable_flashsac()
        from flash_rl.agents.flashSAC.network import FlashSACActor

        torch.manual_seed(19)
        actor = FlashSACActor(num_blocks=2, input_dim=5, hidden_dim=16, action_dim=2)
        obs = torch.randn(3, 5)
        with torch.no_grad():
            mean, _ = actor.get_mean_and_std(obs, training=False)
            expected = mean.tanh()
        # Exercise a compiled training checkpoint loaded by an uncompiled actor.
        state = {"_orig_mod." + key: value for key, value in actor.state_dict().items()}
        metadata = {
            "observation_dim": 5,
            "action_dim": 2,
            "agent": {
                "actor_num_blocks": 2,
                "actor_hidden_dim": 16,
                "asymmetric_observation": False,
            },
        }
        env = SimpleNamespace(observation_dim=5, action_dim=2, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            torch.save({"network_state_dict": state}, Path(directory) / "actor.pt")
            args = SimpleNamespace(agent="trained", checkpoint=Path(directory))
            policy = make_policy(args, env, metadata)
            actual = policy(obs)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(policy(obs), actual)
            self.assertFalse(actual.requires_grad)
            self.assertTrue(bool((actual.abs() <= 1).all()))


if __name__ == "__main__":
    unittest.main()
