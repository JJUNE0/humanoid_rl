"""Regression for early timeout bootstrapping in upstream n-step replay."""

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch

from humanoid_rl.algorithm.flashsac.upstream import enable_flashsac

enable_flashsac()
from flash_rl.agents.flashSAC.update import _compute_categorical_td_target  # noqa: E402
from flash_rl.buffers.torch_buffer import TorchUniformBuffer  # noqa: E402


class ReplayTests(unittest.TestCase):
    def test_timeout_keeps_terminal_state_and_actual_discount(self):
        space = gym.spaces.Box(-np.inf, np.inf, (2, 1), np.float32)
        replay = TorchUniformBuffer(space, space, 3, 0.9, 8, 1, 2, "cpu")
        # Env 0 times out immediately, then resets to a visibly different state.
        # Env 1 continues for all three steps.
        for i in range(3):
            replay.add(
                {
                    "observation": torch.tensor([[float(i)], [float(i)]]),
                    "action": torch.zeros(2, 1),
                    "reward": torch.ones(2),
                    "terminated": torch.tensor([False, False]),
                    "truncated": torch.tensor([i == 0, False]),
                    "next_observation": torch.tensor(
                        [[10.0 if i == 0 else 100.0], [float(i + 1)]]
                    ),
                }
            )
        batch = replay.sample(np.array([0, 1]))
        torch.testing.assert_close(batch["reward"], torch.tensor([1.0, 2.71]))
        torch.testing.assert_close(batch["discount"], torch.tensor([0.9, 0.729]))
        torch.testing.assert_close(
            batch["next_observation"], torch.tensor([[10.0], [3.0]])
        )
        self.assertFalse(batch["terminated"].any())
        self.assertTrue(bool(batch["truncated"][0]))
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/replay.pt"
            replay.save(path)
            replay.reset()
            replay.load(path)
            torch.testing.assert_close(
                replay.sample(np.array([0, 1]))["discount"], batch["discount"]
            )

    def test_distributional_target_uses_each_samples_discount(self):
        log_probs = (
            torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]).clamp_min(1e-30).log()
        )
        target = _compute_categorical_td_target(
            log_probs,
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            torch.tensor([0.5, 1.0]),
            3,
            0.0,
            2.0,
        )
        torch.testing.assert_close(
            target, torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), atol=1e-6, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
