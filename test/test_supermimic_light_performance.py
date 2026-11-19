"""Performance changes preserve finite-state diagnostics and resume semantics."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from humanoid_rl.algorithm.supermimic.performance import first_nonfinite
from humanoid_rl.algorithm.supermimic.muon import zeropower_fp16
from humanoid_rl.algorithm.supermimic.train import check_resume_settings
from humanoid_rl.algorithm.supermimic.tune import candidate_settings
from humanoid_rl.launchers.config import load_named, resolve


def settings(runtime="colab_t4"):
    return resolve(load_named("light_supermimic_t4_long", "experiments"),
                   load_named(runtime, "runtimes"), run_id="test").settings


class PerformanceTests(unittest.TestCase):
    def test_fp16_muon_tracks_fp32_and_preserves_small_gradients(self):
        from torch.optim._muon import _zeropower_via_newtonschulz

        coefficients = (3.4445, -4.7750, 2.0315)
        for shape in ((32, 64), (64, 32), (64, 64), (512, 512)):
            torch.manual_seed(42)
            grad = torch.randn(shape)
            value = grad / grad.norm().clamp(min=1e-7)
            if shape[0] > shape[1]: value = value.T
            for _ in range(5):
                gram = value @ value.T
                update = torch.addmm(gram, gram, gram, beta=coefficients[1], alpha=coefficients[2])
                value = torch.addmm(value, update, value, beta=coefficients[0])
            reference = value.T if shape[0] > shape[1] else value
            fp16 = zeropower_fp16(grad, coefficients, 5, 1e-7).float()
            bf16 = _zeropower_via_newtonschulz(grad, coefficients, 5, 1e-7).float()
            self.assertTrue(torch.isfinite(fp16).all())
            self.assertLess((fp16 - reference).norm(), (bf16 - reference).norm())
            self.assertGreater(zeropower_fp16(grad * 1e-8, coefficients, 5, 1e-7).abs().max(), 0)
        self.assertEqual(zeropower_fp16(torch.zeros(8, 8), coefficients, 5, 1e-7).abs().max(), 0)

    def test_fp16_muon_reuses_optimizer_state_and_keeps_weights_in_fp32(self):
        from torch.optim import _muon

        torch.manual_seed(42)
        parameter = torch.nn.Parameter(torch.randn(32, 64))
        optimizer = torch.optim.Muon([parameter], lr=2e-4)
        parameter.grad = torch.randn_like(parameter) * .01
        optimizer.step()
        restored = torch.nn.Parameter(parameter.detach().clone())
        next_optimizer = torch.optim.Muon([restored], lr=2e-4)
        next_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
        restored.grad = torch.randn_like(restored) * .01
        with patch.object(_muon, "_zeropower_via_newtonschulz", zeropower_fp16):
            next_optimizer.step()
        self.assertEqual(restored.dtype, torch.float32)
        self.assertTrue(torch.isfinite(restored).all())
        self.assertEqual(next_optimizer.state[restored]["momentum_buffer"].dtype, torch.float32)

    def test_combined_guard_checks_every_field_and_identifies_first_failure(self):
        fields = {"observation": torch.zeros(3, 65), "reward": torch.ones(3),
                  "qvel": torch.zeros(3, 32), "qacc": torch.zeros(3, 32)}
        self.assertIsNone(first_nonfinite(fields))
        fields["qacc"][2, 1] = float("inf")
        self.assertEqual(first_nonfinite(fields), "qacc")
        fields["qvel"][1, 2] = float("nan")
        self.assertEqual(first_nonfinite(fields), "qvel")
        self.assertIsNone(first_nonfinite({}))

    def test_tuning_preserves_samples_per_real_transition(self):
        base = settings("colab_t4_optimized")
        original = deepcopy(base)
        for count in (512, 1024, 2048, 4096):
            candidate = candidate_settings(base, count)
            updates = candidate["training"]["updates_per_step"]
            self.assertEqual(updates * candidate["runner"]["wm"]["batch_size"] / count, 1.)
            self.assertEqual(updates * candidate["runner"]["policy"]["batch_size"] / count, .5)
            self.assertTrue(candidate["training"]["cuda_graphs"])
            self.assertNotIn("tuning", candidate)
        self.assertEqual(base, original)
        with self.assertRaisesRegex(ValueError, "update-to-data"):
            candidate_settings(base, 513)

    def test_resume_allows_resource_changes_but_rejects_model_and_task_changes(self):
        prior = settings()
        current = candidate_settings(settings("colab_t4_optimized"), 1024, compile=True)
        changes = check_resume_settings(prior, current)
        self.assertEqual(changes["training.num_envs"], [64, 1024])
        self.assertEqual(changes["runner.policy.batch_size"], [32, 128])
        for section, key, value in (("runner", "dynamics_hidden_dims", [64]),
                                    ("environment", "physics_dt", .00125)):
            bad = deepcopy(current); bad[section][key] = value
            with self.assertRaisesRegex(ValueError, "Resume"):
                check_resume_settings(prior, bad)
        bad = deepcopy(current); bad["runner"]["policy"]["horizon"] = 16
        with self.assertRaisesRegex(ValueError, "runner"):
            check_resume_settings(prior, bad)


if __name__ == "__main__":
    unittest.main()
