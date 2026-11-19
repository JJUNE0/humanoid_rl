"""CPU verification for light_supermimic_walk1 config, coupled action scale, and ReturnSampler setup."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class LightSuperMimicWalk1ConfigTests(unittest.TestCase):
    def test_walk1_plan_and_coupled_action_cfg(self):
        from mjlab.envs.mdp.actions import JointPositionActionCfg
        from humanoid_rl.algorithm.supermimic.motions import prepare
        from humanoid_rl.launchers.config import load_named, resolve
        from humanoid_rl.tasks.motion_tracking.light_supermimic import KEY_BODY_NAMES, make_cfg
        from robots.humanoid.humanoid_light.config import JOINT_NAMES

        plan = resolve(
            load_named("light_supermimic_walk1", "experiments"),
            load_named("local", "runtimes"),
            run_id="test_walk1",
        )
        self.assertEqual(plan.settings["runner"]["sampling_mode"], "return")
        self.assertEqual(plan.settings["environment"]["action_mode"], "coupled_scale")
        self.assertTrue(plan.settings["environment"]["key_bodies_only"])

        with tempfile.TemporaryDirectory() as tmp:
            motion_settings = dict(plan.settings["motion"], cache_dir=tmp)
            packed, meta = prepare(motion_settings)
            self.assertEqual(meta["frames"], 13041)
            self.assertGreater(meta["clips"], 20)

            env_cfg = make_cfg(
                packed,
                plan.settings["environment"],
                num_envs=2,
                seed=42,
                sampling_mode=plan.settings["runner"]["sampling_mode"],
            )
            action_cfg = env_cfg.actions["joint_pos"]
            self.assertIsInstance(action_cfg, JointPositionActionCfg)
            self.assertEqual(set(action_cfg.scale.keys()), set(JOINT_NAMES))
            # Coupled ankle effective stiffness is 8*20=160 -> scale = 0.25*28/160 = 0.04375
            self.assertAlmostEqual(action_cfg.scale["left_ankle_pitch_joint"], 0.04375)
            # Torso pitch/roll coupled stiffness is 2*100=200 -> scale = 0.25*36/200 = 0.045
            self.assertAlmostEqual(action_cfg.scale["torso_pitch_joint"], 0.045)
            # Knee uncoupled stiffness is 100 -> scale = 0.25*120/100 = 0.30
            self.assertAlmostEqual(action_cfg.scale["left_knee_joint"], 0.30)

            cmd = env_cfg.commands["motion"]
            self.assertEqual(cmd.sampling_mode, "external")
            self.assertEqual(cmd.body_names, KEY_BODY_NAMES)
            self.assertEqual(cmd.body_names[0], cmd.anchor_body_name)

    def test_lafan_all_t4_and_resume_compat(self):
        from humanoid_rl.algorithm.supermimic.motions import prepare
        from humanoid_rl.algorithm.supermimic.train import check_resume_settings
        from humanoid_rl.launchers.config import load_named, resolve
        from humanoid_rl.launchers.data import collect_inputs

        plan_t4 = resolve(
            load_named("light_supermimic_lafan_all_t4", "experiments"),
            load_named("colab", "runtimes"),
            run_id="test_lafan_t4",
        )
        plan_full = resolve(
            load_named("light_supermimic_lafan_all", "experiments"),
            load_named("local", "runtimes"),
            run_id="test_lafan_full",
        )
        changes = check_resume_settings(plan_t4.settings, plan_full.settings)
        self.assertIn("training.num_envs", changes)
        self.assertIn("runner.wm.batch_size", changes)

        files, _ = collect_inputs(plan_t4)
        self.assertGreaterEqual(len(files), 64)

        plan_local = resolve(
            load_named("light_supermimic_lafan_all_t4", "experiments"),
            load_named("local", "runtimes"),
            run_id="test_lafan_local",
        )
        packed, meta = prepare(plan_local.settings["motion"])
        self.assertTrue(packed.is_dir())
        self.assertEqual(meta["frames"], 668208)
        self.assertEqual(meta["clips"], 1888)


if __name__ == "__main__":
    unittest.main()
