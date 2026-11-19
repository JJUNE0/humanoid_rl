"""Screened interval conversion and Light-specific dispatch, without a GPU."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from humanoid_rl.algorithm.supermimic.motions import prepare
from humanoid_rl.launchers.config import load_named, resolve
from humanoid_rl.launchers.source import sha256
from robots.humanoid.humanoid_light.config import JOINT_NAMES


class LightSuperMimicTests(unittest.TestCase):
    def test_task_dispatch_selects_isolated_official_environment(self):
        plan = resolve(load_named("light_supermimic_smoke", "experiments"), load_named("colab", "runtimes"), run_id="test")
        self.assertEqual(plan.method.environment, "supermimic_light")
        self.assertIn(".venv_supermimic_light", plan.python)
        self.assertIn("humanoid_rl.algorithm.supermimic.train", plan.snapshot()["command"])
        self.assertNotIn("${", plan.settings["motion"]["root"])
        self.assertEqual(plan.inputs[0]["kind"], "motion_manifest")

    def test_gaps_and_validation_are_preserved_and_angular_velocity_is_world_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            q = np.zeros((12, 33), np.float32)
            q[:, 2] = .71
            # Root yaw 90 degrees: local +x angular velocity becomes world +y.
            q[:, 3] = q[:, 6] = np.sqrt(.5)
            q[:, 7] = np.arange(12) / 100
            v = np.zeros((12, 32), np.float32); v[:, 3] = 1.
            source = root / "motion.npz"
            np.savez(source, qpos=q, qvel=v, joint_names=np.array(("root",) + JOINT_NAMES),
                     frequency=np.array(50.), split_points=np.array([0, 12]))
            def record(start, stop):
                return dict(path=source.name, start_frame=start, stop_frame=stop,
                            sha256=sha256(source), source="cmu")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(version=1, robot="humanoid_light", splits={
                "train": [record(0, 4), record(8, 12)], "validation": [record(4, 8)]})))
            settings = dict(root=str(root), manifest=str(manifest), cache_dir=str(root / "cache"),
                            split="train", chunk_frames=3)
            packed, metadata = prepare(settings)
            np.testing.assert_array_equal(np.load(packed / "motion_ends.npy"), [3, 6])
            np.testing.assert_allclose(np.load(packed / "joint_pos.npy")[:, 0], [0., .01, .02, .08, .09, .10])
            np.testing.assert_allclose(np.load(packed / "root_ang_vel_w.npy"), [[0., 1., 0.]] * 6, atol=2e-7)
            self.assertEqual(metadata["frames"], 6)
            self.assertEqual(prepare(settings)[1]["fingerprint"], metadata["fingerprint"])
            with source.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare(settings)
            settings["split"] = "validation"
            with self.assertRaisesRegex(ValueError, "held out"):
                prepare(settings)


if __name__ == "__main__":
    unittest.main()
