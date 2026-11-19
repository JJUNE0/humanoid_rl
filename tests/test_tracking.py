"""Tracking keeps vector-step axes and late TensorBoard evaluation records."""

from pathlib import Path
from types import SimpleNamespace
import unittest

from humanoid_rl.algorithm.flashsac.tensorboard_sync import pending_rows
from humanoid_rl.algorithm.flashsac.tracking import experiment_identity


class TrackingTests(unittest.TestCase):
    def test_add_logs_object_tracking_and_control_under_its_actual_method(self):
        metadata = {"task": "G1-Carry-Box", "num_envs": 1024, "observation_dim": 121,
                    "reward_mode": "add", "stage_model": None,
                    "add": {"feature_dim": 109, "config": {"observe_object": True,
                    "control": {"action_rate": -.1}, "single_motion": "clip/motion.npz",
                    "fixed_initial_state": True}}}
        identity = experiment_identity(metadata, Path("logs/add-run"))
        self.assertIn("object_add", identity["name"])
        self.assertIn("object-add", identity["tags"])
        self.assertIn("control-critic", identity["tags"])
        self.assertEqual(identity["config"]["add_pair_dim"], 218)
        self.assertIsNone(identity["config"]["amp_pair_dim"])
        self.assertTrue(identity["config"]["uses_object_add"])
        self.assertFalse(identity["config"]["uses_object_amp"])

    def test_late_evaluation_at_an_existing_step_is_not_dropped_or_duplicated(self):
        events = {"train/reward_mean": [SimpleNamespace(step=102400, value=.2, wall_time=1.)]}
        reader = SimpleNamespace(Tags=lambda: {"scalars": list(events)}, Scalars=lambda tag: events[tag])
        rows, counts = pending_rows(reader, {})
        self.assertEqual(rows[0][0], 102400)
        self.assertEqual(len(rows), 1)
        # Evaluation is flushed later at the same environment step.
        events["eval/survival_10s"] = [SimpleNamespace(step=102400, value=.5, wall_time=2.)]
        late, next_counts = pending_rows(reader, counts)
        self.assertEqual(late[0][0], 102400)
        self.assertEqual(late[0][1]["eval/survival_10s"], .5)
        self.assertNotIn("train/reward_mean", late[0][1])
        self.assertEqual(pending_rows(reader, next_counts)[0], [])
        events["train/reward_mean"] = []
        with self.assertRaisesRegex(ValueError, "shrank"):
            pending_rows(reader, next_counts)

    def test_identity_is_stable_and_records_the_actual_ablations(self):
        metadata = {"task": "G1-Carry-Box", "num_envs": 1024, "observation_dim": 120,
                    "stage_model": None, "amp": {"feature_dim": 104,
                    "config": {"observe_object": True, "control": {"action_rate": -.1}}}}
        first = experiment_identity(metadata, Path("logs/run1"))
        again = experiment_identity(metadata, Path("/relocated/logs/run1"))
        self.assertEqual(first["id"], again["id"])
        self.assertIn("no_mlp", first["name"])
        self.assertFalse(first["config"]["uses_frozen_mlp"])
        self.assertEqual(first["config"]["amp_pair_dim"], 208)
        changed = experiment_identity(metadata | {"stage_model": {"source": "frozen"}, "observation_dim": 148}, Path("logs/run2"))
        self.assertTrue(changed["config"]["uses_frozen_mlp"])
        self.assertIn("frozen_mlp", changed["name"])
        self.assertNotEqual(changed["id"], first["id"])


if __name__ == "__main__":
    unittest.main()
