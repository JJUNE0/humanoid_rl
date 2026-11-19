"""An interrupted optional replay export never publishes a partial replay pair."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from humanoid_rl.algorithm.flashsac.checkpoint import is_replay_io_error, replay_directory, save_replay_atomic, write_json_atomic


class CheckpointTests(unittest.TestCase):
    def test_replay_pair_is_published_together_after_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def writer(directory):
                destination = Path(directory)
                (destination / "replay_buffer.pt").write_bytes(b"base")
                self.assertIsNone(replay_directory(root))
                (destination / "amp_replay.pt").write_bytes(b"amp")

            save_replay_atomic(SimpleNamespace(save_replay_buffer=writer), root)
            self.assertEqual(replay_directory(root), root / "replay")
            self.assertEqual((root / "replay/amp_replay.pt").read_bytes(), b"amp")
            self.assertFalse(list(root.glob(".replay-writing-*")))

    def test_failed_export_keeps_model_metadata_and_no_resumeable_partial_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json_atomic(root / "run.json", {"env_steps": 51200000})
            (root / "actor.pt").write_bytes(b"actor")

            def writer(directory):
                (Path(directory) / "replay_buffer.pt").write_bytes(b"incomplete pair")
                raise OSError("No space left on device")

            with self.assertRaises(OSError):
                save_replay_atomic(SimpleNamespace(save_replay_buffer=writer), root)
            self.assertIsNone(replay_directory(root))
            self.assertEqual((root / "actor.pt").read_bytes(), b"actor")
            self.assertTrue((root / "run.json").exists())
            self.assertFalse(list(root.glob(".replay-writing-*")))

    def test_legacy_replay_and_io_error_classification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "replay_buffer.pt").write_bytes(b"legacy")
            self.assertEqual(replay_directory(root), root)
        self.assertTrue(is_replay_io_error(RuntimeError("unexpected pos 960 vs 854")))
        self.assertTrue(is_replay_io_error(OSError("disk full")))
        self.assertFalse(is_replay_io_error(RuntimeError("matmul shapes do not match")))


if __name__ == "__main__":
    unittest.main()
