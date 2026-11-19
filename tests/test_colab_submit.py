"""Direct Colab execution preserves trainer failures, results, and session ownership."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from humanoid_rl.launchers import colab_job, colab_submit
from humanoid_rl.launchers.colab import prepare
from humanoid_rl.launchers.config import ROOT, load_named, read_json, resolve
from humanoid_rl.launchers.source import sha256


class DirectColabTests(unittest.TestCase):
    def test_checkpoint_transfers_are_bounded_without_deleting_existing_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index in range(4):
                p = root / "training/checkpoints" / f"step_{index}"
                p.mkdir(parents=True)
                (p / "last.pt").write_bytes(b"weights")
                os.utime(p, ns=(index + 1, index + 1))
                paths.append(p)
            (paths[0].parent / ".writing").mkdir()
            self.assertEqual(colab_job.archived_checkpoints(root), set(paths[-2:]))
            self.assertTrue(all((p / "last.pt").exists() for p in paths))

    def test_submit_dry_run_needs_no_gpu_or_colab_client_and_rejects_local_submit(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-S", str(ROOT / "humanoid.py"), "launch", "--experiment",
                       "light_supermimic_smoke", "--runtime", "colab", "--submit", "--dry-run",
                       "--results-root", directory]
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True, check=True)
            saved = json.loads(result.stdout)
            self.assertEqual(saved["action"], "submit_colab")
            self.assertEqual(saved["colab_gpu"], "T4")
            self.assertIsNone(saved["runtime"]["persistent_root"])
            self.assertEqual(list(Path(directory).iterdir()), [])
            command[command.index("colab")] = "local"
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--submit requires --runtime colab", result.stderr)

    def test_exported_kernel_job_and_download_preserve_failed_trainer_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runtime = load_named("colab", "runtimes")
            runtime.update(persistent_root=None, run_root=str(directory / "remote_runs"))
            plan = resolve(load_named("cartpole_flashsac_smoke", "experiments"), runtime, run_id="failed")
            bundle_dir = prepare(plan, directory / "bundle", bundle_data=True, include_notebook=False)
            self.assertFalse((bundle_dir / "train_colab.ipynb").exists())
            uploads = directory / "uploads"
            uploads.mkdir()
            files = []
            names = ["bundle.json", "experiment.json", "runtime.json", "inputs.json"]
            names += list(read_json(bundle_dir / "bundle.json")["archives"])
            for name in names:
                path = uploads / name
                shutil.copyfile(bundle_dir / name, path)
                files.append(dict(name=name, sha256=sha256(path), parts=[str(path)]))
            checkpoint = plan.run_dir / "training/checkpoints/step_1"
            stub = ("from pathlib import Path; import time; "
                    f"p=Path({str(checkpoint)!r}); p.mkdir(parents=True); "
                    "(p/'model.pt').write_bytes(b'completed checkpoint'); "
                    "(p.parent/'.unfinished').mkdir(); "
                    "print('diagnostic from failed trainer',flush=True); time.sleep(.1); raise SystemExit(3)")
            real_popen = subprocess.Popen

            def popen(argv, **kwargs):
                if "train" in argv:
                    return real_popen([sys.executable, "-c", stub], **kwargs)
                return real_popen([sys.executable, "-c", "print('hardware/setup stub')"], **kwargs)

            work = directory / "work"
            with patch.object(colab_job.subprocess, "Popen", side_effect=popen), \
                 patch.object(colab_job.shutil, "which", return_value="uv"):
                colab_job.run(dict(work=str(work), files=files, sync_interval_seconds=.02))
            destination = directory / "local_results"

            class DownloadClient:
                def call(self, command, remote, local, **kwargs):
                    shutil.copyfile(work / "results.zip", local)
                    return subprocess.CompletedProcess([], 0, "", "")

            final = colab_submit.retrieve(DownloadClient(), str(work), destination)
            self.assertEqual(final["state"], "failed")
            self.assertEqual(final["exit_code"], 3)
            self.assertEqual((destination / "training/checkpoints/step_1/model.pt").read_bytes(), b"completed checkpoint")
            self.assertFalse((destination / "training/checkpoints/.unfinished").exists())
            self.assertIn("diagnostic from failed trainer", (destination / "colab_job.log").read_text())

    def test_upload_corruption_is_reported_before_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            part = directory / "part"
            part.write_bytes(b"corrupt")
            work = directory / "work"
            colab_job.run(dict(work=str(work), files=[dict(name="bundle.json", parts=[str(part)], sha256="wrong")]))
            with zipfile.ZipFile(work / "results.zip") as archive:
                final = json.loads(archive.read("colab_job.json"))
                self.assertEqual(final["state"], "failed")
                self.assertIn("Upload checksum differs", final["error"])

    def test_submit_returns_remote_failure_and_releases_its_dedicated_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runtime = load_named("colab", "runtimes")
            plan = resolve(load_named("cartpole_flashsac_smoke", "experiments"), runtime, run_id="failure")
            bundle = directory / "bundle"
            calls = []

            def fake_prepare(plan, output, **kwargs):
                self.assertIsNone(plan.runtime["persistent_root"])
                bundle.mkdir()
                for name in ("bundle.json", "experiment.json", "runtime.json", "inputs.json"):
                    (bundle / name).write_text(json.dumps({"archives": {}} if name == "bundle.json" else {}))
                return bundle

            def fake_call(client, command, *args, **kwargs):
                calls.append((client.session, command))
                return subprocess.CompletedProcess([], 0, "", "")

            final = {"state": "failed", "exit_code": 7}
            with patch.object(colab_submit.shutil, "which", return_value="colab"), \
                 patch.object(colab_submit, "prepare", side_effect=fake_prepare), \
                 patch.object(colab_submit.Client, "call", fake_call), \
                 patch.object(colab_submit, "retrieve", return_value=final):
                code = colab_submit.execute(plan, bundle, results_root=directory / "results")
            self.assertEqual(code, 7)
            saved = read_json(bundle / "submission.json")
            self.assertEqual(saved["state"], "failed")
            self.assertTrue(saved["resource_released"])
            self.assertEqual(calls[-1], (saved["session"], "stop"))
            self.assertEqual({session for session, command in calls}, {saved["session"]})


if __name__ == "__main__":
    unittest.main()
