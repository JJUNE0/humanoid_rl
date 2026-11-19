"""Cross-runtime dispatch, portable data, persistence, and executable exports."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from humanoid_rl.launchers.colab import prepare
from humanoid_rl.launchers.config import ROOT, load_named, read_json, resolve
from humanoid_rl.launchers.data import collect_inputs
from humanoid_rl.launchers.local import RunSync, execute
from humanoid_rl.launchers.source import sha256, snapshot
from humanoid_rl.methods import get_method


def plan(name="light_bfm_zero", runtime="local", **kwargs):
    return resolve(load_named(name, "experiments"), load_named(runtime, "runtimes"), run_id="verification", **kwargs)


class DispatchTests(unittest.TestCase):
    def test_planning_from_another_directory_needs_no_site_packages_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-S", str(ROOT / "humanoid.py"), "launch",
                                     "--experiment", "light_bfm_zero", "--runtime", "colab",
                                     "--run-root", directory + "/runs", "--dry-run"],
                                    cwd=directory, text=True, capture_output=True, check=True)
            saved = json.loads(result.stdout)
            self.assertEqual(saved["action"], "export_colab")
            self.assertEqual(saved["experiment"]["method"], "bfm_zero")
            self.assertFalse((Path(directory) / "runs").exists())
        code = ("import sys;sys.path.insert(0,'src');from humanoid_rl.cli import inventory;inventory();"
                "assert not {'torch','mujoco','mjlab','super_mimic'} & sys.modules.keys()")
        subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, check=True)

    def test_locations_preserve_learning_settings_and_choose_method_interpreters(self):
        local, remote = plan(), plan(runtime="colab")
        training = dict(local.settings["training"])
        remote_training = dict(remote.settings["training"])
        training.pop("log_dir"); remote_training.pop("log_dir")
        self.assertEqual(training, remote_training)
        for key in ("agent", "model", "environment"):
            self.assertEqual(local.settings[key], remote.settings[key])
        self.assertEqual(local.settings["environment"]["physics_dt"] * local.settings["environment"]["decimation"], .02)
        self.assertIn("/.venv/bin/python", local.python)
        self.assertIn("/.venv_super_mimic/bin/python", plan("g1_supermimic_stand").python)

    def test_override_precedence_validation_and_no_input_mutation(self):
        experiment, runtime = load_named("light_bfm_zero", "experiments"), load_named("local", "runtimes")
        original = deepcopy(experiment)
        runtime["overrides"] = {"bfm_zero": {"training": {"num_envs": 32}}}
        configured = resolve(experiment, runtime, overrides=["training.num_envs=8"], run_id="check")
        self.assertEqual(configured.settings["training"]["num_envs"], 8)
        self.assertEqual(experiment, original)
        with self.assertRaisesRegex(ValueError, "Unknown bfm_zero training"):
            plan(overrides=["training.num_envss=8"])
        with self.assertRaisesRegex(ValueError, "Undefined path variable"):
            plan(overrides=["motion.cache_dir=${typo}"])
        with self.assertRaisesRegex(ValueError, "does not support"):
            get_method("supermimic", "Humanoid-Light-Motion-Prior", "humanoid_light")
        bad = load_named("local", "runtimes"); bad["devcie"] = "cuda:0"
        with self.assertRaisesRegex(ValueError, "Unknown runtime"):
            resolve(experiment, bad)

    def test_resume_dispatch_and_overridden_dataset_are_explicit(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            saved = deepcopy(plan().settings); saved["environment"]["lie_down_probability"] = .15
            (Path(checkpoint) / "run.json").write_text(json.dumps({"algorithm": "bfm_zero",
                "task": "Humanoid-Light-Motion-Prior", "train_config": saved}))
            configured = plan(resume=checkpoint, overrides=["motion.manifest=/tmp/selected.json"])
            command = configured.method.adapter().command(configured)
            self.assertNotIn("--train-config", command)
            self.assertIn("--motion-data", command)
            self.assertIn("--motion-cache", command)
            self.assertIn("--compile", command)
            self.assertEqual(configured.settings["environment"]["lie_down_probability"], .15)
            self.assertEqual(configured.inputs[0]["path"], "/tmp/selected.json")
            self.assertEqual(configured.inputs[0]["kind"], "motion_manifest")
            saved["training"]["compile"] = False
            (Path(checkpoint) / "run.json").write_text(json.dumps({"algorithm": "bfm_zero",
                "task": "Humanoid-Light-Motion-Prior", "train_config": saved}))
            self.assertIn("--no-compile", plan(resume=checkpoint).method.adapter().command(plan(resume=checkpoint)))
            with self.assertRaisesRegex(ValueError, "restores learning settings"):
                plan(resume=checkpoint, overrides=["environment.physics_dt=0.001"])
        sm = plan("g1_supermimic_stand", resume="/tmp/last.pt")
        command = sm.method.adapter().command(sm)
        self.assertEqual(command[:4], [sm.python, "-m", "super_mimic", "train"])
        self.assertIn("--resume", command)


class PortableDataTests(unittest.TestCase):
    def fixture(self, directory):
        directory = Path(directory)
        path = directory / "motions/example.npz"
        path.parent.mkdir(); path.write_bytes(b"unchanged motion contents")
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps({"splits": {"train": [{"path": "motions/example.npz", "sha256": sha256(path)}],
                                                    "validation": []}}))
        configured = plan(overrides=["motion.manifest=" + str(manifest)])
        configured.paths["data_root"] = str(directory)
        configured.inputs[0]["root"] = str(directory)
        return configured, manifest, path

    def test_manifest_bytes_preserved_and_checksum_or_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            configured, manifest, path = self.fixture(directory)
            before = manifest.read_bytes()
            files, hashes = collect_inputs(configured)
            self.assertEqual(set(files), {"manifest.json", "motions/example.npz"})
            self.assertEqual(hashes["motions/example.npz"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(manifest.read_bytes(), before)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                collect_inputs(configured)
            data = json.loads(before); data["splits"]["train"][0]["path"] = "../outside.npz"
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "escapes data_root"):
                collect_inputs(configured)

    def test_export_contains_working_source_and_runs_without_git_or_installed_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data_dir = directory / "data"; data_dir.mkdir()
            configured, manifest, path = self.fixture(data_dir)
            configured.runtime["name"] = "colab"
            configured.overrides = ("motion.manifest=" + str(manifest),)
            output = prepare(configured, directory / "export", bundle_data=True)
            saved = read_json(output / "bundle.json")
            for name, expected in saved["archives"].items():
                self.assertEqual(sha256(output / name), expected)
            self.assertNotIn(str(data_dir), (output / "experiment.json").read_text())
            self.assertIn("${data_root}", saved["overrides"][0])
            with zipfile.ZipFile(output / "data.zip") as archive:
                self.assertEqual(archive.read("motions/example.npz"), path.read_bytes())
                self.assertEqual(archive.read("manifest.json"), manifest.read_bytes())
            extracted = directory / "checkout"
            with zipfile.ZipFile(output / "source.zip") as archive:
                self.assertNotIn(".git/config", archive.namelist())
                self.assertFalse(any("/artifacts/" in name for name in archive.namelist()))
                archive.extractall(extracted)
            result = subprocess.run([sys.executable, "-S", str(extracted / "humanoid.py"), "list", "--json"],
                                    cwd=directory, capture_output=True, text=True, check=True)
            self.assertIn("supermimic", json.loads(result.stdout)["methods"])
            for cell in read_json(output / "train_colab.ipynb")["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), "colab-cell", "exec")
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(configured, output)

    def test_snapshot_excludes_data_and_environment_files_but_keeps_untracked_code(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "src/example"; source.mkdir(parents=True)
            (source / "new.py").write_text("working_change = True\n")
            (source / ".env").write_text("TOKEN=private")
            (source / "model.pt").write_bytes(b"weights")
            (source / "artifacts").mkdir(); (source / "artifacts/old.py").write_text("old")
            output = directory / "source.zip"
            snapshot(directory, output)
            with zipfile.ZipFile(output) as archive:
                self.assertIn("src/example/new.py", archive.namelist())
                self.assertNotIn("src/example/.env", archive.namelist())
                self.assertNotIn("src/example/model.pt", archive.namelist())
                self.assertNotIn("src/example/artifacts/old.py", archive.namelist())


class ExecutionTests(unittest.TestCase):
    def test_subprocess_failure_is_recorded_and_results_are_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            runtime = load_named("local", "runtimes")
            runtime.update(run_root=str(directory / "runs"), persistent_root=str(directory / "saved"))
            configured = resolve(load_named("cartpole_flashsac_smoke", "experiments"), runtime, run_id="test")
            script = ("from pathlib import Path; "
                      f"p=Path({str(configured.run_dir / 'training/checkpoints/step_1')!r}); "
                      "p.mkdir(parents=True); (p/'model.pt').write_bytes(b'complete'); "
                      "print('trainer output', flush=True); raise SystemExit(3)")
            adapter = configured.method.adapter()
            with patch("humanoid_rl.launchers.local.probe", return_value={"test": True}), \
                 patch("humanoid_rl.launchers.source.describe", return_value={"source_sha256": "test"}), \
                 patch.object(adapter, "command", return_value=[sys.executable, "-c", script]):
                self.assertEqual(execute(configured), 3)
            saved = directory / "saved/cartpole_flashsac_smoke/test"
            self.assertEqual(read_json(saved / "launch_status.json")["state"], "failed")
            self.assertEqual(read_json(saved / "launch_status.json")["exit_code"], 3)
            self.assertEqual((saved / "training/checkpoints/step_1/model.pt").read_bytes(), b"complete")
            self.assertIn("trainer output", (saved / "training.log").read_text())
            self.assertTrue((saved / "launch.json").is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                with patch("humanoid_rl.launchers.local.probe", return_value={}):
                    execute(configured)

    def test_sync_excludes_partial_checkpoints_and_publishes_immutable_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "run", Path(directory) / "saved"
            checkpoint = source / "training/checkpoints/step_2"
            checkpoint.mkdir(parents=True); (checkpoint / "model.pt").write_bytes(b"weights")
            partial = checkpoint.parent / ".writing-step3"
            partial.mkdir(); (partial / "model.pt").write_bytes(b"partial")
            (source / "last.pt.tmp").write_bytes(b"partial")
            sync = RunSync(source, destination); sync.sync(); sync.sync()
            self.assertEqual((destination / "training/checkpoints/step_2/model.pt").read_bytes(), b"weights")
            self.assertFalse((destination / "training/checkpoints/.writing-step3").exists())
            self.assertFalse((destination / "last.pt.tmp").exists())
            self.assertFalse(list(destination.rglob(".sync-*")))
            with self.assertRaisesRegex(ValueError, "outside"):
                RunSync(source, source / "nested")


if __name__ == "__main__":
    unittest.main()
