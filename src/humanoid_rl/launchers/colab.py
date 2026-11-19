"""Prepare a portable Colab bundle; no remote job is submitted by this module."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import zipfile

from humanoid_rl.launchers.config import write_json
from humanoid_rl.launchers.data import collect_inputs
from humanoid_rl.launchers.source import sha256, snapshot


def portable(value, plan):
    if isinstance(value, dict):
        return {k: portable(v, plan) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v, plan) for v in value]
    if isinstance(value, str) and Path(value).is_absolute():
        roots = sorted(((plan.paths[key], key) for key in ("data_root", "repo", "cache_root", "run_root")),
                       key=lambda pair: len(pair[0]), reverse=True)
        for root, key in roots:
            if Path(value).is_relative_to(root):
                relative = Path(value).relative_to(root).as_posix()
                return "${" + key + "}" + ("/" + relative if relative != "." else "")
        raise ValueError(f"External path cannot be exported portably: {value}. Use paths under --data-root.")
    return value


def portable_overrides(plan):
    result = []
    for expression in plan.overrides:
        key, _, raw = expression.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        result.append(key + "=" + json.dumps(portable(value, plan)))
    return result


def notebook(bundle_dir="/content/drive/MyDrive/humanoid_rl/colab/REPLACE_WITH_BUNDLE_FOLDER"):
    def cell(kind, source):
        result = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
        if kind == "code":
            result.update(execution_count=None, outputs=[])
        return result

    cells = [
        cell("markdown", "# Humanoid training: local and Colab share the same experiment\n\n"
             "Select a GPU runtime. Upload the exported bundle folder to Drive and set BUNDLE_DIR below. "
             "The bundle contains the current working source, including uncommitted changes. "
             "Use `--bundle-data` when exporting to include the required motions/caches. "
             "Training uses the VM disk; completed checkpoints are copied to Drive periodically.\n"),
        cell("code", "from google.colab import drive\ndrive.mount('/content/drive')\n"
             "from pathlib import Path\n" + f"BUNDLE_DIR = Path({str(bundle_dir)!r})\n"),
        cell("code", "import hashlib, json, subprocess, sys, zipfile\n"
             "bundle = json.loads((BUNDLE_DIR / 'bundle.json').read_text())\n"
             "REPO = Path('/content') / ('humanoid_rl_' + bundle['run_id'])\n"
             "REPO.mkdir(parents=True, exist_ok=True)\n"
             "def extract_verified(name, destination):\n"
             "    path = BUNDLE_DIR / name\n"
             "    digest = hashlib.sha256()\n"
             "    with path.open('rb') as stream:\n"
             "        for block in iter(lambda: stream.read(1048576), b''):\n"
             "            digest.update(block)\n"
             "    if digest.hexdigest() != bundle['archives'][name]:\n"
             "        raise ValueError('Archive checksum differs: ' + name)\n"
             "    with zipfile.ZipFile(path) as archive:\n"
             "        for info in archive.infolist():\n"
             "            if not (destination / info.filename).resolve().is_relative_to(destination.resolve()):\n"
             "                raise ValueError('Archive path escapes destination')\n"
             "        archive.extractall(destination)\n"
             "extract_verified('source.zip', REPO)\n"
             "if 'data.zip' in bundle['archives']:\n"
             "    extract_verified('data.zip', REPO)\n"
             "if 'resume.zip' in bundle['archives']:\n"
             "    extract_verified('resume.zip', REPO / '_resume')\n"
             "subprocess.run(['nvidia-smi'], check=True)\n"),
        cell("code", "import os\n"
             "subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)\n"
             "subprocess.run([sys.executable, '-m', 'humanoid_rl', 'setup',\n"
             "                '--method', bundle['method'], '--task', bundle['task']],\n"
             "               cwd=REPO, env=os.environ | {'PYTHONPATH': str(REPO / 'src')}, check=True)\n"),
        cell("markdown", "## Run or resume\n\n"
             "The experiment includes CLI overrides from export. The exact resolved configuration, "
             "environment versions and source fingerprints are saved in the run folder. "
             "If data.zip was omitted, copy the inputs listed in inputs.json under REPO first. "
             "For a later session, set RESUME to a checkpoint copied back from Drive. "
             "Choose a new RUN_ID for each invocation; the saved weights continue into a new run folder.\n"),
        cell("code", "import os\n"
             "RUN_ID = bundle['run_id']\n"
             "RESUME = str(REPO / '_resume' / bundle['resume']) if bundle.get('resume') else None\n"
             "command = [sys.executable, '-m', 'humanoid_rl', 'train',\n"
             "           '--experiment', str(BUNDLE_DIR / 'experiment.json'),\n"
             "           '--runtime', str(BUNDLE_DIR / 'runtime.json'), '--run-id', RUN_ID]\n"
             "if RESUME:\n"
             "    command += ['--resume', RESUME]\n"
             "for override in bundle.get('overrides', []):\n"
             "    command += ['--set', override]\n"
             "subprocess.run(command, cwd=REPO, env=os.environ | {'PYTHONPATH': str(REPO / 'src')}, check=True)\n"),
    ]
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                 language_info=dict(name="python"), accelerator="GPU"), nbformat=4, nbformat_minor=5)


def prepare(plan, output, *, bundle_data=False, include_notebook=True):
    output = Path(output).expanduser().absolute()
    if output.exists():
        raise ValueError(f"Bundle directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".colab-", dir=output.parent) as temporary:
        directory = Path(temporary) / "bundle"
        directory.mkdir()
        source = snapshot(plan.root, directory / "source.zip")
        archives = {"source.zip": sha256(directory / "source.zip")}
        input_list = []
        for item in plan.inputs:
            portable_input = dict(item, path=Path(item["path"]).relative_to(plan.paths["data_root"]).as_posix())
            if "root" in portable_input:
                portable_input["root"] = Path(item["root"]).relative_to(plan.paths["data_root"]).as_posix()
            input_list.append(portable_input)
        write_json(directory / "inputs.json", input_list)
        data_files = {}
        if bundle_data:
            files, data_files = collect_inputs(plan)
            with zipfile.ZipFile(directory / "data.zip", "w", compression=zipfile.ZIP_STORED) as archive:
                for relative, path in files.items():
                    archive.write(path, relative)
            archives["data.zip"] = sha256(directory / "data.zip")
        resume = None
        if plan.resume:
            plan.method.adapter().check_resume(plan.resume, plan.task)
            path = plan.resume
            if plan.method.environment == "supermimic":
                path = plan.method.adapter().checkpoint_file(path)
            resume = path.name
            paths = sorted(path.rglob("*")) if path.is_dir() else [path]
            with zipfile.ZipFile(directory / "resume.zip", "w", compression=zipfile.ZIP_STORED) as archive:
                for child in paths:
                    if child.is_file():
                        archive.write(child, (Path(resume) / child.relative_to(path)).as_posix() if path.is_dir() else resume)
            archives["resume.zip"] = sha256(directory / "resume.zip")
        runtime = deepcopy(plan.runtime)
        # Data archives preserve paths relative to the local data root. Extract
        # them under the fresh checkout, regardless of the exporting host path.
        runtime["data_root"] = "${repo}"
        write_json(directory / "experiment.json", portable(plan.experiment, plan))
        write_json(directory / "runtime.json", runtime)
        write_json(directory / "bundle.json", dict(version=1, method=plan.method.name, run_id=plan.run_id,
                   experiment=plan.experiment["name"], task=plan.task, source_sha256=source["source_sha256"],
                   archives=archives, data_files=data_files, resume=resume, overrides=portable_overrides(plan)))
        if include_notebook:
            drive_dir = "/content/drive/MyDrive/humanoid_rl/colab/" + output.name
            write_json(directory / "train_colab.ipynb", notebook(drive_dir))
            (directory / "README.txt").write_text(
                "Upload this entire folder to MyDrive/humanoid_rl/colab/" + output.name + "\n"
                "Open train_colab.ipynb in Colab, choose a GPU runtime, and run its cells.\n"
                "BUNDLE_DIR in the notebook must point to the uploaded folder.\n"
                "No remote training job has been submitted by this export.\n"
                "Required input paths are listed in inputs.json.\n")
        else:
            (directory / "README.txt").write_text(
                "Direct Colab CLI submission; no notebook or Drive upload is required.\n"
                "See submission.json for execution status, local result location and VM cleanup.\n"
                "Required input paths are listed in inputs.json.\n")
        directory.rename(output)
    return output
