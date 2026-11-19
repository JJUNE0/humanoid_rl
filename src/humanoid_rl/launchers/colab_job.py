"""Stdlib-only job sent to a Colab kernel by the direct launcher."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback
import zipfile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def extract(path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not (destination / info.filename).resolve().is_relative_to(destination):
                raise ValueError("Archive path escapes destination")
        archive.extractall(destination)


def archived_checkpoints(root, keep=2):
    """Bound each transfer while keeping prior checkpoints on the VM and PC."""
    selected = set()
    for directory in Path(root).rglob("checkpoints"):
        children = [p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")]
        selected.update(sorted(children, key=lambda p: p.stat().st_mtime_ns)[-keep:])
    return selected


def run(job):
    work = Path(job["work"])
    work.mkdir(parents=True, exist_ok=False)
    mirror = work / "results"
    mirror.mkdir()
    status = {"state": "setup", "exit_code": None}
    source = None
    synchronizer = None
    stop = threading.Event()
    thread = None
    log = (mirror / "colab_job.log").open("w", buffering=1)

    def emit(text):
        log.write(text)
        print(text, end="", flush=True)

    def command(argv, cwd=None):
        emit("Running: " + repr(argv) + "\n")
        with subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace", bufsize=1) as process:
            for line in process.stdout:
                emit(line)
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, argv)

    def publish():
        if synchronizer is not None and source.is_dir():
            synchronizer.sync()
        status["archive_checkpoint_policy"] = "Latest two checkpoint directories per group; prior VM and PC copies retained"
        (mirror / "colab_job.json").write_text(json.dumps(status, indent=2) + "\n")
        temporary = work / "results.zip.tmp"
        checkpoints = archived_checkpoints(mirror)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(mirror.rglob("*")):
                if path.is_file() and not any(part.startswith(".") for part in path.relative_to(mirror).parts):
                    checkpoint = next((p for p in path.parents if p.parent.name == "checkpoints"), None)
                    if checkpoint is not None and checkpoint not in checkpoints:
                        continue
                    archive.write(path, path.relative_to(mirror).as_posix())
        temporary.replace(work / "results.zip")

    try:
        bundle_dir = work / "bundle"
        bundle_dir.mkdir()
        for item in job["files"]:
            destination = bundle_dir / item["name"]
            with destination.open("wb") as output:
                for part in item["parts"]:
                    with Path(part).open("rb") as stream:
                        shutil.copyfileobj(stream, output)
            if digest(destination) != item["sha256"]:
                raise ValueError("Upload checksum differs: " + item["name"])
            for part in item["parts"]:
                Path(part).unlink()
        bundle = json.loads((bundle_dir / "bundle.json").read_text())
        repo = work / "repo"
        for name, expected in bundle["archives"].items():
            if digest(bundle_dir / name) != expected:
                raise ValueError("Archive checksum differs: " + name)
            extract(bundle_dir / name, repo / "_resume" if name == "resume.zip" else repo)
        sys.path.insert(0, str(repo / "src"))
        from humanoid_rl.launchers.config import read_json, resolve
        from humanoid_rl.launchers.local import RunSync

        runtime = read_json(bundle_dir / "runtime.json")
        if runtime.get("persistent_root") is not None:
            raise ValueError("Direct Colab jobs retrieve results locally; Drive mounting is not required")
        resume = repo / "_resume" / bundle["resume"] if bundle.get("resume") else None
        plan = resolve(read_json(bundle_dir / "experiment.json"), runtime, root=repo,
                       run_id=bundle["run_id"], overrides=bundle.get("overrides", []), resume=resume)
        source = plan.run_dir
        synchronizer = RunSync(source, mirror)
        command(["nvidia-smi"])
        if shutil.which("uv") is None:
            command([sys.executable, "-m", "pip", "install", "uv"])
        command([sys.executable, str(repo / "humanoid.py"), "setup", "--method", bundle["method"],
                 "--task", bundle["task"]], cwd=repo)
        status["state"] = "running"
        publish()

        def backup():
            while not stop.wait(job.get("sync_interval_seconds", 60)):
                try:
                    publish()
                except Exception as error:
                    emit("Colab checkpoint snapshot failed: " + str(error) + "\n")

        thread = threading.Thread(target=backup, daemon=True)
        thread.start()
        argv = [sys.executable, str(repo / "humanoid.py"), "train", "--experiment",
                str(bundle_dir / "experiment.json"), "--runtime", str(bundle_dir / "runtime.json"),
                "--run-id", bundle["run_id"]]
        if resume:
            argv += ["--resume", str(resume)]
        for override in bundle.get("overrides", []):
            argv += ["--set", override]
        command(argv, cwd=repo)
        status.update(state="complete", exit_code=0)
    except BaseException as error:
        status.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      exit_code=130 if isinstance(error, KeyboardInterrupt) else
                      (getattr(error, "returncode", 1) or 1), error=str(error))
        emit(traceback.format_exc())
    finally:
        stop.set()
        if thread:
            thread.join()
        log.flush()
        publish()
        log.close()
    print("Colab job finished: " + json.dumps(status), flush=True)

