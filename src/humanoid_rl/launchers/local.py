"""Run any method locally (also used inside Colab) and persist its results."""

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import uuid

from humanoid_rl.launchers.config import write_json
from humanoid_rl.launchers.environment import probe


def validate_inputs(plan):
    from humanoid_rl.launchers.data import collect_inputs
    collect_inputs(plan, verify=False)
    if plan.resume:
        plan.method.adapter().check_resume(plan.resume, plan.task)


class RunSync:
    """Copy published checkpoints and changed small files to persistent storage.

    Checkpoint directories are immutable in the existing BFM/FlashSAC trainers.
    Copy each into a hidden directory before publishing it. Temporary training
    files are excluded; ordinary files are also replaced atomically.
    """

    def __init__(self, source, destination):
        self.source, self.destination = Path(source), Path(destination)
        self.signatures = {}
        if self.source == self.destination or self.destination.is_relative_to(self.source):
            raise ValueError("Persistent destination must be outside the active run directory")

    def sync(self):
        self.destination.mkdir(parents=True, exist_ok=True)
        for directory, children, files in os.walk(self.source):
            directory = Path(directory)
            children[:] = [name for name in children if not name.startswith(".")]
            if directory.name == "checkpoints":
                for name in list(children):
                    source = directory / name
                    destination = self.destination / source.relative_to(self.source)
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        temporary = destination.with_name(".sync-" + uuid.uuid4().hex)
                        try:
                            shutil.copytree(source, temporary)
                            temporary.rename(destination)
                        finally:
                            if temporary.exists():
                                shutil.rmtree(temporary)
                    children.remove(name)
            for name in files:
                if name.startswith(".") or name.endswith((".tmp", ".lock")):
                    continue
                source = directory / name
                relative = source.relative_to(self.source)
                try:
                    before = source.stat()
                    signature = before.st_mtime_ns, before.st_size
                    if self.signatures.get(relative) == signature:
                        continue
                    destination = self.destination / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(".sync-" + uuid.uuid4().hex)
                    try:
                        shutil.copy2(source, temporary)
                        after = source.stat()
                        if signature == (after.st_mtime_ns, after.st_size):
                            temporary.replace(destination)
                            self.signatures[relative] = signature
                    finally:
                        temporary.unlink(missing_ok=True)
                except FileNotFoundError:
                    # Checkpoint retention may remove old files during a copy.
                    continue


def execute(plan):
    validate_inputs(plan)
    environment = probe(plan)
    if plan.run_dir.exists():
        raise ValueError(f"Run directory already exists: {plan.run_dir}. Choose another --run-id.")
    persistent = plan.paths.get("persistent_root")
    if plan.runtime["name"] == "colab" and persistent and str(persistent).startswith("/content/drive/"):
        if not Path("/content/drive/MyDrive").is_dir():
            raise ValueError("Mount Google Drive in the notebook before training with the colab profile")
    sync = RunSync(plan.run_dir, Path(persistent) / plan.experiment["name"] / plan.run_id) if persistent else None
    if sync and sync.destination.exists():
        raise ValueError(f"Persistent run already exists: {sync.destination}. Choose a new --run-id to resume.")
    plan.run_dir.mkdir(parents=True)
    write_json(plan.run_dir / "launch.json", plan.snapshot())
    write_json(plan.run_dir / "method.json", plan.settings)
    write_json(plan.run_dir / "environment.json", environment)
    from humanoid_rl.launchers.source import describe
    write_json(plan.run_dir / "source.json", describe(plan.root))
    env = os.environ.copy()
    env.update(plan.env)
    env["PYTHONPATH"] = str(plan.root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    command = plan.method.adapter().command(plan)
    status = dict(state="preparing", started=datetime.now(timezone.utc).isoformat(), command=command)
    status_path = plan.run_dir / "launch_status.json"
    write_json(status_path, status)
    process = None
    reader = None
    log = None
    sync_error = None

    def persist():
        nonlocal sync_error
        if sync:
            try:
                sync.sync()
                sync_error = None
            except OSError as error:
                sync_error = str(error)
                print(f"Persistent copy failed; local results are in {plan.run_dir}: {error}", file=sys.stderr)

    try:
        plan.method.adapter().prepare(plan)
        print(f"Run: {plan.run_dir}\nInterpreter: {plan.python}", flush=True)
        log = (plan.run_dir / "training.log").open("w")
        process = subprocess.Popen(command, cwd=plan.root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1, start_new_session=True)
        status.update(state="running", pid=process.pid)
        write_json(status_path, status)
        persist()

        def output():
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except BrokenPipeError:
                    pass

        reader = threading.Thread(target=output, daemon=True)
        reader.start()
        while True:
            try:
                code = process.wait(timeout=plan.runtime.get("sync_interval_seconds", 60))
                break
            except subprocess.TimeoutExpired:
                persist()
        reader.join()
        status.update(state="complete" if code == 0 else "failed", exit_code=code)
    except KeyboardInterrupt:
        status.update(state="interrupted", exit_code=130)
    except BaseException as error:
        status.update(state="failed", error=str(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        if reader is not None:
            reader.join(timeout=5)
        if process is not None and process.stdout is not None:
            process.stdout.close()
        if log is not None:
            log.close()
        status["finished"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, status)
        persist()
        if sync_error:
            status["persistence_error"] = sync_error
            write_json(status_path, status)
    if sync_error and status["exit_code"] == 0:
        return 1
    return status["exit_code"]
