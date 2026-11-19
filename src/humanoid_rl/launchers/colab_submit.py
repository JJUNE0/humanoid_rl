"""Submit working source to a dedicated Colab VM and retrieve its results."""

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import zipfile

from humanoid_rl.launchers.colab import prepare
from humanoid_rl.launchers.config import read_json, write_json
from humanoid_rl.launchers.local import RunSync
from humanoid_rl.launchers.source import sha256

CHUNK_BYTES = 8 * 1024 * 1024


class Client:
    def __init__(self, executable, state_file, session):
        self.prefix = [executable, "--config", str(state_file)]
        self.session = session

    def call(self, command, *arguments, capture=False, check=True, timeout=None):
        return subprocess.run(self.prefix + [command, "-s", self.session] + list(map(str, arguments)),
                              text=True, capture_output=capture, check=check, timeout=timeout)


def retrieve(client, remote, destination):
    """Publish completed checkpoint directories atomically on the local PC."""
    with tempfile.TemporaryDirectory(prefix=".colab-download-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / "results.zip"
        result = client.call("download", remote + "/results.zip", archive_path, capture=True,
                             check=False, timeout=180)
        if result.returncode:
            raise OSError((result.stdout + result.stderr).strip())
        extracted = temporary / "results"
        from humanoid_rl.launchers.colab_job import extract
        extract(archive_path, extracted)
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise OSError("Colab result archive is incomplete")
        RunSync(extracted, destination).sync()
        return read_json(extracted / "colab_job.json")


def execute(plan, output, *, gpu="T4", results_root=None):
    executable = shutil.which("colab")
    if executable is None:
        raise ValueError("Colab CLI is missing. Install it with: uv tool install google-colab-cli")
    if gpu not in ("T4", "L4", "G4", "A100", "H100"):
        raise ValueError("Unknown Colab GPU: " + str(gpu))
    destination = Path(results_root or plan.root / "runs").expanduser().absolute() / plan.experiment["name"] / plan.run_id
    if destination.exists():
        raise ValueError(f"Local result directory already exists: {destination}. Choose another --run-id.")
    # Direct submission uses the PC for persistence, without a Drive mount.
    plan = replace(plan, runtime=plan.runtime | {"persistent_root": None})
    bundle_dir = prepare(plan, output, bundle_data=True, include_notebook=False)
    destination.mkdir(parents=True)
    session = "humanoid_" + uuid.uuid4().hex[:12]
    state_file = bundle_dir / "cli_sessions.json"
    client = Client(executable, state_file, session)
    remote = "/content/" + session
    status = dict(state="preparing", session=session, gpu=gpu, results=str(destination), resource_released=False)
    status_file = bundle_dir / "submission.json"
    write_json(status_file, status)
    stop = threading.Event()
    thread = None
    allocated = False
    try:
        # The state file belongs only to this invocation. Cleanup never touches other sessions.
        allocated = True
        client.call("new", "--gpu", gpu)
        status["state"] = "uploading"
        write_json(status_file, status)
        files = []
        names = ["bundle.json", "experiment.json", "runtime.json", "inputs.json"]
        names += list(read_json(bundle_dir / "bundle.json")["archives"])
        with tempfile.TemporaryDirectory(prefix=".upload-", dir=bundle_dir) as temporary:
            chunk_path = Path(temporary) / "chunk"
            for name in names:
                path = bundle_dir / name
                parts = []
                with path.open("rb") as stream:
                    for index, block in enumerate(iter(lambda: stream.read(CHUNK_BYTES), b"")):
                        chunk_path.write_bytes(block)
                        remote_part = remote + "_" + name + f".part{index:05d}"
                        client.call("upload", chunk_path, remote_part, capture=True, timeout=180)
                        parts.append(remote_part)
                files.append(dict(name=name, sha256=sha256(path), parts=parts))
                print(f"Colab uploaded: {name} ({path.stat().st_size / 1048576:.1f} MiB)", flush=True)
        job = dict(work=remote, files=files, sync_interval_seconds=plan.runtime.get("sync_interval_seconds", 60))
        script = bundle_dir / "remote_run.py"
        script.write_text(Path(__file__).with_name("colab_job.py").read_text() + "\nrun(" + repr(job) + ")\n")

        def backup():
            while not stop.wait(job["sync_interval_seconds"]):
                try:
                    retrieve(client, remote, destination)
                except (OSError, ValueError, zipfile.BadZipFile, subprocess.SubprocessError):
                    # Setup may not have published an archive yet. Final retrieval is mandatory.
                    continue

        status["state"] = "running"
        write_json(status_file, status)
        print(f"Colab session: {session}\nLocal results: {destination}", flush=True)
        thread = threading.Thread(target=backup, daemon=True)
        thread.start()
        result = client.call("exec", "-f", script, "--timeout", "0", check=False)
        stop.set()
        thread.join()
        final = retrieve(client, remote, destination)
        if final["state"] not in ("complete", "failed", "interrupted"):
            raise OSError("Colab execution ended before a final job status was retrieved")
        code = final["exit_code"] or result.returncode
        status.update(state="complete" if code == 0 else "failed", exit_code=code)
        return code
    except KeyboardInterrupt:
        status.update(state="interrupted", exit_code=130)
        return 130
    except BaseException as error:
        status.update(state="failed", error=str(error), exit_code=getattr(error, "returncode", 1) or 1)
        raise
    finally:
        stop.set()
        if thread:
            thread.join()
        if allocated:
            # Retrieve partial results before releasing the VM, including failure diagnostics.
            if status["state"] != "complete":
                try:
                    retrieve(client, remote, destination)
                except (OSError, ValueError, zipfile.BadZipFile, subprocess.SubprocessError):
                    pass
            try:
                result = client.call("stop", capture=True, check=False, timeout=90)
                status["resource_released"] = result.returncode == 0
                if result.returncode:
                    print(f"Could not release Colab session {session}: {result.stdout} {result.stderr}\n"
                          f"Run: colab --config {state_file} stop -s {session}", file=sys.stderr)
            except (OSError, subprocess.SubprocessError) as error:
                status["cleanup_error"] = str(error)
                print(f"Colab cleanup failed: {error}. Run: colab --config {state_file} stop -s {session}", file=sys.stderr)
        write_json(status_file, status)
