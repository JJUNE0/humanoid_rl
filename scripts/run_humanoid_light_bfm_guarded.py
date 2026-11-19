"""Run Light preparation, caching and training in a memory-limited service.

The parent uses only the standard library. Each stage has its own process group;
resource guards stop only that group. Run under systemd with MemoryMax <= 24G.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30


def memory_available():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].split()[0])*1024


def cgroup_directory():
    relative = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines()
                    if line.startswith("0::"))
    return Path("/sys/fs/cgroup")/relative.lstrip("/")


def gpu_free():
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True)
    return int(result.stdout.strip())*2**20


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2)+"\n")
    temporary.replace(path)


def stop_stage(process):
    # All descendants inherit this new process group, outside other jobs' groups.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-dir", type=Path, required=True)
    cli.add_argument("--gpu-fraction", type=float, default=.60)
    cli.add_argument("--min-available-gib", type=float, default=12.)
    cli.add_argument("--min-gpu-free-gib", type=float, default=2.)
    cli.add_argument("--train-child", action="store_true", help=argparse.SUPPRESS)
    args = cli.parse_args()
    if not 0 < args.gpu_fraction <= .65:
        cli.error("The shared-GPU allocator fraction must be in (0, 0.65]")
    os.chdir(ROOT)
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.train_child:
        import torch
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=0)
        print(f"PyTorch GPU allocation limit: {args.gpu_fraction:.0%} of GPU 0; "
              "MuJoCo/Warp allocations are monitored separately.", flush=True)
        sys.argv = [str(ROOT/"train.py"), "--algorithm", "bfm_zero",
                    "--task", "Humanoid-Light-Motion-Prior"]
        runpy.run_path(str(ROOT/"train.py"), run_name="__main__")
        return

    group = cgroup_directory()
    limit = (group/"memory.max").read_text().strip()
    if limit == "max" or int(limit) > 24*GIB:
        raise RuntimeError("Refusing to start without a dedicated MemoryMax <= 24G cgroup")
    status = dict(started=datetime.now().isoformat(), state="starting", supervisor_pid=os.getpid(),
                  cgroup=str(group), memory_max_bytes=int(limit), workers=2,
                  gpu_allocator_fraction=args.gpu_fraction,
                  min_available_gib=args.min_available_gib, min_gpu_free_gib=args.min_gpu_free_gib)
    write_json(directory/"status.json", status)
    executable = str(ROOT/".venv/bin/python")
    stages = [
        ("prepare", [executable, "-u", "scripts/prepare_humanoid_light_bfm.py", "--workers", "2"]),
        ("cache", [executable, "-u", "scripts/cache_humanoid_light_bfm.py"]),
        ("train", [executable, "-u", str(Path(__file__).resolve()), "--train-child",
                   "--run-dir", str(directory), "--gpu-fraction", str(args.gpu_fraction)]),
    ]
    active = None

    def interrupted(signum, frame):
        raise InterruptedError(f"Service received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for stage, command in stages:
            # Wait before allocation if another job needs the available memory.
            status.update(state="waiting", stage=stage, command=command)
            write_json(directory/"status.json", status)
            while memory_available() < (args.min_available_gib+int(limit)/GIB)*GIB or (
                    stage == "train" and gpu_free() < 19*GIB):
                print(f"Waiting for headroom before {stage}", flush=True)
                time.sleep(10)
            environment = os.environ.copy()
            environment.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="2",
                               MUJOCO_GL="egl", PYTHONUNBUFFERED="1",
                               TORCHINDUCTOR_COMPILE_THREADS="1", MAX_JOBS="1")
            if stage != "train":
                environment["CUDA_VISIBLE_DEVICES"] = ""
            log = directory/f"{stage}.log"
            with log.open("a", buffering=1) as stream:
                active = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                status.update(state="running", stage=stage, stage_pid=active.pid,
                              stage_started=datetime.now().isoformat(), log=str(log))
                write_json(directory/"status.json", status)
                print(f"Started {stage}: pid={active.pid}, log={log}", flush=True)
                low_ram = low_gpu = 0
                sampled = 0
                cached_gpu = None
                while active.poll() is None:
                    available = memory_available()
                    if stage == "train" and sampled % 5 == 0:
                        cached_gpu = gpu_free()
                    low_ram = low_ram+1 if available < args.min_available_gib*GIB else 0
                    low_gpu = low_gpu+1 if (stage == "train" and cached_gpu is not None
                                           and cached_gpu < args.min_gpu_free_gib*GIB) else 0
                    if sampled % 5 == 0:
                        resource = dict(time=datetime.now().isoformat(), stage=stage,
                                        available_ram_gib=available/GIB,
                                        cgroup_memory_gib=int((group/"memory.current").read_text())/GIB,
                                        gpu_free_gib=cached_gpu/GIB if cached_gpu is not None else None)
                        with (directory/"resources.jsonl").open("a") as resources:
                            resources.write(json.dumps(resource)+"\n")
                        write_json(directory/"resources_latest.json", resource)
                    if low_ram >= 3 or low_gpu >= 3:
                        raise RuntimeError(f"Resource guard stopped only the new {stage} job: "
                                           f"RAM available={available/GIB:.2f} GiB, "
                                           f"GPU free={None if cached_gpu is None else cached_gpu/GIB} GiB")
                    sampled += 1
                    time.sleep(2)
                code = active.wait()
                active = None
                if code:
                    raise RuntimeError(f"{stage} exited with status {code}; see {log}")
                status.update(state="stage_complete", stage_exit_code=code)
                write_json(directory/"status.json", status)
                print(f"Completed {stage}", flush=True)
        status.update(state="complete", finished=datetime.now().isoformat())
        write_json(directory/"status.json", status)
    except BaseException as error:
        if active is not None:
            stop_stage(active)
        status.update(state="stopped", error=str(error), finished=datetime.now().isoformat())
        write_json(directory/"status.json", status)
        raise


if __name__ == "__main__":
    main()
