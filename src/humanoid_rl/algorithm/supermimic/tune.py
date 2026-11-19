"""Compare real simulator/learner throughput, then continue on the same GPU."""

from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading

from humanoid_rl.launchers.config import write_json


def candidate_settings(base, num_envs, *, compile=False):
    settings = deepcopy(base)
    tuning = settings.pop("tuning")
    reference = tuning["reference"]
    wm_batch, policy_batch = tuning["wm_batch_size"], tuning["policy_batch_size"]
    multiplier = wm_batch / reference["wm_batch_size"]
    if multiplier != policy_batch / reference["policy_batch_size"]:
        raise ValueError("WM and policy batch scaling must match to preserve sample ratios")
    updates = reference["updates_per_step"] * num_envs / reference["num_envs"] / multiplier
    if updates < 1 or updates != int(updates):
        raise ValueError("Environment/batch counts must preserve the reference update-to-data ratio")
    settings["runner"]["wm"]["batch_size"] = wm_batch
    settings["runner"]["policy"]["batch_size"] = policy_batch
    settings["training"].update(num_envs=num_envs, updates_per_step=int(updates),
                                compile=compile, cuda_graphs=True)
    return settings


def run(settings, output, resume):
    if resume is None:
        raise ValueError("Throughput tuning requires a checkpoint so all candidates start from the same weights")
    output, resume = Path(output), Path(resume)
    output.mkdir(parents=True, exist_ok=True)
    tuning = settings["tuning"]
    initial = json.loads((resume / "run.json").read_text())["progress"]["env_steps"]
    results = []

    def launch(config, destination, *, benchmark, timeout=None):
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "settings.json"
        write_json(path, config)
        argv = [sys.executable, "-m", "humanoid_rl.algorithm.supermimic.train", "--config", str(path),
                "--output", str(destination), "--resume", str(resume)]
        if benchmark:
            argv.append("--benchmark")
        expired = threading.Event()
        with (destination / "process.log").open("w", buffering=1) as log:
            with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors="replace", bufsize=1, start_new_session=True) as process:
                def terminate():
                    expired.set()
                    try:
                        import os
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                timer = threading.Timer(timeout, terminate) if timeout else None
                if timer:
                    timer.start()
                try:
                    for line in process.stdout:
                        log.write(line)
                        print(line, end="", flush=True)
                    code = process.wait()
                finally:
                    if timer:
                        timer.cancel()
        return code, expired.is_set()

    def measure(num_envs, compile, *, reference=False):
        name = f"env_{num_envs}_" + ("eager_reference" if reference else "compiled" if compile else "graphs")
        print("Tuning candidate: " + name, flush=True)
        if reference:
            config = deepcopy(settings)
            spec = config.pop("tuning")["reference"]
            config["training"].update(num_envs=num_envs, updates_per_step=spec["updates_per_step"],
                                        compile=False, cuda_graphs=False, muon_dtype="bfloat16")
            config["runner"]["wm"]["batch_size"] = spec["wm_batch_size"]
            config["runner"]["policy"]["batch_size"] = spec["policy_batch_size"]
        else:
            config = candidate_settings(settings, num_envs, compile=compile)
        count = config["training"]["warmup_steps"] + 8 + tuning["measure_steps"]
        config["training"].update(total_env_steps=initial + num_envs * count,
                                   save_every=2**62, log_every=4)
        destination = output / "benchmark" / name
        code, timed_out = launch(config, destination, benchmark=True,
                                 timeout=tuning.get("compile_timeout_seconds", 600) if compile else tuning.get("candidate_timeout_seconds", 900))
        result = dict(name=name, num_envs=num_envs, compile=compile, reference=reference, exit_code=code, timed_out=timed_out)
        performance = destination / "performance.json"
        if code == 0 and performance.exists():
            result.update(json.loads(performance.read_text()))
            result["eligible"] = result["gpu_free_mib"] / result["gpu_total_mib"] >= tuning.get("minimum_free_memory_fraction", .12)
        else:
            result["eligible"] = False
            text = (destination / "process.log").read_text()
            if (destination / "nonfinite_physics.pt").exists() or "Nonfinite" in text:
                raise FloatingPointError(f"Nonfinite state during {name}; stopped tuning rather than ignoring the failure")
        results.append(result)
        write_json(output / "tuning.json", dict(state="benchmarking", candidates=results))
        print("Tuning result: " + json.dumps(result), flush=True)
        return result

    if tuning.get("measure_reference", True):
        measure(tuning["reference"]["num_envs"], False, reference=True)
    for num_envs in tuning["env_counts"]:
        result = measure(num_envs, False)
        if not result["eligible"] and result["exit_code"] == 0:
            break
        if not result["eligible"] and result["exit_code"] != 0:
            if any(r["eligible"] and not r["reference"] for r in results):
                break
    eligible = [r for r in results if r["eligible"] and not r["reference"]]
    if not eligible:
        raise RuntimeError("No GPU configuration passed the throughput and memory checks; inspect benchmark logs")
    best = max(eligible, key=lambda r: r["env_steps_per_second"])
    estimated_seconds = (settings["training"]["total_env_steps"] - initial) / best["env_steps_per_second"]
    if tuning.get("try_compile", True) and estimated_seconds >= tuning.get("compile_min_training_seconds", 0):
        compiled = measure(best["num_envs"], True)
        if compiled["eligible"] and compiled["env_steps_per_second"] > best["env_steps_per_second"]:
            best = compiled
    selected = candidate_settings(settings, best["num_envs"], compile=best["compile"])
    target = selected["training"]["total_env_steps"]
    selected["training"]["total_env_steps"] = initial + ((target - initial + best["num_envs"] - 1) // best["num_envs"]) * best["num_envs"]
    write_json(output / "selected_settings.json", selected)
    write_json(output / "tuning.json", dict(state="selected", candidates=results, selected=best,
               training_target=selected["training"]["total_env_steps"], reference=tuning["reference"],
               estimated_training_seconds=estimated_seconds,
               compile_min_training_seconds=tuning.get("compile_min_training_seconds", 0)))
    print("Selected training configuration: " + json.dumps(best), flush=True)
    code, _ = launch(selected, output, benchmark=False)
    if code:
        raise subprocess.CalledProcessError(code, "selected Light SuperMimic trainer")
