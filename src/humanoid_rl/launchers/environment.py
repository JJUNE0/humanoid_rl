"""Install/probe method environments without merging incompatible dependencies."""

import json
import os
from pathlib import Path
import shutil
import subprocess

from humanoid_rl.launchers.config import read_json
from humanoid_rl.methods import get_method


def setup_commands(root, method_name, task=None):
    method = get_method(method_name, task)
    profile = read_json(root / "environments" / (method.environment + ".json"))
    uv = shutil.which("uv") or "uv"
    python = root / profile["venv"] / "bin/python"
    if method.environment == "supermimic_light":
        commands = []
        for relative, source in read_json(root / "environments/supermimic_sources.json").items():
            destination = root / relative
            if not (destination / "pyproject.toml").is_file():
                commands += [["git", "clone", "--no-checkout", "--filter=blob:none", source["url"], str(destination)],
                             ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", source["revision"]],
                             ["git", "-C", str(destination), "checkout", "--detach", source["revision"]]]
        return commands + [["env", f"UV_PROJECT_ENVIRONMENT={python.parent.parent}", uv, "sync", "--locked", "--no-dev",
                            "--project", str(root / "environments/light_supermimic"), "--python", profile["python"]]]
    if method.environment == "mjlab":
        command = [uv, "sync", "--locked", "--project", str(root), "--python", profile["python"]]
        for extra in profile.get("extras", []):
            command += ["--extra", extra]
        return [command]
    commands = []
    if not python.is_file():
        commands.append([uv, "venv", "--python", profile["python"], str(python.parent.parent)])
    commands += [
        [uv, "pip", "sync", "--python", str(python), str(root / "src/super_mimic/requirements-lock.txt")],
        [uv, "pip", "install", "--python", str(python), "--no-deps", "-e", str(root / "src/super_mimic")],
    ]
    return commands


def setup(root, method, task=None):
    env = os.environ.copy()
    # uv must target the selected repository environment, not an activated shell
    # environment or an unrelated uv workspace.
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env.pop("VIRTUAL_ENV", None)
    for command in setup_commands(root, method, task):
        print(" ".join(command), flush=True)
        subprocess.run(command, cwd=root, env=env, check=True)


def probe(plan):
    profile = read_json(plan.root / "environments" / (plan.method.environment + ".json"))
    hint = f"python humanoid.py setup --method {plan.method.name} --task {plan.task}"
    if not Path(plan.python).is_file():
        raise ValueError(f"Method interpreter is missing: {plan.python}. "
                         f"Run {hint}")
    distributions = {n: n for n in ("torch", "mjlab", "mujoco", "mujoco-warp")}
    distributions["mujoco"] = profile.get("mujoco_distribution", "mujoco")
    code = ("import importlib.metadata as m,json,sys; "
            "print(json.dumps({'python':list(sys.version_info[:3]),"
            f"'packages':{{n:m.version(d) for n,d in {distributions!r}.items()}}}}))")
    result = subprocess.run([plan.python, "-c", code], capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"Incomplete method environment: {plan.python}\n{result.stderr.strip()}\n"
                         f"Run {hint}")
    info = json.loads(result.stdout)
    version = ".".join(str(v) for v in info["python"][:2])
    if version != profile["python"] or info["packages"]["mjlab"] != profile["mjlab"]:
        raise ValueError(f"Environment does not match {plan.method.environment}: {info}. "
                         f"Run {hint}")
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True)
        if gpu.returncode == 0:
            info["gpus"] = gpu.stdout.strip().splitlines()
    return info
