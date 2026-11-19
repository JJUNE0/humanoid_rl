"""Resolve portable experiment/runtime JSON without importing a simulator."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from humanoid_rl.algorithm.configuration import merge_config
from humanoid_rl.methods import get_method

ROOT = Path(__file__).resolve().parents[3]
TOKEN = re.compile(r"\$\{([a-z_]+)\}")


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_named(value, category, root=ROOT):
    path = Path(value).expanduser()
    if not path.is_file():
        path = root / "src/humanoid_rl/configs" / category / (str(value) + ".json")
    return read_json(path)


def expand(value, variables):
    if isinstance(value, str):
        def replace(match):
            if match[1] not in variables or variables[match[1]] is None:
                raise ValueError(f"Undefined path variable ${{{match[1]}}}")
            return str(variables[match[1]])
        return TOKEN.sub(replace, value)
    if isinstance(value, dict):
        return {k: expand(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, variables) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Configuration numbers must be finite")
    return value


def apply_override(config, expression):
    key, separator, raw = expression.partition("=")
    parts = key.split(".")
    if not separator or any(not re.fullmatch(r"[a-z][a-z0-9_]*", p) for p in parts):
        raise ValueError("Overrides use dotted method settings, e.g. training.num_envs=64")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ValueError(f"Override crosses a scalar setting: {key}")
    target[parts[-1]] = value


def validate_keys(value, allowed, label):
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")


@dataclass
class Plan:
    root: Path
    experiment: dict
    runtime: dict
    settings: dict
    paths: dict
    run_id: str
    python: str
    inputs: list
    resume: Path | None = None
    overrides: tuple = ()

    @property
    def method(self):
        return get_method(self.experiment["method"], self.task, self.experiment.get("robot"))

    @property
    def task(self):
        return self.experiment["task"]

    @property
    def run_dir(self):
        return Path(self.paths["run_dir"])

    @property
    def env(self):
        return expand(self.runtime.get("env", {}), self.paths)

    def snapshot(self):
        return dict(version=1, experiment=self.experiment, runtime=self.runtime,
                    settings=self.settings, paths=self.paths, run_id=self.run_id,
                    python=self.python, resume=str(self.resume) if self.resume else None,
                    overrides=list(self.overrides), inputs=self.inputs, command=self.method.adapter().command(self))


def resolve(experiment, runtime, *, root=ROOT, overrides=(), run_id=None, resume=None, python=None):
    root = Path(root).resolve()
    experiment, runtime = deepcopy(experiment), deepcopy(runtime)
    validate_keys(experiment, ("version", "name", "method", "task", "robot", "description", "config", "inputs"), "experiment")
    validate_keys(runtime, ("version", "name", "data_root", "cache_root", "run_root", "persistent_root",
                            "device", "overrides", "env", "sync_interval_seconds"), "runtime")
    if experiment.get("version") != 1 or runtime.get("version") != 1:
        raise ValueError("Experiment and runtime must use version: 1")
    for value in (experiment.get("name", ""), runtime.get("name", "")):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
            raise ValueError("Experiment/runtime names must contain only letters, digits, '_' or '-'")
    method = get_method(experiment["method"], experiment["task"], experiment.get("robot"))
    if runtime["name"] not in ("local", "colab"):
        raise ValueError("Runtime name must be local or colab")
    if not isinstance(runtime.get("env", {}), dict) or any(not isinstance(v, str) for v in runtime.get("env", {}).values()):
        raise ValueError("runtime.env must map environment variable names to strings")
    config = experiment.setdefault("config", {})
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("Run ID must contain only letters, digits, '_' or '-'")
    paths = {"repo": str(root)}
    for name, default in (("data_root", "${repo}"), ("cache_root", "${repo}/data/cache"),
                          ("run_root", "${repo}/runs"), ("persistent_root", None)):
        value = runtime.get(name, default)
        if value is None:
            paths[name] = None
        else:
            path = Path(expand(value, paths)).expanduser()
            paths[name] = str((root / path).resolve())
    paths["run_dir"] = str(Path(paths["run_root"]) / experiment["name"] / run_id)
    merged = merge_config(config, runtime.get("overrides", {}).get(method.name, {}))
    for expression in overrides:
        apply_override(merged, expression)
    settings = expand(method.adapter().settings(experiment["task"], expand(merged, paths)), paths)
    training = settings.setdefault("training", {})
    if set(training) & {"task", "algorithm", "train_config", "resume"}:
        raise ValueError("task/method/resume must be selected through the common launcher")
    training["device"] = runtime.get("device", "cuda:0")
    if method.environment == "supermimic":
        # The preserved trainer's ARTIFACT_ROOT defaults follow the data root.
        runtime.setdefault("env", {}).setdefault("SUPER_MIMIC_HOME", "${data_root}/src/super_mimic/artifacts")
    else:
        training["log_dir"] = str(Path(paths["run_dir"]) / "training")
    expand(runtime.get("env", {}), paths)
    if method.name == "bfm_zero":
        motion = settings["motion"]
        if motion.get("manifest"):
            motion.setdefault("manifest_root", paths["data_root"])
        for key in ("root", "manifest", "manifest_root"):
            if motion.get(key) is not None:
                motion[key] = str((Path(paths["data_root"]) / Path(motion[key]).expanduser()).resolve())
    resume = Path(resume).expanduser().absolute() if resume else None
    if resume is not None and method.name == "bfm_zero":
        settings = method.adapter().resume_settings(settings, resume, experiment["task"], overrides)
    interval = runtime.get("sync_interval_seconds", 60)
    if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval < 1:
        raise ValueError("sync_interval_seconds must be a positive finite number")
    environment = read_json(root / "src/humanoid_rl/environments" / (method.environment + ".json"))
    executable = Path(python).expanduser() if python else root / environment["venv"] / "bin/python"
    inputs = method.adapter().inputs(settings) + expand(experiment.get("inputs", []), paths)
    unique_inputs = {}
    for item in inputs:
        validate_keys(item, ("path", "kind", "root"), "input")
        if item.get("kind", "file") not in ("file", "directory", "motion_manifest"):
            raise ValueError(f"Unknown input kind: {item['kind']}")
        path = Path(item["path"]).expanduser()
        item = dict(item, path=str((root / path).resolve()))
        if "root" in item:
            item["root"] = str((root / Path(item["root"]).expanduser()).resolve())
        unique_inputs[item["path"]] = item
    return Plan(root, experiment, runtime, settings, paths, run_id, str(executable.absolute()),
                list(unique_inputs.values()), resume,
                tuple(overrides))
