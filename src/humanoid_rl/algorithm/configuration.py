"""Load algorithm-local JSON defaults and optional experiment overrides."""

from copy import deepcopy
import json
from pathlib import Path


ALGORITHMS_DIR = Path(__file__).resolve().parent


def merge_config(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_train_config(algorithm, path=None, *, task=None):
    default = ALGORITHMS_DIR/algorithm/"configs/train.json"
    values = json.loads(default.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"Training config must contain a JSON object: {default}")
    if algorithm == "bfm_zero" and task == "G1-Motion-Prior":
        values = merge_config(values,json.loads((default.parent/"g1.json").read_text()))
    if algorithm == "bfm_zero" and task == "Humanoid-Light-Motion-Prior":
        values = merge_config(values,json.loads((default.parent/"humanoid_light.json").read_text()))
    if path is not None and Path(path).resolve() != default:
        override = json.loads(Path(path).read_text())
        if not isinstance(override, dict):
            raise ValueError(f"Training config must contain a JSON object: {path}")
        values = merge_config(values, override)
    return values
