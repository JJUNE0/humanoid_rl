"""Precompute/verify CPU expert features so a mixed Light run can start directly."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def main():
    from humanoid_rl.algorithm.configuration import load_train_config
    from humanoid_rl.algorithm.bfm_zero.config import resolve_agent_config
    from humanoid_rl.algorithm.bfm_zero.dataset import MotionDataset
    from humanoid_rl.tasks.motion_prior.env_cfg import MotionPriorEnvConfig
    from robots.humanoid.humanoid_light.mjlab import load_control
    import torch
    torch.set_num_threads(2)
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--train-config", type=Path)
    args = cli.parse_args()
    values = load_train_config("bfm_zero", args.train_config, task="Humanoid-Light-Motion-Prior")
    profile = resolve_agent_config(values, "Humanoid-Light-Motion-Prior")
    settings = MotionPriorEnvConfig(**values["environment"])
    manifest = Path(values["motion"]["manifest"])
    manifest = manifest if manifest.is_absolute() else ROOT/manifest
    report = dict(manifest=str(manifest.relative_to(ROOT)),
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), splits={})
    for split in ("train", "validation"):
        dataset = MotionDataset(values["motion"] | {"split": split}, settings, load_control(),
                                values["model"]["sequence_length"], expert_device="cpu", agent_cfg=profile)
        report["splits"][split] = dataset.metadata()
        del dataset
    train = set(report["splits"]["train"]["training_motions"])
    validation = set(report["splits"]["validation"]["training_motions"])
    if train & validation:
        raise ValueError("Training and validation contain the same source files")
    cache = Path(values["motion"]["cache_dir"])
    cache = cache if cache.is_absolute() else ROOT/cache
    report["cache_bytes"] = sum(path.stat().st_size for path in cache.glob("*.pt"))
    destination = manifest.parent/"expert_cache.json"
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"cache_bytes": report["cache_bytes"],
                      "split_frames": {key: value["frame_count"] for key, value in report["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
