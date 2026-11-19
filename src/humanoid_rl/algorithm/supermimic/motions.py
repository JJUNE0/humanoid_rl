"""Convert screened Light intervals to official SuperMimic root/joint arrays.

Each accepted interval/chunk remains an independent clip. Source motion files
and the screening manifest are immutable; no padded/interpolated labels are
introduced. MuJoCo free-joint angular qvel is rotated from body to world axes.
"""

from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from humanoid_retarget.datasets.robot_motion import load_robot_motion
from humanoid_rl.launchers.source import sha256
from robots.humanoid.humanoid_light.config import ASSET_DIR, JOINT_NAMES

FIELDS = {"root_pos_w": 3, "root_quat_w": 4, "root_lin_vel_w": 3, "root_ang_vel_w": 3,
          "joint_pos": 26, "joint_vel": 26}


def rotate(q, v):
    uv = 2 * np.cross(q[:, 1:], v)
    return v + q[:, :1] * uv + np.cross(q[:, 1:], uv)


def prepare(settings):
    root, manifest_path = Path(settings["root"]), Path(settings["manifest"])
    contents = manifest_path.read_bytes()
    manifest = json.loads(contents)
    if manifest.get("version") != 1 or manifest.get("robot") != "humanoid_light":
        raise ValueError("Expected a version-1 Humanoid Light screening manifest")
    split = settings.get("split", "train")
    if split != "train":
        raise ValueError("Training only accepts the train split; validation remains held out")
    grouped = defaultdict(list)
    for record in manifest["splits"][split]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Motion record escapes its data root")
        grouped[path].append(record)
    paths = sorted(grouped)
    if settings.get("limit") is not None:
        paths = paths[:settings["limit"]]
    if not paths:
        raise ValueError("No accepted training motions")
    chunks = []
    chunk_frames = settings.get("chunk_frames", 501)
    if not isinstance(chunk_frames, int) or chunk_frames < 2:
        raise ValueError("motion.chunk_frames must be >= 2")
    for path in paths:
        checksum = sha256(path)
        if any(r["sha256"] != checksum for r in grouped[path]):
            raise ValueError(f"Screened motion changed: {path}")
        for record in grouped[path]:
            start, stop = record["start_frame"], record["stop_frame"]
            if settings.get("max_frames") is not None:
                stop = min(stop, start + settings["max_frames"])
            for first in range(start, stop, chunk_frames):
                last = min(first + chunk_frames, stop)
                if last - first >= 2:
                    chunks.append((path, first, last, record))
    if not chunks:
        raise ValueError("Accepted motions contain no complete transitions")
    digest = hashlib.sha256(b"light-supermimic-packed-v1" + contents)
    digest.update(json.dumps({k: settings.get(k) for k in ("split", "limit", "max_frames", "chunk_frames")}, sort_keys=True).encode())
    digest.update((ASSET_DIR / "humanoid_light.xml").read_bytes())
    fingerprint = digest.hexdigest()
    cache = Path(settings["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / fingerprint
    with (cache / (fingerprint + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / "index.json").is_file():
            return destination, json.loads((destination / "index.json").read_text())
        with tempfile.TemporaryDirectory(prefix=".packing-", dir=cache) as folder:
            temporary = Path(folder) / "motions"
            temporary.mkdir()
            total = sum(stop - start for _, start, stop, _ in chunks)
            arrays = {name: np.lib.format.open_memmap(temporary / (name + ".npy"), mode="w+",
                      dtype=np.float32, shape=(total, dim)) for name, dim in FIELDS.items()}
            offset, ends, names, sources = 0, [], [], []
            previous, motion = None, None
            for path, start, stop, record in chunks:
                if path != previous:
                    motion = load_robot_motion(path)
                    if motion.joint_names != ("root",) + JOINT_NAMES or not np.isclose(motion.fps, 50):
                        raise ValueError(f"Motion joint order/frequency differs: {path}")
                    previous = path
                if not any(left <= start < stop <= right for left, right in zip(motion.split_points[:-1], motion.split_points[1:])):
                    raise ValueError(f"Screened interval crosses an original trajectory boundary: {path}")
                q, v = motion.qpos[start:stop], motion.qvel[start:stop]
                count = stop - start
                fields = (q[:, :3], q[:, 3:7], v[:, :3], rotate(q[:, 3:7], v[:, 3:6]), q[:, 7:], v[:, 6:])
                for name, value in zip(FIELDS, fields):
                    arrays[name][offset:offset + count] = value
                offset += count
                ends.append(offset)
                names.append(f"{record['source']}/{path.stem}/{start}-{stop}")
                sources.append(dict(path=record["path"], start=start, stop=stop))
            for array in arrays.values():
                array.flush()
            arrays.clear()
            np.save(temporary / "motion_ends.npy", np.asarray(ends, np.int64))
            (temporary / "joint_names.json").write_text(json.dumps(list(JOINT_NAMES)))
            (temporary / "motion_names.json").write_text(json.dumps(names))
            metadata = dict(version=1, fingerprint=fingerprint, split=split, frames=total,
                            clips=len(ends), sources=sources, frequency=50,
                            angular_velocity_convention="world axes, rotated from MuJoCo body-frame angular qvel")
            (temporary / "index.json").write_text(json.dumps(metadata, indent=2))
            temporary.rename(destination)
    return destination, metadata
