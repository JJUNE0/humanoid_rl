"""Read recorded carry motions and geometry without importing RL or simulators.

These helpers read motions already expressed in G1 joints. They do not retarget
motions between different robot morphologies.
"""

import io
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def scene_description(box_blob, robot_blob):
    """Decode only portable geometry metadata; never import the teacher runtime."""
    import yaml

    box = yaml.safe_load(box_blob)
    position = np.asarray(box["geom_pos"], dtype=np.float32)
    quaternion = np.asarray(box["geom_quat_wxyz"], dtype=np.float32)
    if (position.shape != (3,) or quaternion.shape != (4,)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-4)):
        raise ValueError("Invalid expert box geometry transform")
    xml = ET.fromstring(robot_blob)
    tops = []
    for name in ("initial_box_support", "target_box_support"):
        geom = xml.find(f"./worldbody/geom[@name='{name}']")
        if geom is None or geom.get("type") != "box":
            raise ValueError(f"Object AMP requires a world-space box support: {name}")
        if any(field in geom.attrib for field in ("quat", "euler", "axisangle", "xyaxes", "zaxis")):
            raise ValueError("Object reference supports must be horizontal and axis-aligned")
        top = np.fromstring(geom.attrib["pos"], sep=" ")
        half = np.fromstring(geom.attrib["size"], sep=" ")
        if (top.shape != (3,) or half.shape != (3,) or not np.isfinite(top).all()
                or not np.isfinite(half).all() or not (half > 0).all()):
            raise ValueError(f"Invalid expert support geometry: {name}")
        top[2] += half[2]
        tops.append(top)
    return dict(geom_position=position, geom_quaternion=quaternion,
                source_top=tops[0], target_top=tops[1])


def read_split_lists(root, max_motions=None):
    """Read explicit, disjoint train/validation episode lists."""
    root = Path(root)
    lists = {}
    for split in ("train", "validation"):
        path = root / f"prior_{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"AMP needs the explicit expert split list: {path}")
        paths = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if len(paths) != len(set(paths)):
            raise ValueError(f"Duplicate motions in {path}")
        lists[split] = paths[:max_motions] if max_motions else paths
        if not lists[split]:
            raise ValueError(f"Empty expert split: {path}")
    if set(lists["train"]) & set(lists["validation"]):
        raise ValueError("Expert train and validation motions overlap")
    return lists


def load_motion(source, *, label=None):
    """Load a motion.npz path or bytes; validate the recorded 50 Hz convention."""
    name = label if label is not None else str(source)
    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    with np.load(stream, allow_pickle=False) as archive:
        motion = {key: archive[key] for key in archive.files}
    if str(motion["quaternion_order"].item()) != "wxyz":
        raise ValueError(f"Expected wxyz quaternions in {name}")
    times = motion["time"]
    if len(times) < 2 or not np.allclose(np.diff(times), 0.02, atol=1e-7):
        raise ValueError(f"Expected consecutive 50 Hz states in {name}")
    for key in ("root_quat", "object_quat"):
        if not np.allclose(np.linalg.norm(motion[key], axis=-1), 1, atol=1e-4):
            raise ValueError(f"Non-unit {key} in {name}")
    return motion
