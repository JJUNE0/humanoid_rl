"""Read CMU Acclaim skeletons and fully-specified angle motions.

Column-vector convention: a bone's local rotation is C @ M @ C.T;
fixed-axis XYZ rotations use Rz @ Ry @ Rx. ASF bone directions/lengths
describe endpoints in the reference global basis. Units follow CMU's FAQ:
meters = ASF_units * 0.0254 / ASF_length_scale.

References:
http://mocap.cs.cmu.edu/faqs.php
https://graphics.cs.cmu.edu/nsp/course/cs229/info/ACCLAIMdef.html
https://github.com/facebookresearch/fairmotion/blob/main/fairmotion/data/asfamc.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Bone:
    name: str
    parent: str | None
    direction: np.ndarray
    length: float
    axes_degrees: np.ndarray
    dofs: tuple[str, ...]


@dataclass(frozen=True)
class AcclaimMotion:
    names: tuple[str, ...]
    parents: tuple[int, ...]
    positions: np.ndarray  # endpoints; root is the pelvis origin, meters, Z-up
    rotations: np.ndarray  # reference world bases, Z-up
    coordinate_axes: np.ndarray  # each bone's ASF DOF basis in world space
    frame_ids: np.ndarray
    fps: float
    bones: tuple[Bone, ...]
    meter_scale: float
    angles_degrees: dict[str, np.ndarray]


def read_asf(path):
    lines = [line.split("#", 1)[0].strip() for line in Path(path).read_text().splitlines()]
    lines = [line for line in lines if line]
    length_scale, angle_unit, root_order = None, None, None
    root_position, root_orientation = np.zeros(3), np.zeros(3)
    bones, parents = {}, {}
    section, block = None, None
    for line in lines:
        parts = line.split()
        if line.startswith(":"):
            section = parts[0]
            continue
        if section == ":units":
            if parts[0] == "length": length_scale = float(parts[1])
            if parts[0] == "angle": angle_unit = parts[1]
        elif section == ":root":
            if parts[0] == "order": root_order = tuple(p.lower() for p in parts[1:])
            elif parts[0] == "position": root_position = np.array(parts[1:], float)
            elif parts[0] == "orientation": root_orientation = np.array(parts[1:], float)
            elif parts[0] == "axis" and parts[1] != "XYZ": raise ValueError("Unsupported root axis order")
        elif section == ":bonedata":
            if line == "begin": block = {}
            elif line == "end":
                if block is None: raise ValueError("Unmatched ASF bone end")
                bones[block["name"]] = block
                block = None
            elif block is not None and parts[0] in {"name", "direction", "length", "axis", "dof"}:
                block[parts[0]] = parts[1] if parts[0] in {"name", "length"} else parts[1:]
        elif section == ":hierarchy" and line not in {"begin", "end"}:
            for child in parts[1:]:
                if child in parents: raise ValueError("Multiple ASF parents")
                parents[child] = parts[0]
    if not length_scale or angle_unit != "deg" or root_order is None:
        raise ValueError("Missing/unsupported ASF units or root order")
    if root_order != ("tx", "ty", "tz", "rx", "ry", "rz"):
        raise ValueError(f"Unsupported root DOF order {root_order}")
    result = [Bone("root", None, root_position, 0., root_orientation, root_order)]
    pending = dict(bones)
    while pending:
        added = False
        for name, bone in list(pending.items()):
            parent = parents.get(name)
            if parent not in {b.name for b in result}: continue
            if bone["axis"][-1] != "XYZ": raise ValueError("Unsupported bone axis order")
            dofs = tuple(bone.get("dof", ()))
            if any(d not in {"rx", "ry", "rz"} for d in dofs): raise ValueError("Non-rotational bone DOF")
            result.append(Bone(name, parent, np.array(bone["direction"], float), float(bone["length"]),
                               np.array(bone["axis"][:3], float), dofs))
            del pending[name]
            added = True
        if not added: raise ValueError("Disconnected/cyclic ASF hierarchy")
    return tuple(result), .0254/length_scale


def read_amc(path, bones):
    lines = [line.split("#", 1)[0].strip() for line in Path(path).read_text().splitlines()]
    if ":DEGREES" not in lines or ":FULLY-SPECIFIED" not in lines:
        raise ValueError("Only fully-specified degree AMC supported")
    frames, ids, frame = [], [], None
    expected = {b.name: len(b.dofs) for b in bones if b.dofs}
    for line in lines:
        if not line or line.startswith(":"): continue
        if line.isdigit():
            if frame is not None: frames.append(frame)
            ids.append(int(line))
            frame = {}
        else:
            if frame is None: raise ValueError("AMC data before first frame")
            name, *tokens = line.split()
            if name in frame or name not in expected: raise ValueError(f"Unknown/duplicate AMC channel: {name}")
            values = np.array(tokens, float)
            if len(values) != expected[name] or not np.isfinite(values).all(): raise ValueError("Invalid AMC values")
            frame[name] = values
    if frame is not None: frames.append(frame)
    if not frames or not np.all(np.diff(ids) == 1): raise ValueError("Empty/gapped AMC frames")
    if any(set(f) != set(expected) for f in frames): raise ValueError("Missing fully-specified AMC channels")
    return {name: np.stack([f[name] for f in frames]) for name in expected}, np.asarray(ids)


def c3d_header(path):
    header = Path(path).read_bytes()[:512]
    if len(header) != 512 or header[1] != 0x50: raise ValueError("Invalid C3D header")
    first, last = struct.unpack_from("<HH", header, 6)
    fps = struct.unpack_from("<f", header, 20)[0]
    if not 1 <= fps <= 1000 or first > last: raise ValueError("Unsupported C3D header byte order")
    return {"fps": fps, "first_frame": first, "last_frame": last, "frames": last-first+1}


def load_asfamc(asf, amc, header_path=None, *, fps=None):
    bones, scale = read_asf(asf)
    channels, frame_ids = read_amc(amc, bones)
    if header_path is not None:
        timing = c3d_header(header_path)
        if timing["frames"] != len(frame_ids): raise ValueError("AMC frame count disagrees with C3D")
        if fps is not None and not np.isclose(fps,timing['fps']):raise ValueError('Official FPS disagrees with C3D header')
        fps=timing['fps']
    if fps is None or not np.isfinite(fps) or fps<=0:
        raise ValueError('Supply a C3D header or an independently verified source FPS')
    count, nb = len(frame_ids), len(bones)
    names = tuple(b.name for b in bones)
    parents = tuple(names.index(b.parent) if b.parent is not None else -1 for b in bones)
    positions, rotations, axes = np.zeros((count, nb, 3)), np.zeros((count, nb, 3, 3)), np.zeros((count, nb, 3, 3))
    for j, bone in enumerate(bones):
        c = Rotation.from_euler("xyz", bone.axes_degrees, degrees=True).as_matrix()
        if j == 0:
            values = channels["root"]
            positions[:, j] = (values[:, :3]+bone.direction)*scale
            rotations[:, j] = c @ Rotation.from_euler("xyz", values[:, 3:], degrees=True).as_matrix() @ c.T
        else:
            degrees = np.zeros((count, 3))
            for i, dof in enumerate(bone.dofs): degrees[:, "xyz".index(dof[1])] = channels[bone.name][:, i]
            local = c @ Rotation.from_euler("xyz", degrees, degrees=True).as_matrix() @ c.T
            rotations[:, j] = rotations[:, parents[j]] @ local
            positions[:, j] = positions[:, parents[j]] + np.einsum("tij,j->ti", rotations[:, j], bone.direction*bone.length*scale)
        axes[:, j] = rotations[:, j] @ c
    yup_to_zup = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    return AcclaimMotion(names, parents, positions@yup_to_zup.T, yup_to_zup@rotations,
                         yup_to_zup@axes, frame_ids, float(fps), bones, scale, channels)
