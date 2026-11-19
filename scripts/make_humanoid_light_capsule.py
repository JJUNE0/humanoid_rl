"""Replace Humanoid Light's CAD visuals with capsules, retaining its physics.

Build and verify a separate staging model before publishing it. Remove only
generated mesh files no longer referenced by the published MJCF/URDF. The
original USD is untouched and can regenerate the detailed visuals.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import trimesh

from convert_humanoid_light import sha256
from validate_humanoid_light import compare_source, dynamics, render, rotation


def values(element, name, default):
    return np.fromstring(element.get(name, default), sep=" ")


def numbers(array):
    return " ".join(format(float(v), ".17g") for v in array)


def fit_capsule(points):
    origin = points.mean(axis=0)
    _, basis = np.linalg.eigh(np.cov((points - origin).T))
    projected = (points - origin) @ basis
    lower, upper = projected.min(axis=0), projected.max(axis=0)
    extents = upper - lower
    main = int(extents.argmax())
    axis = basis[:, main]
    center = origin + basis @ ((lower + upper) / 2)
    radius = max(0.001, 0.49 * float(np.delete(extents, main).min()))
    radius = min(radius, 0.49 * float(extents[main]))
    half_length = max(1e-5, float(extents[main]) / 2 - radius)
    return center - axis * half_length, center + axis * half_length, radius


def capsule_color(name):
    if name == "head_link":
        return [0.78, 0.84, 0.88, 1]
    if "torso" in name or name == "pelvis_link":
        return [0.20, 0.36, 0.49, 1]
    return [0.47, 0.64, 0.75, 1]


def make_mjcf(source: Path, destination: Path):
    tree = ET.parse(source / "humanoid_light.xml")
    root = tree.getroot()
    meshes = {mesh.get("name"): mesh for mesh in root.findall("asset/mesh")}
    capsules = {}
    for body in root.findall(".//body"):
        visuals = [g for g in body.findall("geom") if g.get("group") == "2"]
        if not visuals:
            continue
        colliders = [g for g in body.findall("geom") if g.get("group") == "3"]
        points = []
        for collider in colliders:
            asset = meshes[collider.get("mesh")]
            if any(k in asset.attrib for k in ["refpos", "refquat"]):
                raise ValueError("Unexpected source mesh reference transform")
            mesh = trimesh.load_mesh(source / asset.get("file"), process=False)
            scale = values(asset, "scale", "1 1 1")
            position = values(collider, "pos", "0 0 0")
            matrix = rotation(values(collider, "quat", "1 0 0 0"))
            points.append((mesh.vertices * scale) @ matrix.T + position)
        if not points:
            raise ValueError(f"No collision geometry to fit: {body.get('name')}")
        start, end, radius = fit_capsule(np.concatenate(points))
        name = body.get("name")
        color = capsule_color(name)
        # One visual is retained at the same geom index. All physical geoms keep
        # their indices and every collision/contact parameter.
        if len(visuals) != 1:
            raise ValueError(f"Expected one original visual per body: {name}")
        visual = visuals[0]
        visual.attrib.clear()
        visual.attrib.update(name=f"{name}_capsule", type="capsule", size=str(radius),
                             fromto=numbers(np.r_[start, end]), mass="0",
                             contype="0", conaffinity="0", group="2", rgba=numbers(color))
        capsules[name] = {"start": start.tolist(), "end": end.tolist(),
                          "radius_m": radius, "rgba": color}
    referenced = {g.get("mesh") for g in root.findall(".//geom") if g.get("mesh")}
    for asset in list(root.findall("asset/mesh")):
        if asset.get("name") not in referenced:
            root.find("asset").remove(asset)
        else:
            file = Path(asset.get("file"))
            if file.is_absolute() or not (source / file).resolve().is_relative_to(source):
                raise ValueError(f"Unexpected external mesh: {file}")
            (destination / file).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / file, destination / file)
    ET.indent(tree, space="  ")
    tree.write(destination / "humanoid_light.xml", encoding="utf-8", xml_declaration=True)
    return capsules


def axis_rpy(axis):
    # A right-handed frame whose local Z follows the capsule's center line.
    helper = np.array([1., 0, 0]) if abs(axis[0]) < 0.8 else np.array([0., 1, 0])
    x = np.cross(helper, axis)
    x /= np.linalg.norm(x)
    y = np.cross(axis, x)
    matrix = np.column_stack([x, y, axis])
    pitch = np.arctan2(-matrix[2, 0], np.hypot(matrix[0, 0], matrix[1, 0]))
    if abs(np.cos(pitch)) < 1e-10:
        roll, yaw = 0., np.arctan2(-matrix[0, 1], matrix[1, 1])
    else:
        roll, yaw = np.arctan2(matrix[2, 1], matrix[2, 2]), np.arctan2(matrix[1, 0], matrix[0, 0])
    return [roll, pitch, yaw]


def make_urdf(source: Path, destination: Path, capsules):
    tree = ET.parse(source / "humanoid_light.urdf")
    for link in tree.getroot().findall("link"):
        name = link.get("name")
        if name not in capsules:
            continue
        for element in list(link.findall("visual")):
            link.remove(element)
        capsule = capsules[name]
        start, end, radius = np.array(capsule["start"]), np.array(capsule["end"]), capsule["radius_m"]
        length = float(np.linalg.norm(end - start))
        rpy = axis_rpy((end - start) / length)
        # URDF has no standard capsule primitive: a cylinder and two spheres
        # give the same silhouette without requiring any visual mesh files.
        parts = [("cylinder", (start + end) / 2, {"radius": str(radius), "length": str(length)}, rpy),
                 ("sphere", start, {"radius": str(radius)}, [0, 0, 0]),
                 ("sphere", end, {"radius": str(radius)}, [0, 0, 0])]
        for i, (kind, pos, attributes, angles) in enumerate(parts):
            visual = ET.SubElement(link, "visual", name=f"{name}_capsule_{i}")
            ET.SubElement(visual, "origin", xyz=numbers(pos), rpy=numbers(angles))
            ET.SubElement(ET.SubElement(visual, "geometry"), kind, **attributes)
            material = ET.SubElement(visual, "material", name=f"{name}_capsule_color")
            ET.SubElement(material, "color", rgba=numbers(capsule["rgba"]))
    ET.indent(tree, space="  ")
    tree.write(destination / "humanoid_light.urdf", encoding="utf-8", xml_declaration=True)


def compare_physics(original: Path, simplified: Path):
    started = time.perf_counter()
    before = mujoco.MjModel.from_xml_path(str(original / "scene.xml"))
    original_seconds = time.perf_counter() - started
    started = time.perf_counter()
    after = mujoco.MjModel.from_xml_path(str(simplified / "scene.xml"))
    simplified_seconds = time.perf_counter() - started
    fields = ["body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat",
              "body_gravcomp", "jnt_type", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited",
              "jnt_stiffness", "dof_damping", "dof_armature", "dof_frictionloss", "qpos0",
              "actuator_trnid", "actuator_trntype", "actuator_gear", "actuator_gainprm", "actuator_biasprm",
              "actuator_dynprm", "actuator_ctrlrange", "actuator_forcerange"]
    for name in fields:
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name), err_msg=name)
    assert (before.nq, before.nv, before.nu, before.ngeom) == (after.nq, after.nv, after.nu, after.ngeom)
    physical = np.flatnonzero((before.geom_contype != 0) | (before.geom_conaffinity != 0))
    for name in ["geom_type", "geom_bodyid", "geom_pos", "geom_quat", "geom_size", "geom_contype",
                 "geom_conaffinity", "geom_friction", "geom_solref", "geom_solimp", "geom_solmix",
                 "geom_condim", "geom_margin", "geom_gap"]:
        np.testing.assert_array_equal(getattr(before, name)[physical], getattr(after, name)[physical], err_msg=name)
    a, b = mujoco.MjData(before), mujoco.MjData(after)
    rng = np.random.default_rng(28)
    max_acceleration_error = 0.
    for _ in range(20):
        pose = before.qpos0.copy()
        pose[2] += rng.uniform(0, 0.25)
        pose[7:] = rng.uniform(-0.1, 0.1, before.nq - 7)
        velocity = rng.normal(0, 0.1, before.nv)
        control = rng.uniform(-1, 1, before.nu)
        for model, data in [(before, a), (after, b)]:
            data.qpos[:], data.qvel[:], data.ctrl[:] = pose, velocity, control
            mujoco.mj_forward(model, data)
        assert a.ncon == b.ncon
        # MuJoCo 3.11 renamed the packed mass matrix from qM to M.
        mass_a = a.M if hasattr(a, "M") else a.qM
        mass_b = b.M if hasattr(b, "M") else b.qM
        np.testing.assert_allclose(mass_a, mass_b, atol=1e-12, rtol=0)
        error = float(np.max(np.abs(a.qacc - b.qacc)))
        max_acceleration_error = max(error, max_acceleration_error)
        np.testing.assert_allclose(a.qacc, b.qacc, atol=1e-9, rtol=1e-12)
    mujoco.mj_resetData(before, a)
    mujoco.mj_resetData(after, b)
    max_trajectory_error = 0.
    for _ in range(500):
        mujoco.mj_step(before, a)
        mujoco.mj_step(after, b)
        error = float(np.max(np.abs(a.qpos - b.qpos)))
        max_trajectory_error = max(error, max_trajectory_error)
        assert error < 1e-8, error
    return {"status": "passed", "identical_physics_arrays": fields,
            "physical_geom_count_including_floor": len(physical), "random_states": 20,
            "max_acceleration_difference": max_acceleration_error,
            "one_second_max_qpos_difference": max_trajectory_error,
            "original_load_seconds": original_seconds, "capsule_load_seconds": simplified_seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/humanoid_light_conversion")
    args = parser.parse_args()
    source = args.asset_dir.resolve()
    original_xml = ET.parse(source / "humanoid_light.xml").getroot()
    old_meshes = [(source / m.get("file")).resolve() for m in original_xml.findall("asset/mesh")]
    size_before = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
    with tempfile.TemporaryDirectory(prefix=".humanoid-capsule-", dir=source.parent) as folder:
        stage = Path(folder)
        capsules = make_mjcf(source, stage)
        make_urdf(source, stage, capsules)
        for name in ["scene.xml", "source_properties.json", "control.json"]:
            shutil.copy2(source / name, stage / name)
        equivalence = compare_physics(source, stage)
        print("PHYSICS_EQUIVALENCE", json.dumps(equivalence), flush=True)
        validation = {"status": "passed", "comparison": compare_source(stage, 20),
                      "dynamics": dynamics(stage)}
        render(stage)
        history = source / "history"
        history.mkdir(exist_ok=True)
        for name in ["humanoid_light.xml", "humanoid_light.urdf", "preview.png", "validation.json"]:
            backup = history / (Path(name).stem + "_fullres" + Path(name).suffix)
            if not backup.exists():
                shutil.copy2(source / name, backup)
        # Publish verified runtime files before removing any unused generated CAD assets.
        for name in ["humanoid_light.xml", "humanoid_light.urdf", "scene.xml", "preview.png", "collision_preview.png"]:
            shutil.copy2(stage / name, source / name)
        kept = {(source / m.get("file")).resolve() for m in ET.parse(source / "humanoid_light.xml").getroot().findall("asset/mesh")}
        removed = []
        for file in old_meshes:
            if file not in kept:
                if not file.is_relative_to(source / "meshes"):
                    raise ValueError(f"Refusing to remove external generated file: {file}")
                removed.append({"path": str(file.relative_to(source)), "bytes": file.stat().st_size})
                file.unlink()
            # The retained convex hull OBJs have no material references.
            material = file.with_suffix(".mtl")
            if material.exists():
                removed.append({"path": str(material.relative_to(source)), "bytes": material.stat().st_size})
                material.unlink()
        report = {"visual_style": "capsule", "capsule_count": len(capsules), "capsules": capsules,
                  "physics_equivalence": equivalence, "removed_generated_assets": removed,
                  "original_usd_unchanged": True, "size_before_bytes": size_before}
        (source / "capsule_visuals.json").write_text(json.dumps(report, indent=2) + "\n")
        validation.update(previews=[str(source / n) for n in ["preview.png", "collision_preview.png"]],
                          visual_style="capsule", physics_equivalence=equivalence,
                          limitations=["Visual geometry is intentionally approximate; camera/depth policies see a different appearance.",
                                       "Delayed PD and coupled ankle/torso control remain separate from the independent torque inputs."])
        (source / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(visual_style="capsule", simplified_at=datetime.now().astimezone().isoformat(),
                        retained_visual_meshes=0, collision_meshes=len(kept), capsule_count=len(capsules),
                        high_resolution_visuals="Regenerate from the preserved USD with scripts/convert_humanoid_light.py.")
        manifest["sha256"] = {n: sha256(source / n) for n in manifest["files"].values() if (source / n).is_file()}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        size_after = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
        report["size_after_bytes"] = size_after
        (source / "capsule_visuals.json").write_text(json.dumps(report, indent=2) + "\n")
        validation["disk_bytes"] = size_after
        (source / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        # The validation file was updated with the final size after publication.
        manifest["sha256"]["validation.json"] = sha256(source / "validation.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print("CAPSULE_RESULT", json.dumps({"capsules": len(capsules), "before_MiB": size_before/2**20,
              "after_MiB": size_after/2**20, "removed_files": len(removed),
              "physics_equivalence": equivalence["status"]}), flush=True)


if __name__ == "__main__":
    main()
