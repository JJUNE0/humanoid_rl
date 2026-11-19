"""Export the local Humanoid Light USD through NVIDIA's URDF exporter to MuJoCo.

Uses the already installed Isaac Sim Python/USD libraries in a subprocess;
does not start Isaac Sim or modify its environment or the source robot.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "cocelo-rl-isaaclab/lab/cocelo/assets/data/Robots/Flamingo/humanoid_light_rev_1_0_12_45/assets (2)/humanoid_light_rev_1_0_12_45_merge_joints.usd"
ISAAC_PYTHON = Path("/home/cocel/isaac/env_isaaclab/bin/python")
CONTROL_SOURCE = ROOT.parent / "cocelo-rl-isaaclab/lab/cocelo/assets/flamingo/humanoid_light_v1.py"


def isaac_environment(python: Path):
    simulator = next((python.parent.parent / "lib").glob("python*/site-packages/isaacsim"))
    usd = next((simulator / "extscache").glob("omni.usd.libs-*"))
    exporter = simulator / "exts/isaacsim.asset.exporter.urdf/pip_prebundle"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(usd), str(exporter), env.get("PYTHONPATH")]))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [
        str(usd / "bin"), str(simulator / "kit/lib"), str(simulator / "kit"),
        str(python.resolve().parent.parent / "lib"), env.get("LD_LIBRARY_PATH"),
    ]))
    env["HUMANOID_LIGHT_EXPORTER_PATH"] = str(exporter)
    return env


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_usd(source: Path, output: Path):
    import nvidia
    # Isaac's installed nvidia namespace is also used by CUDA pip packages.
    nvidia.__path__.append(str(Path(os.environ["HUMANOID_LIGHT_EXPORTER_PATH"]) / "nvidia"))
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from nvidia.srl.from_usd.to_urdf import UsdToUrdf

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError(f"Cannot open USD stage: {source}")
    if UsdGeom.GetStageUpAxis(stage) != "Z" or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1):
        raise ValueError("This conversion expects the source's Z-up, metre convention")
    cache = UsdGeom.XformCache()
    quaternion = lambda q: [float(q.GetReal()), *map(float, q.GetImaginary())]
    bodies, joints, meshes, colliders = [], [], [], []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append({"path": str(prim.GetPath()),
                              "enabled": UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                              "approximation": prim.GetAttribute("physics:approximation").Get()})
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.MassAPI(prim)
            bodies.append({
                "name": str(prim.GetName()), "path": str(prim.GetPath()),
                "mass": api.GetMassAttr().Get(),
                "center_of_mass": list(api.GetCenterOfMassAttr().Get()),
                "diagonal_inertia": list(api.GetDiagonalInertiaAttr().Get()),
                "principal_axes_wxyz": quaternion(api.GetPrincipalAxesAttr().Get()),
                "world_transform": np.asarray(cache.GetLocalToWorldTransform(prim)).T.tolist(),
            })
        if prim.IsA(UsdPhysics.Joint):
            api = UsdPhysics.Joint(prim)
            entry = {
                "name": str(prim.GetName()), "type": prim.GetTypeName(),
                "parent": [str(p) for p in api.GetBody0Rel().GetTargets()],
                "child": [str(p) for p in api.GetBody1Rel().GetTargets()],
                "local_pos0": list(api.GetLocalPos0Attr().Get()),
                "local_pos1": list(api.GetLocalPos1Attr().Get()),
                "local_rot0_wxyz": quaternion(api.GetLocalRot0Attr().Get()),
                "local_rot1_wxyz": quaternion(api.GetLocalRot1Attr().Get()),
            }
            if prim.IsA(UsdPhysics.RevoluteJoint):
                hinge = UsdPhysics.RevoluteJoint(prim)
                entry.update(axis=hinge.GetAxisAttr().Get(), lower_degrees=hinge.GetLowerLimitAttr().Get(),
                             upper_degrees=hinge.GetUpperLimitAttr().Get())
            joints.append(entry)
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            ancestor = prim
            collision = False
            while ancestor and not ancestor.IsPseudoRoot():
                collision |= ancestor.HasAPI(UsdPhysics.CollisionAPI)
                ancestor = ancestor.GetParent()
            meshes.append({"path": str(prim.GetPath()), "vertices": len(mesh.GetPointsAttr().Get()),
                           "polygons": len(mesh.GetFaceVertexCountsAttr().Get()),
                           "collision": collision})
    properties = {
        "source": str(source), "source_files_sha256": {
            str(p.relative_to(source.parent)): sha256(p)
            for p in sorted(source.parent.rglob("*")) if p.is_file()
        },
        "usd_version": Usd.GetVersion(), "default_prim": str(stage.GetDefaultPrim().GetPath()),
        "bodies": bodies, "joints": joints, "meshes": meshes, "colliders": colliders,
        "total_mass": sum(b["mass"] or 0 for b in bodies),
    }
    (output / "source_properties.json").write_text(json.dumps(properties, indent=2) + "\n")
    print("SOURCE_SUMMARY", json.dumps({"bodies": len(bodies), "joints": len(joints),
        "meshes": len(meshes), "mass": properties["total_mass"],
        "vertices": sum(m["vertices"] for m in meshes)}), flush=True)
    converter = UsdToUrdf(stage)
    path = converter.save_to_file(output / "humanoid_light.urdf", mesh_dir=output / "meshes")
    print("URDF_EXPORTED", path, flush=True)


def read_control(joint_names):
    """Read literal actuator settings without importing or initializing Isaac Lab."""
    tree = ast.parse(CONTROL_SOURCE.read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "HUMANOID_LIGHT_CFG" for t in node.targets))
    actuators = next(k.value for k in assignment.value.keywords if k.arg == "actuators")
    groups, joints = {}, {}
    for name, call in zip(actuators.keys, actuators.values):
        values = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        values["actuator_class"] = call.func.id
        groups[ast.literal_eval(name)] = values
        for joint in joint_names:
            if not any(re.fullmatch(pattern, joint) for pattern in values["joint_names_expr"]):
                continue
            if joint in joints:
                raise ValueError(f"Overlapping actuator groups: {joint}")
            params = {"group": ast.literal_eval(name)}
            for key in ("effort_limit", "velocity_limit", "stiffness", "damping", "armature"):
                matches = [v for pattern, v in values.get(key, {}).items() if re.fullmatch(pattern, joint)]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {key} for {joint}, got {matches}")
                params[key] = matches[0]
            joints[joint] = params
    if set(joints) != set(joint_names):
        raise ValueError("Actuator settings do not cover the exported joints")
    return {"source": str(CONTROL_SOURCE), "source_sha256": sha256(CONTROL_SOURCE),
            "groups": groups, "joints": joints,
            "implementation": "MJCF exposes independent torque inputs; coupled torque limits and delayed PD require a separate controller."}


def normalize_urdf(output: Path):
    import numpy as np
    import mujoco

    properties = json.loads((output / "source_properties.json").read_text())
    path = output / "humanoid_light.urdf"
    original = output / "humanoid_light_exported.urdf"
    if not original.exists():
        original.write_bytes(path.read_bytes())
    tree = ET.parse(original)
    robot = tree.getroot()
    robot.set("name", "humanoid_light")
    source_bodies = [b["name"] for b in properties["bodies"]]
    source_joints = [j["name"] for j in properties["joints"]]

    def canonical(name, candidates):
        matches = [n for n in candidates if name == n or name.endswith("_" + n)]
        if not matches:
            return name
        return max(matches, key=len)

    names = {link.get("name"): canonical(link.get("name"), source_bodies) for link in robot.findall("link")}
    root_names = set(names) - {j.find("child").get("link") for j in robot.findall("joint")}
    if len(root_names) != 1:
        raise ValueError(f"Exported URDF has ambiguous roots: {root_names}")
    root_name = next(iter(root_names))
    # SRL merges the pelvis's identity transform into the stage's root Xform.
    if names[root_name] not in source_bodies:
        names[root_name] = "pelvis_link"
    if len(set(names.values())) != len(names):
        raise ValueError("Canonical body names are not unique")
    for link in robot.findall("link"):
        link.set("name", names[link.get("name")])
    removed_visuals = 0
    for link in robot.findall("link"):
        collision_files = {m.get("filename") for m in link.findall("collision/geometry/mesh")}
        for visual in list(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is not None and mesh.get("filename") in collision_files:
                link.remove(visual)
                removed_visuals += 1
    for joint in robot.findall("joint"):
        joint.set("name", canonical(joint.get("name"), source_joints))
        for element in (joint.find("parent"), joint.find("child")):
            element.set("link", names[element.get("link")])
    # The exporter changes body frames when aligning joint origins. Its MassAPI
    # values must be expressed in those new frames, rather than copied verbatim.
    poses = {names[root_name]: np.eye(4)}
    pending = list(robot.findall("joint"))
    while pending:
        previous_count = len(pending)
        for joint in list(pending):
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in poses:
                continue
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
            rotate = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                               [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = rotate, xyz
            poses[child] = poses[parent] @ pose
            pending.remove(joint)
        if len(pending) == previous_count:
            raise ValueError("URDF joint graph is not a connected tree")
    restored_pelvis = False
    corrected_inertials = []
    for source_body in properties["bodies"]:
        name = source_body["name"]
        link = next(link for link in robot.findall("link") if link.get("name") == name)
        old = link.find("inertial")
        if name == "pelvis_link":
            restored_pelvis = old is None
        if old is not None:
            link.remove(old)
        if not source_body["mass"]:
            continue
        correction = np.linalg.inv(poses[name]) @ np.asarray(source_body["world_transform"])
        center = correction[:3, :3] @ source_body["center_of_mass"] + correction[:3, 3]
        q = np.asarray(source_body["principal_axes_wxyz"])
        principal = np.empty(9)
        mujoco.mju_quat2Mat(principal, q / np.linalg.norm(q))
        rotate = correction[:3, :3] @ principal.reshape(3, 3)
        inertia = rotate @ np.diag(source_body["diagonal_inertia"]) @ rotate.T
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(map(str, center)), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=str(source_body["mass"]))
        ET.SubElement(inertial, "inertia", **{
            key: str(inertia[i, j]) for key, i, j in
            [("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2), ("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2)]})
        corrected_inertials.append(name)
    actuated = [j.get("name") for j in robot.findall("joint") if j.get("type") != "fixed"]
    control = read_control(actuated)
    # Include visual geometry and keep fixed sensor/camera links and their masses.
    mujoco = ET.SubElement(robot, "mujoco")
    ET.SubElement(mujoco, "compiler", discardvisual="false", fusestatic="false", strippath="false")
    for joint in robot.findall("joint"):
        if joint.get("name") in control["joints"]:
            params = control["joints"][joint.get("name")]
            limit = joint.find("limit")
            if limit is None:
                limit = ET.SubElement(joint, "limit")
            # The USD's default drives can be unbounded; Isaac's explicit actuators
            # replace those drive limits at runtime with these finite settings.
            limit.set("effort", str(params["effort_limit"]))
            limit.set("velocity", str(params["velocity_limit"]))
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    (output / "control.json").write_text(json.dumps(control, indent=2) + "\n")
    (output / "name_map.json").write_text(json.dumps(names, indent=2) + "\n")
    repairs = {"pelvis_mass_and_inertia_restored_from_usd": restored_pelvis,
               "inertial_frames_recomputed_from_usd_world_properties": corrected_inertials,
               "collision_visual_duplicates_removed": removed_visuals,
               "finite_effort_and_velocity_limits_from": str(CONTROL_SOURCE)}
    (output / "urdf_repairs.json").write_text(json.dumps(repairs, indent=2) + "\n")
    return control


def prepare_collision_meshes(output: Path):
    """Bake the source's explicit convexHull collision approximation into URDF."""
    import numpy as np
    import trimesh

    source = json.loads((output / "source_properties.json").read_text())
    if not source.get("colliders") or any(c["approximation"] != "convexHull" for c in source["colliders"]):
        raise ValueError("Collision baking is only supported for the source's convexHull colliders")
    report_path = output / "collision_meshes.json"
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    robot = ET.parse(output / "humanoid_light.urdf").getroot()
    results = {}
    for element in robot.findall("link/collision/geometry/mesh"):
        relative = element.get("filename")
        path = (output / relative).resolve()
        if not path.is_relative_to(output):
            raise ValueError(f"Unexpected external collision mesh: {path}")
        digest = sha256(path)
        if relative in previous and previous[relative]["sha256"] == digest:
            results[relative] = previous[relative]
            continue
        mesh = trimesh.load_mesh(path, process=False)
        original_bytes = path.stat().st_size
        hull = mesh.convex_hull
        error = float(np.abs(mesh.bounds - hull.bounds).max())
        if error > 1e-7 or not hull.is_watertight or not hull.is_convex:
            raise ValueError(f"Invalid convex collision geometry: {relative}, bounds error={error}")
        path.write_text(trimesh.exchange.obj.export_obj(hull, include_normals=False, include_color=False))
        results[relative] = {"source_triangles": len(mesh.faces), "hull_triangles": len(hull.faces),
                             "source_bytes": original_bytes, "bytes": path.stat().st_size,
                             "bounds_error_m": error, "sha256": sha256(path)}
        report_path.write_text(json.dumps(results, indent=2) + "\n")
        print("COLLISION_HULL", Path(relative).name, len(mesh.faces), "->", len(hull.faces), flush=True)
    report_path.write_text(json.dumps(results, indent=2) + "\n")


def convert_mjcf(output: Path, control):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(output / "humanoid_light.urdf"))
    root = spec.body("pelvis_link")
    root.add_freejoint(name="root")
    root.pos = [0, 0, 0.71]
    for name, params in control["joints"].items():
        joint = spec.joint(name)
        joint.armature = params["armature"]
        limit = params["effort_limit"]
        spec.add_actuator(name=name.removesuffix("_joint") + "_motor", target=name,
                          trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrllimited=True,
                          ctrlrange=[-limit, limit])
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            # Same self-collision setting as the Isaac asset; ground still contacts.
            geom.contype, geom.conaffinity, geom.group = 0, 1, 3
        else:
            geom.group = 2
    spec.modelname = "humanoid_light"
    model = spec.compile()
    xml = ET.fromstring(spec.to_xml())
    # MuJoCo's default XML writer rounds to six significant digits. Keep the
    # source-derived mass properties at full precision in the durable artifact.
    for body in xml.findall(".//body"):
        compiled = model.body(body.get("name"))
        inertial = body.find("inertial")
        if inertial is not None:
            inertial.set("mass", repr(float(compiled.mass[0])))
            inertial.set("pos", " ".join(map(repr, map(float, compiled.ipos))))
            inertial.set("quat", " ".join(map(repr, map(float, compiled.iquat))))
            inertial.set("diaginertia", " ".join(map(repr, map(float, compiled.inertia))))
    ET.indent(xml, space="  ")
    ET.ElementTree(xml).write(output / "humanoid_light.xml", encoding="utf-8", xml_declaration=True)
    scene = '''<mujoco model="humanoid_light_scene">
  <include file="humanoid_light.xml"/>
  <visual>
    <global offwidth="1200" offheight="1000"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.65 0.65 0.65"/>
  </visual>
  <worldbody>
    <light pos="1 -1 3" dir="-0.2 0.2 -1" diffuse="0.7 0.7 0.7"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.24 0.27 0.3 1"
          contype="1" conaffinity="0" friction="1 0.005 0.0001"/>
  </worldbody>
</mujoco>
'''
    (output / "scene.xml").write_text(scene)
    print("MJCF_EXPORTED", json.dumps({"nbody": model.nbody, "njnt": model.njnt,
          "nq": model.nq, "nv": model.nv, "nu": model.nu}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/humanoid_light_conversion")
    parser.add_argument("--isaac-python", type=Path, default=ISAAC_PYTHON)
    parser.add_argument("--export-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.export_worker:
        export_usd(args.source.resolve(), output)
        return
    if not args.skip_export:
        command = [str(args.isaac_python), str(Path(__file__).resolve()), "--export-worker",
                   "--source", str(args.source.resolve()), "--output-dir", str(output)]
        with (output / "usd_to_urdf.log").open("w") as log:
            subprocess.run(command, env=isaac_environment(args.isaac_python), stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        (output / "humanoid_light_exported.urdf").write_bytes((output / "humanoid_light.urdf").read_bytes())
    control = normalize_urdf(output)
    prepare_collision_meshes(output)
    convert_mjcf(output, control)
    print("URDF", output / "humanoid_light.urdf")
    print("MJCF", output / "humanoid_light.xml")


if __name__ == "__main__":
    main()
