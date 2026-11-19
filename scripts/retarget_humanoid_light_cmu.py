"""Download CMU acrobatics, validate ASF/AMC FK, and retarget a Light sample.

Use original source files with C3D-verified sampling rates. ASF anatomical
frames are mapped directly to Light's physical limb directions. Inferred
airborne frames release foot pose targets while retaining the full root turn.
Outputs are kinematic experiments; they do not certify dynamic execution.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import sys
import time
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from scipy.spatial.transform import Rotation as R, Slerp

import retarget_humanoid_light_lafan as ik

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from humanoid_retarget.datasets.cmu.asfamc import c3d_header, load_asfamc

DATA = ROOT / "src/humanoid_retarget/datasets/cmu"
OUTPUT = ROOT / "outputs/humanoid_light_cmu"


def download():
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ["88.asf", "88_01.amc", "88_05.amc", "88_01.c3d.header", "88_05.c3d.header"]:
        header_only = name.endswith(".header")
        remote = name.removesuffix(".header")
        url = "http://mocap.cs.cmu.edu/subjects/88/" + remote
        path = raw / name
        if not path.exists():
            request = Request(url, headers={"Range": "bytes=0-511"} if header_only else {})
            with urlopen(request, timeout=30) as response:
                blob = response.read(512 if header_only else -1)
            if blob.lstrip().startswith(b"<"): raise ValueError("HTML instead of CMU data")
            path.write_bytes(blob)
        entry = {"path": "raw/"+name, "source_url": url, "bytes": path.stat().st_size, "sha256": ik.digest(path)}
        if header_only:
            entry.update(partial_download="bytes 0-511 only", timing=c3d_header(path))
        records.append(entry)
    manifest = {"dataset": "CMU Graphics Lab Motion Capture Database", "subject": 88,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "files": records,
        "clips": {"88_01": "backflip", "88_05": "cartwheel into backflip"},
        "source_index_url": "http://mocap.cs.cmu.edu/search.php?subjectnumber=88",
        "units_reference_url": "http://mocap.cs.cmu.edu/faqs.php", "original_motion_files_untouched": True,
        "original_motion_and_skeleton_bytes": sum(e["bytes"] for e in records if not e["path"].endswith(".header"))}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def check_source_fk(motion):
    """Independent native MuJoCo hinge-chain validation of ASF C M C^-1 FK."""
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    base = ET.SubElement(world, "body", name="source_root")
    ET.SubElement(base, "freejoint", name="source_root_joint")
    ET.SubElement(base, "inertial", mass=".001", pos="0 0 0", diaginertia=".001 .001 .001")
    endpoints = {"root": base}
    for bone in motion.bones[1:]:
        c = R.from_euler("xyz", bone.axes_degrees, degrees=True).as_matrix()
        pre = ET.SubElement(endpoints[bone.parent], "body", quat=ik_numbers(R.from_matrix(c).as_quat(scalar_first=True)))
        for axis in "zyx":
            pre = ET.SubElement(pre, "body")
            ET.SubElement(pre, "inertial", mass=".001", pos="0 0 0", diaginertia=".001 .001 .001")
            ET.SubElement(pre, "joint", name=bone.name+"_"+axis, type="hinge", axis=ik_numbers(np.eye(3)["xyz".index(axis)]))
        endpoint = ET.SubElement(pre, "body", name=bone.name,
            pos=ik_numbers(c.T @ (bone.direction*bone.length*motion.meter_scale)),
            quat=ik_numbers(R.from_matrix(c.T).as_quat(scalar_first=True)))
        endpoints[bone.name] = endpoint
    model = ik.mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = ik.mujoco.MjData(model)
    errors = []
    for t in np.linspace(0, len(motion.frame_ids)-1, 20, dtype=int):
        data.qpos[:3] = motion.positions[t, 0]
        data.qpos[3:7] = R.from_matrix(motion.rotations[t, 0]).as_quat(scalar_first=True)
        for bone in motion.bones[1:]:
            for axis in "xyz":
                value = motion.angles_degrees[bone.name][t, bone.dofs.index("r"+axis)] if "r"+axis in bone.dofs else 0.
                data.qpos[model.joint(bone.name+"_"+axis).qposadr[0]] = np.deg2rad(value)
        ik.mujoco.mj_kinematics(model, data)
        for j, bone in enumerate(motion.bones[1:], 1):
            errors.append(np.linalg.norm(data.xpos[model.body(bone.name).id]-motion.positions[t, j]))
    maximum = float(max(errors))
    if maximum > 1e-8: raise AssertionError(f"ASF FK disagrees with native hinge-chain FK: {maximum}")
    return {"independent_mujoco_hinge_chain_samples": 20, "max_endpoint_error_m": maximum,
            "source_fps_from_c3d_header": motion.fps, "source_frames": len(motion.frame_ids),
            "source_duration_s": (len(motion.frame_ids)-1)/motion.fps, "meter_scale": motion.meter_scale}


def ik_numbers(values):
    return " ".join(format(float(v), ".17g") for v in values)


def basis(forward, left):
    forward = np.asarray(forward, float)
    forward /= np.linalg.norm(forward, axis=-1, keepdims=True)
    left = np.asarray(left, float) - np.sum(left*forward, axis=-1, keepdims=True)*forward
    left /= np.linalg.norm(left, axis=-1, keepdims=True)
    return np.stack([forward, left, np.cross(forward, left)], axis=-1)


def limb_basis(direction, left):
    z = -direction/np.linalg.norm(direction, axis=-1, keepdims=True)
    y = left-np.sum(left*z, axis=-1, keepdims=True)*z
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    return np.stack([np.cross(y, z), y, z], axis=-1)


def targets(motion, model, fps):
    source_time = np.arange(len(motion.frame_ids))/motion.fps
    times = np.arange(int(np.floor(source_time[-1]*fps+1e-8))+1)/fps
    idx = {n: i for i, n in enumerate(motion.names)}
    position = np.stack([np.stack([np.interp(times, source_time, motion.positions[:, j, k]) for k in range(3)], axis=-1) for j in range(len(idx))], axis=1)
    rotation = np.stack([Slerp(source_time, R.from_matrix(motion.rotations[:, j]))(times).as_matrix() for j in range(len(idx))], axis=1)
    axes = np.stack([Slerp(source_time, R.from_matrix(motion.coordinate_axes[:, j]))(times).as_matrix() for j in range(len(idx))], axis=1)
    # ASF reference: +X left, +Y up, +Z forward. Robot: +X forward,
    # +Y left, +Z up. This permutation preserves the full pelvis rotation.
    anatomical = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])
    root_rotation = rotation[:, 0] @ anatomical
    initial_yaw = np.arctan2(root_rotation[0, 1, 0], root_rotation[0, 0, 0])
    align = R.from_euler("z", -initial_yaw).as_matrix()
    origin = np.r_[position[0, 0, :2], 0.]
    position = (position-origin) @ align.T
    rotation, axes, root_rotation = align@rotation, align@axes, align@root_rotation
    canonical = {"Hips": "root", "Spine": "lowerback", "Spine1": "upperback", "Spine2": "thorax", "Neck": "upperneck", "Head": "head"}
    for side, prefix in [("Left", "l"), ("Right", "r")]:
        canonical.update({side+"UpLeg": prefix+"hipjoint", side+"Leg": prefix+"femur", side+"Foot": prefix+"tibia",
            side+"Toe": prefix+"foot", side+"Shoulder": "thorax", side+"Arm": prefix+"clavicle",
            side+"ForeArm": prefix+"humerus", side+"Hand": prefix+"radius"})
    human_names = list(canonical)
    human = np.stack([position[:, idx[value]] for value in canonical.values()], axis=1)
    zero = ik.mujoco.MjData(model)
    ik.mujoco.mj_forward(model, zero)
    def robot_position(name): return zero.xpos[model.body(name).id]
    robot_leg = sum(np.linalg.norm(robot_position("left_"+b+"_link")-robot_position("left_"+a+"_link")) for a, b in [("hip_pitch", "knee"), ("knee", "ankle_pitch")])
    human_leg = np.mean([sum(b.length*motion.meter_scale for b in motion.bones if b.name in {prefix+"femur", prefix+"tibia"}) for prefix in ["l", "r"]])
    scale = robot_leg/human_leg
    root = position[:, 0]*scale
    root[:, 2] += model.qpos0[2]-position[0, 0, 2]*scale
    target_rotation = {"pelvis_link": root_rotation}
    for _, h1, h2, r1, r2 in ik.bone_correspondences():
        side, part = r1.split("_", 1)[0], h1.removeprefix("Left").removeprefix("Right")
        prefix = "l" if side == "left" else "r"
        human_axis_name = prefix+("tibia" if part in {"UpLeg", "Leg"} else "radius")
        left = axes[:, idx[human_axis_name], :, 0].copy()
        if np.dot(left[0], root_rotation[0, :, 1]) < 0: left *= -1
        source_basis = limb_basis(human[:, human_names.index(h2)]-human[:, human_names.index(h1)], left)
        bend = model.joint(side+("_knee_joint" if part in {"UpLeg", "Leg"} else "_elbow_joint")).id
        robot_left = zero.xaxis[bend].copy()
        if robot_left[1] < 0: robot_left *= -1
        robot_basis = limb_basis(robot_position(r2)-robot_position(r1), robot_left)
        suffix = {"UpLeg": "hip_yaw", "Leg": "knee", "Arm": "shoulder_yaw", "ForeArm": "elbow"}[part]
        body = side+"_"+suffix+"_link"
        target_rotation[body] = source_basis @ robot_basis.T @ zero.xmat[model.body(body).id].reshape(3, 3)
        if part == "ForeArm":
            wrist = side+"_wrist_link"
            target_rotation[wrist] = source_basis @ robot_basis.T @ zero.xmat[model.body(wrist).id].reshape(3, 3)
    torso_z = position[:, idx["upperneck"]]-position[:, idx["root"]]
    torso_z /= np.linalg.norm(torso_z, axis=-1, keepdims=True)
    torso_y = position[:, idx["lclavicle"]]-position[:, idx["rclavicle"]]
    torso_x = np.cross(torso_y, torso_z)
    torso_basis = basis(torso_x, torso_y)
    target_rotation["torso_roll_link"] = torso_basis @ zero.xmat[model.body("torso_roll_link").id].reshape(3, 3)
    target_rotation["head_link"] = rotation[:, idx["head"]] @ anatomical @ zero.xmat[model.body("head_link").id].reshape(3, 3)
    contact = np.zeros((len(times), 2), bool)
    foot_targets = np.empty((len(times), 2, 3))
    _, robot_feet = ik.geometry(model)
    human_initial_width = np.linalg.norm(position[0, idx["ltibia"]]-position[0, idx["rtibia"]])
    robot_width = np.linalg.norm(robot_position("left_ankle_roll_link")-robot_position("right_ankle_roll_link"))
    width_offset = max(0., .5*(robot_width-human_initial_width*scale))
    toe_heights = np.stack([position[:, idx[p+"foot"], 2] for p in ["l", "r"]], axis=1)
    floor = float(np.quantile(toe_heights, .02))
    airborne = np.min(toe_heights, axis=1) > floor+.10
    airborne = binary_closing(airborne, iterations=1)
    for side_id, (side, prefix) in enumerate([("left", "l"), ("right", "r")]):
        left = axes[:, idx[prefix+"foot"], :, 0].copy()
        if np.dot(left[0], root_rotation[0, :, 1]) < 0: left *= -1
        forward = position[:, idx[prefix+"toes"]]-position[:, idx[prefix+"foot"]]
        foot_basis = basis(forward, left)
        body = side+"_ankle_roll_link"
        # Source toe flex is not a rigid robot foot. Calibrate the standing
        # first frame to a level sole, then preserve its subsequent rotation.
        yaw = np.arctan2(foot_basis[0, 1, 0], foot_basis[0, 0, 0])
        standing = R.from_euler("z", yaw).as_matrix() @ zero.xmat[model.body(body).id].reshape(3, 3)
        target_rotation[body] = foot_basis @ foot_basis[0].T @ standing
        ankle = position[:, idx[prefix+"tibia"]]
        foot_targets[:, side_id] = ankle*scale
        foot_targets[:, side_id, 2] += model.qpos0[2]-position[0, 0, 2]*scale
        foot_targets[:, side_id] += (1 if side == "left" else -1)*width_offset*root_rotation[:, :, 1]
        # Morphology-adjusted ankle targets must clear the actual sole hull.
        # This corrects the target ankle placement, not the root trajectory.
        sole_clearance = -np.min(np.einsum("ti,vi->tv", target_rotation[body][:, 2], robot_feet[side][1]),axis=1)+.001
        foot_targets[~airborne, side_id, 2] = np.maximum(foot_targets[~airborne, side_id, 2],sole_clearance[~airborne])
        speed = np.linalg.norm(np.gradient(ankle[:, :2], 1/fps, axis=0), axis=-1)
        support = (toe_heights[:, side_id] < floor+.045) & (speed < .25) & ~airborne
        support = binary_opening(np.pad(support, (2, 2), mode="edge"), iterations=1)[2:-2]
        contact[:, side_id] = support
        for first, end in ik.contact_intervals(support):
            foot_targets[first:end, side_id, :2] = np.median(foot_targets[first:end, side_id, :2], axis=0)
    return root, target_rotation, foot_targets, contact, airborne, human, human_names, times, scale


def accumulated_pitch(rotations):
    relative = np.swapaxes(rotations[:-1], -1, -2) @ rotations[1:]
    return float(np.rad2deg(R.from_matrix(relative).as_rotvec()[:, 1].sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", choices=["88_01", "88_05"], default="88_01")
    parser.add_argument("--fps", type=float, default=50.)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    manifest = download()
    print("DOWNLOAD", manifest["original_motion_and_skeleton_bytes"], "bytes", flush=True)
    if args.download_only: return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    motion = load_asfamc(DATA/"raw/88.asf", DATA/f"raw/{args.clip}.amc", DATA/f"raw/{args.clip}.c3d.header")
    audit = check_source_fk(motion)
    print("SOURCE", json.dumps(audit), flush=True)
    mink, _, _ = ik.prepare()
    asset = ROOT/"outputs/humanoid_light_conversion/scene.xml"
    model = ik.mujoco.MjModel.from_xml_path(str(asset))
    root, rotations, feet, contacts, flight, human, names, times, scale = targets(motion, model, args.fps)
    meshes, _ = ik.geometry(model)
    speeds = json.loads((ik.OUTPUT/"light_profile.json").read_text())["velocity_limits_joint_coordinate_rad_s"]
    print("TARGET", len(times), "frames; airborne", int(flight.sum()), "leg scale", scale, flush=True)
    started = time.perf_counter()
    qpos = ik.run_ik(mink, model, root, rotations, feet, contacts, args.fps, meshes, speeds, human, names,
                     flight_mask=flight, root_position_cost=35., root_orientation_cost=35.)
    elapsed = time.perf_counter()-started
    qvel = ik.differentiate(model, qpos, args.fps)
    report = ik.quality(model, qpos, qvel, rotations, feet, contacts, args.fps, meshes, speeds, human, names, root)
    achieved = R.from_quat(qpos[:, 3:7], scalar_first=True).as_matrix()
    root_rotation_error = np.rad2deg(R.from_matrix(achieved@np.swapaxes(rotations["pelvis_link"], -1, -2)).magnitude())
    if np.quantile(root_rotation_error, .95) > 20: report["issues"].append("Root rotation tracking error p95 exceeds 20 degrees")
    report.update(clip=args.clip, source_asf_sha256=ik.digest(DATA/"raw/88.asf"), source_amc_sha256=ik.digest(DATA/f"raw/{args.clip}.amc"),
        robot_xml_sha256=ik.digest(asset.parent/"humanoid_light.xml"), driver_sha256=ik.digest(Path(__file__)), shared_ik_sha256=ik.digest(Path(ik.__file__)),
        source_validation=audit, target_fps=args.fps, target_leg_scale=scale, airborne_frames=int(flight.sum()),
        source_accumulated_local_pitch_deg=accumulated_pitch(rotations["pelvis_link"]),
        achieved_accumulated_local_pitch_deg=accumulated_pitch(achieved),
        root_rotation_error_deg={"median": float(np.median(root_rotation_error)), "p95": float(np.quantile(root_rotation_error,.95)), "max": float(root_rotation_error.max())},
        ik_wall_seconds=elapsed, time_scaling="original timing resampled; no ballistic or torque optimization",
        note="Retargeted joint coordinates are constrained to Light limits; deeply tucked human poses can be unreachable.")
    report["status"] = "kinematic_sample_needs_correction" if report["issues"] else "kinematic_sample_needs_review"
    np.savez_compressed(OUTPUT/f"{args.clip}_source.npz", positions=motion.positions, rotations=motion.rotations,
                        names=np.asarray(motion.names), fps=np.asarray(motion.fps), parents=np.asarray(motion.parents))
    joint_names = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == ik.mujoco.mjtJoint.mjJNT_HINGE]
    np.savez_compressed(OUTPUT/f"{args.clip}_50hz.npz", qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32),
                        frequency=np.asarray(args.fps), joint_names=np.asarray(["root"]+joint_names), split_points=np.array([0,len(qpos)]),
                        source_contact=contacts, source_airborne=flight, training_ready=np.asarray(False))
    np.savez_compressed(OUTPUT/f"{args.clip}_targets.npz", root=root, rotations=np.stack(list(rotations.values()),axis=1),
                        rotation_body_names=np.asarray(list(rotations)), feet=feet, contacts=contacts, airborne=flight,
                        human_positions=human, human_joint_names=np.asarray(names), source_time=times)
    (OUTPUT/f"{args.clip}_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("RESULT",json.dumps({k:v for k,v in report.items() if k not in {"orientation_tracking","bone_direction_tracking","limitations"}}),flush=True)
    if not args.no_render: ik.render(model,qpos,human,names,args.fps,OUTPUT/f"{args.clip}_comparison.mp4",human_scale=scale)


if __name__ == "__main__":
    main()
