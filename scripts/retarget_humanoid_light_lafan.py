"""Retarget short original LAFAN1 clips to the 26-DoF Humanoid Light.

Uses pinned GMR BVH decoding / human-axis conventions and Mink/DAQP for
sequential constrained differential IK. Robot-specific offsets are computed
from the compiled robot's bone directions, not copied from the G1 joint list.
Stance targets are constant over each inferred source contact interval. Joint
position, cumulative per-frame speed and selected collision distances are QP
constraints. This is kinematic retargeting, not a dynamic tracking certificate.

Run with the repository .venv. Optional IK dependencies are isolated under
outputs/humanoid_light_retarget/python_deps; training dependencies are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types
from urllib.request import urlopen
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.ndimage import binary_closing, binary_erosion, binary_opening
from scipy.spatial.transform import Rotation as R, Slerp

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/humanoid_light_retarget"
REVISION = "25f56db79ac411258b377b6d4ccba0c8ff3372ac"
UPSTREAM_FILES = ["LICENSE", "DOC.md", "general_motion_retargeting/motion_retarget.py",
    "general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json",
    "general_motion_retargeting/utils/lafan_vendor/extract.py",
    "general_motion_retargeting/utils/lafan_vendor/utils.py", "assets/unitree_g1/g1_mocap_29dof.xml"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    vendor = OUTPUT / "upstream"
    records = []
    for name in UPSTREAM_FILES:
        path = vendor / name
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(urlopen(f"https://raw.githubusercontent.com/YanjieZe/GMR/{REVISION}/{name}", timeout=30).read())
        records.append({"path": name, "sha256": digest(path)})
    # GMR's BVH helpers are imported without its package root, which also imports
    # unrelated SMPL/streaming/viewer dependencies and robot assets.
    package = types.ModuleType("light_lafan_vendor")
    package.__path__ = [str(vendor / "general_motion_retargeting/utils/lafan_vendor")]
    sys.modules[package.__name__] = package
    dependencies = OUTPUT / "python_deps"
    sys.path.insert(0, str(dependencies))
    if not (dependencies / "mink").is_dir():
        subprocess.run(["uv", "pip", "install", "--target", str(dependencies), "--no-deps",
                        "mink==1.3.0", "qpsolvers==4.13.0", "daqp==0.10.3"], check=True)
    import mink
    (vendor / "revision.json").write_text(json.dumps({"repository": "https://github.com/YanjieZe/GMR",
        "commit": REVISION, "files": records, "usage": "BVH decoder and axis convention reference; custom Light IK driver"}, indent=2)+"\n")
    return mink, importlib.import_module("light_lafan_vendor.extract"), importlib.import_module("light_lafan_vendor.utils")


def load_human(path, extract, utils):
    animation = extract.read_bvh(str(path))
    quats, positions = utils.quat_fk(animation.quats, animation.pos, animation.parents)
    yup_to_zup = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    positions = positions @ yup_to_zup.T / 100
    rotations = yup_to_zup @ R.from_quat(quats.reshape(-1, 4), scalar_first=True).as_matrix().reshape(*quats.shape[:-1], 3, 3)
    # Read actual frame time instead of assuming that changing an FPS label
    # resamples a trajectory. LAFAN's header uses rounded 0.033333 seconds.
    with path.open() as stream:
        for line in stream:
            if line.strip().startswith("Frame Time:"):
                fps = 1 / float(line.split(":")[1])
                break
    return animation.bones, positions, rotations, animation.offsets / 100, fps


def kinematic_reference_g1():
    root = ET.parse(OUTPUT / "upstream/assets/unitree_g1/g1_mocap_29dof.xml").getroot()
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag in {"geom", "asset", "keyframe"}:
                parent.remove(child)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def anatomical_basis(direction, bend_axis):
    z = -direction / np.linalg.norm(direction)
    y = bend_axis - np.dot(bend_axis, z) * z
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


def calibrated_offsets(model, data):
    gm, gd = kinematic_reference_g1()
    upstream = json.loads((OUTPUT / "upstream/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json").read_text())["ik_match_table1"]
    mapping = {"pelvis_link": "pelvis", "torso_roll_link": "torso_link"}
    for side in ["left", "right"]:
        for suffix in ["hip_yaw", "knee", "ankle_roll", "shoulder_yaw", "elbow"]:
            mapping[f"{side}_{suffix}_link"] = f"{side}_{suffix}_link"
        mapping[f"{side}_wrist_link"] = f"{side}_wrist_yaw_link"
    result = {}
    anatomical = {}
    for robot, g1 in mapping.items():
        human = upstream[g1][0]
        original = R.from_quat(upstream[g1][-1], scalar_first=True).as_matrix()
        gr, lr = gd.xmat[gm.body(g1).id].reshape(3, 3), data.xmat[model.body(robot).id].reshape(3, 3)
        ag = al = np.eye(3)
        side = robot.split("_")[0]
        if any(part in robot for part in ["hip_yaw", "knee", "shoulder_yaw", "elbow", "wrist"]):
            if "hip_yaw" in robot:
                start, end, bend = "hip_pitch", "knee", "knee"
            elif "knee" in robot:
                start, end, bend = "knee", "ankle_pitch", "knee"
            elif "shoulder_yaw" in robot:
                start, end, bend = "shoulder_pitch", "elbow", "elbow"
            else:
                start, end, bend = "elbow", "wrist", "elbow"
            g_end = "wrist_yaw" if end == "wrist" else end
            gp = gd.xpos[gm.body(f"{side}_{g_end}_link").id] - gd.xpos[gm.body(f"{side}_{start}_link").id]
            lp = data.xpos[model.body(f"{side}_{end}_link").id] - data.xpos[model.body(f"{side}_{start}_link").id]
            ga = gd.xaxis[gm.joint(f"{side}_{bend}_joint").id]
            la = data.xaxis[model.joint(f"{side}_{bend}_joint").id]
            # Mirrored elbow axes need a consistent anatomical left direction.
            if ga[1] < 0: ga = -ga
            if la[1] < 0: la = -la
            ag, al = anatomical_basis(gp, ga), anatomical_basis(lp, la)
        offset = original @ gr.T @ ag @ al.T @ lr
        if any(part in robot for part in ["hip_yaw", "knee", "shoulder_yaw", "elbow", "wrist"]):
            # LAFAN's child offsets explicitly identify the source long-bone
            # axis as +X. Enforce that exact axis rather than transferring G1's
            # fixed link offsets / non-straight zero-pose limb proportions.
            lbone = lr.T @ lp
            lbone /= np.linalg.norm(lbone)
            lbend = lr.T @ la
            lbend -= np.dot(lbend, lbone)*lbone
            lbend /= np.linalg.norm(lbend)
            hbone = np.array([1., 0, 0])
            hbend = original @ gr.T @ ga
            hbend -= np.dot(hbend, hbone)*hbone
            hbend /= np.linalg.norm(hbend)
            offset = np.column_stack([hbone, hbend, np.cross(hbone, hbend)]) @ np.column_stack([lbone, lbend, np.cross(lbone, lbend)]).T
        result[robot] = {"human": human, "rotation_wxyz": R.from_matrix(offset).as_quat(scalar_first=True).tolist()}
        anatomical[robot] = {"g1_anatomical_basis_world": ag.tolist(), "light_anatomical_basis_world": al.tolist()}
    # LAFAN head and torso share the same longitudinal bone convention. Light
    # has one head yaw coordinate, so head tilt is only a low-priority target.
    torso_source = R.from_quat(upstream["torso_link"][-1], scalar_first=True).as_matrix()
    result["head_link"] = {"human": "Head", "rotation_wxyz": R.from_matrix(torso_source @ data.xmat[model.body("head_link").id].reshape(3, 3)).as_quat(scalar_first=True).tolist()}
    return result, anatomical


def geometry(model):
    meshes, feet = [], {}
    for gid in np.flatnonzero(model.geom_group == 3):
        mid = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vertices = np.asarray(model.mesh_vert[start:start+count], float)
        meshes.append((gid, vertices))
        name = model.body(model.geom_bodyid[gid]).name
        if name in {"left_ankle_roll_link", "right_ankle_roll_link"}:
            # Mesh vertices in the body frame include MuJoCo's compiled mesh
            # principal-axis transform; geometry is the retained collision hull.
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, model.geom_quat[gid])
            feet[name.split("_")[0]] = (gid, vertices @ matrix.reshape(3, 3).T + model.geom_pos[gid])
    return meshes, feet


def contact_intervals(mask):
    change = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))


def self_collision_pairs(model):
    by_body = {model.body(model.geom_bodyid[g]).name: int(g) for g in np.flatnonzero(model.geom_group == 3)}
    groups = []
    torso = ["torso_roll_link", "pelvis_link", "head_link"]
    for side in ["left", "right"]:
        groups.append(([side+"_elbow_link", side+"_wrist_link"], torso))
    groups.append((["left_elbow_link", "left_wrist_link"], ["right_elbow_link", "right_wrist_link"]))
    groups.append((["left_knee_link", "left_ankle_roll_link"], ["right_knee_link", "right_ankle_roll_link"]))
    return [(by_body[a], by_body[b]) for first, second in groups for a in first for b in second]


def targets_for_clip(names, positions, rotations, source_fps, start, seconds, offsets, model, reference, foot_meshes, fps):
    times = start + np.arange(int(round(seconds * fps)) + 1) / fps
    source_time = np.arange(len(positions)) / source_fps
    if times[-1] > source_time[-1]:
        raise ValueError("Requested clip extends past source motion")
    pos = np.stack([np.stack([np.interp(times, source_time, positions[:, j, k]) for k in range(3)], axis=-1) for j in range(len(names))], axis=1)
    rot = np.stack([Slerp(source_time, R.from_matrix(rotations[:, j]))(times).as_matrix() for j in range(len(names))], axis=1)
    idx = {name: i for i, name in enumerate(names)}
    source_root_offset = R.from_quat(offsets["pelvis_link"]["rotation_wxyz"], scalar_first=True).as_matrix()
    initial_root = rot[0, idx["Hips"]] @ source_root_offset
    yaw = np.arctan2(initial_root[1, 0], initial_root[0, 0])
    align = R.from_euler("z", -yaw).as_matrix()
    origin = np.r_[pos[0, idx["Hips"], :2], 0]
    pos = (pos-origin) @ align.T
    rot = align @ rot
    # One reusable morphology scale measured from the reference BVH's segment
    # lengths and Light's actual hip/knee/ankle joint-center geometry.
    scale, root_z_reference, human_root_z_reference, human_ankle_z_reference = reference
    root = pos[:, idx["Hips"]] * scale
    root[:, 2] = root_z_reference + scale * (pos[:, idx["Hips"], 2] - human_root_z_reference)
    target_rot = {}
    for robot, entry in offsets.items():
        human = entry["human"].replace("FootMod", "Toe")
        target_rot[robot] = rot[:, idx[human]] @ R.from_quat(entry["rotation_wxyz"], scalar_first=True).as_matrix()
    contacts = np.zeros((len(times), 2), dtype=bool)
    foot_targets = np.zeros((len(times), 2, 3))
    for side_id, side in enumerate(["left", "right"]):
        foot = pos[:, idx[side.title()+"Foot"]]
        toe = pos[:, idx[side.title()+"Toe"]]
        speed = np.linalg.norm(np.gradient(foot[:, :2], 1/fps, axis=0), axis=-1)
        floor = np.quantile(np.minimum(pos[:, idx["LeftToe"], 2], pos[:, idx["RightToe"], 2]), .05)
        body = side+"_ankle_roll_link"
        desired_r = target_rot[body]
        # Flatten only source-labelled support feet. Preserve swing foot tilt.
        zero = mujoco.MjData(model)
        mujoco.mj_forward(model, zero)
        r0 = zero.xmat[model.body(body).id].reshape(3, 3)
        heading = desired_r @ r0.T
        yaw_foot = np.arctan2(heading[:, 1, 0], heading[:, 0, 0])
        yaw_rate = np.gradient(np.unwrap(yaw_foot), 1/fps)
        # A whole-foot stationary anchor does not model ball-of-foot pivots.
        # Label only the slow, nearly-flat support phases as FULL-SOLE contacts;
        # leave pivots for a later toe/heel contact model instead of freezing them.
        contact = ((toe[:, 2] < floor + .045) & (speed < .15)
                   & (foot[:, 2] < human_ankle_z_reference + .07)
                   & (abs(yaw_rate) < .8) & (heading[:, 2, 2] > np.cos(np.deg2rad(20))))
        contact = binary_opening(np.pad(contact, (3, 3), mode="edge"), iterations=2)[3:-3]
        contacts[:, side_id] = contact
        for first, end in contact_intervals(contact):
            angle = float(np.median(np.unwrap(yaw_foot[first:end])))
            desired_r[first:end] = R.from_euler("z", angle).as_matrix() @ r0
        vertices = foot_meshes[side][1]
        sole_offset = -np.min(np.einsum("ti,vi->tv", desired_r[:, 2], vertices), axis=1) + .001
        target = foot * scale
        target[:, 2] = sole_offset + scale * np.maximum(0, foot[:, 2]-human_ankle_z_reference)
        # Match Light's wider hip spacing without uniformly stretching its legs.
        width_delta = .5 * ((zero.xpos[model.body("left_ankle_roll_link").id, 1] - zero.xpos[model.body("right_ankle_roll_link").id, 1])
                           - np.linalg.norm(positions[0, idx["LeftFoot"], :2]-positions[0, idx["RightFoot"], :2])*scale)
        target[:, :2] += (1 if side == "left" else -1) * width_delta * target_rot["pelvis_link"][:, :2, 1]
        for first, end in contact_intervals(contact):
            target[first:end, :2] = np.median(target[first:end, :2], axis=0)
            target[first:end, 2] = sole_offset[first:end]
        foot_targets[:, side_id] = target
    return root, target_rot, foot_targets, contacts, pos, times, scale


def bone_correspondences():
    return [(side+"_"+label, side.title()+h1, side.title()+h2, side+"_"+r1+"_link", side+"_"+r2+"_link")
            for side in ["left", "right"] for label, h1, h2, r1, r2 in [
                ("thigh", "UpLeg", "Leg", "hip_pitch", "knee"),
                ("shin", "Leg", "Foot", "knee", "ankle_pitch"),
                ("upper_arm", "Arm", "ForeArm", "shoulder_pitch", "elbow"),
                ("forearm", "ForeArm", "Hand", "elbow", "wrist")]]


def run_ik(mink, model, root, rotations, foot_targets, contacts, fps, meshes, speed_limits, human, human_names, hard_stance=False,
           flight_mask=None, root_position_cost=12., root_orientation_cost=8.):
    config = mink.Configuration(model)
    tasks, by_body = [], {}
    for body in rotations:
        cost = 1.0
        if body == "pelvis_link": cost = root_orientation_cost
        if body == "torso_roll_link": cost = 5.
        if body == "head_link": cost = .3
        position = 0.
        if body == "pelvis_link": position = root_position_cost
        if "ankle_roll" in body: position, cost = 150., 15.
        task = mink.FrameTask(frame_name=body, frame_type="body", position_cost=position,
                              orientation_cost=cost, lm_damping=.1)
        tasks.append(task)
        by_body[body] = task
    predictor = mink.PostureTask(model, cost=.5)
    predictor.set_target(model.qpos0)
    tasks.append(predictor)

    class BoneDirectionTask(mink.Task):
        def __init__(self, first, end):
            super().__init__(cost=np.full(3, 8.), gain=1., lm_damping=.1)
            self.first, self.end = model.body(first).id, model.body(end).id
            self.target = np.array([0., 0, -1])

        def compute_error(self, configuration):
            bone = configuration.data.xpos[self.end]-configuration.data.xpos[self.first]
            return bone/np.linalg.norm(bone)-self.target

        def compute_jacobian(self, configuration):
            data = configuration.data
            bone = data.xpos[self.end]-data.xpos[self.first]
            length = np.linalg.norm(bone)
            unit = bone/length
            ja, jb = np.empty((3, model.nv)), np.empty((3, model.nv))
            mujoco.mj_jacBody(model, data, ja, None, self.first)
            mujoco.mj_jacBody(model, data, jb, None, self.end)
            return (np.eye(3)-np.outer(unit, unit)) @ (jb-ja)/length

    bone_tasks = []
    for _, h1, h2, r1, r2 in bone_correspondences():
        task = BoneDirectionTask(r1, r2)
        target = human[:, human_names.index(h2)]-human[:, human_names.index(h1)]
        target /= np.linalg.norm(target, axis=-1, keepdims=True)
        bone_tasks.append((task, target))
        tasks.append(task)

    class StancePositionConstraint(mink.Task):
        """Holonomic no-slip constraint after contact-entry settling."""
        def __init__(self, body):
            super().__init__(cost=np.ones(3), gain=.95, lm_damping=0.)
            self.bid = model.body(body).id
            self.target = None

        def compute_error(self, configuration):
            return configuration.data.xpos[self.bid]-self.target

        def compute_jacobian(self, configuration):
            jac = np.empty((3, model.nv))
            mujoco.mj_jacBody(model, configuration.data, jac, None, self.bid)
            return jac

    support_constraints = [StancePositionConstraint(side+"_ankle_roll_link") for side in ["left", "right"]]
    ids = np.array([model.joint(name).qposadr[0] for name in speed_limits])
    dofs = np.array([model.joint(name).dofadr[0] for name in speed_limits])
    vmax = np.array(list(speed_limits.values()))

    class FrameSpeedLimit(mink.Limit):
        """Bound TOTAL frame displacement, across all inner IK iterations."""
        previous = None

        def compute_qp_inequalities(self, configuration, dt):
            if self.previous is None:
                return mink.Constraint()
            delta = configuration.q[ids] - self.previous[ids]
            projection = np.eye(model.nv)[dofs]
            return mink.Constraint(G=np.r_[projection, -projection], h=np.r_[vmax/fps-delta, vmax/fps+delta])

    speed = FrameSpeedLimit()

    class TrustRegion(mink.Limit):
        def compute_qp_inequalities(self, configuration, dt):
            bound = np.full(model.nv, .04)
            bound[:3] = .015
            return mink.Constraint(G=np.r_[np.eye(model.nv), -np.eye(model.nv)], h=np.r_[bound, bound])

    class GeometricLimit(mink.Limit):
        """Linearized distances in QP DISPLACEMENT units, including recovery.

        Explicit hull vertex heights also catch foot corners whose lowest
        vertex changes during an IK step. Negative current clearances create a
        recovery constraint instead of merely stopping further penetration.
        """
        def compute_qp_inequalities(self, configuration, dt):
            data = configuration.data
            rows, bounds = [], []
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            for gid, vertices in meshes:
                world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
                near = world[world[:, 2] < .026]
                if not len(near): continue
                bid = model.geom_bodyid[gid]
                mujoco.mj_jacBody(model, data, jp, jr, bid)
                radius = near-data.xpos[bid]
                derivative = jp[2] + radius[:, 0, None]*jr[1] - radius[:, 1, None]*jr[0]
                rows.extend(-derivative)
                bounds.extend(.9*(near[:, 2]-.0005))
            for g1, g2 in self_collision_pairs(model):
                points = np.empty(6)
                distance = mujoco.mj_geomDistance(model, data, g1, g2, .04, points)
                if distance >= .04-1e-10: continue
                normal = points[3:]-points[:3]
                length = np.linalg.norm(normal)
                if length < 1e-12: continue
                normal *= (1 if distance >= 0 else -1)/length
                jac1, jac2 = np.empty((3, model.nv)), np.empty((3, model.nv))
                mujoco.mj_jac(model, data, jac1, None, points[:3], model.geom_bodyid[g1])
                mujoco.mj_jac(model, data, jac2, None, points[3:], model.geom_bodyid[g2])
                rows.append(-normal @ (jac2-jac1))
                bounds.append(.85*(distance-.002))
            self.last = mink.Constraint(G=np.asarray(rows), h=np.asarray(bounds)) if rows else mink.Constraint()
            return self.last

    geometric = GeometricLimit()
    limits = [mink.ConfigurationLimit(model, gain=.95, min_distance_from_limits=.0001), speed, geometric, TrustRegion()]
    q = model.qpos0.copy()
    q[:3] = root[0]
    q[3:7] = R.from_matrix(rotations["pelvis_link"][0]).as_quat(scalar_first=True)
    config.update(q)
    # Place only the initial solver seed above the floor. Published frames are
    # solved with contact constraints; there is no per-frame height shifting.
    seed_bottom = min(float((vertices @ config.data.geom_xmat[g].reshape(3, 3)[2]+config.data.geom_xpos[g, 2]).min()) for g, vertices in meshes)
    q[2] += max(0., .002-seed_bottom)
    config.update(q)
    collision_pairs = self_collision_pairs(model)

    def clear_geometry():
        data = config.data
        if any(mujoco.mj_geomDistance(model, data, a, b, .04, None) < .00005 for a, b in collision_pairs):
            return False
        for gid, vertices in meshes:
            if (vertices @ data.geom_xmat[gid].reshape(3, 3)[2]+data.geom_xpos[gid, 2]).min() < .00005:
                return False
        return True

    if not clear_geometry():
        raise ValueError("Initial zero-joint seed is not collision-free under the selected screening pairs")
    trajectory = []
    dt = 1/fps
    start = time.perf_counter()
    for frame in range(len(root)):
        active_support = []
        for side, support in enumerate(support_constraints):
            stable = hard_stance and frame >= 3 and contacts[frame-3:frame+1, side].all()
            if stable:
                if support.target is None:
                    # Use the reached, collision-free support location as the
                    # anchor instead of teleporting the foot to a source median.
                    support.target = config.data.xpos[support.bid].copy()
                active_support.append(support)
            else:
                support.target = None
        for task, target in bone_tasks:
            task.target = target[frame]
        for body, task in by_body.items():
            target = root[frame] if body == "pelvis_link" else np.zeros(3)
            if "ankle_roll" in body:
                side_id = int(body.startswith("right"))
                target = foot_targets[frame, side_id]
                task.set_orientation_cost(100. if contacts[frame, side_id] else 15.)
                if flight_mask is not None:
                    flying = bool(flight_mask[frame])
                    task.set_position_cost(2. if flying else (150. if contacts[frame, side_id] else 25.))
                    task.set_orientation_cost(2. if flying else (100. if contacts[frame, side_id] else 10.))
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rotations[body][frame]), target))
        if trajectory:
            speed.previous = trajectory[-1]
            prediction = trajectory[-1].copy()
            if len(trajectory) > 1:
                prediction[7:] += .5*(trajectory[-1][7:]-trajectory[-2][7:])
            predictor.set_target(prediction)
        for iteration in range(150 if frame == 0 else 24):
            velocity = mink.solve_ik(config, tasks, dt, solver="daqp", damping=.05, limits=limits,
                                     constraints=active_support)
            bound = geometric.last
            if not bound.inactive and np.max(bound.G @ (velocity*dt)-bound.h) > 2e-5:
                raise RuntimeError("QP solver failed the geometric inequality residual check")
            previous = config.q.copy()
            accepted = False
            # The QP uses local linear geometry. Backtrack on the actual hulls
            # to retain a feasible iterate when support vertices/faces change.
            for halving in range(9):
                config.update(previous)
                config.integrate_inplace(velocity, dt/(2**halving))
                if clear_geometry():
                    accepted = True
                    break
            if not accepted:
                config.update(previous)
                break
            if np.linalg.norm(velocity*dt) < 1e-5:
                break
        trajectory.append(config.q.copy())
        if frame % 100 == 0:
            print(f"IK {frame+1}/{len(root)} ({time.perf_counter()-start:.1f}s)", flush=True)
    return np.asarray(trajectory)


def differentiate(model, qpos, fps):
    velocity = np.empty((len(qpos), model.nv))
    for t in range(len(qpos)):
        lo, hi = max(0, t-1), min(len(qpos)-1, t+1)
        mujoco.mj_differentiatePos(model, velocity[t], (hi-lo)/fps, qpos[lo], qpos[hi])
    return velocity


def quality(model, qpos, qvel, target_r, foot_targets, contacts, fps, meshes, speed_limits, human, human_names, root_target, hard_stance=False):
    data = mujoco.MjData(model)
    bottom, footpos, errors, self_distances, bone_errors = [], [], [], [], []
    pairs = self_collision_pairs(model)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        bottom.append(min(float((vertices @ data.geom_xmat[g].reshape(3, 3)[2]+data.geom_xpos[g, 2]).min()) for g, vertices in meshes))
        footpos.append([data.xpos[model.body(side+"_ankle_roll_link").id].copy() for side in ["left", "right"]])
        errors.append([R.from_matrix(data.xmat[model.body(body).id].reshape(3, 3) @ r[frame].T).magnitude() for body, r in target_r.items()])
        self_distances.append(min(mujoco.mj_geomDistance(model, data, a, b, .04, None) for a, b in pairs))
        bone_frame = []
        for _, h1, h2, r1, r2 in bone_correspondences():
            h = human[frame, human_names.index(h2)]-human[frame, human_names.index(h1)]
            r = data.xpos[model.body(r2).id]-data.xpos[model.body(r1).id]
            cosine = np.dot(h, r)/np.linalg.norm(h)/np.linalg.norm(r)
            bone_frame.append(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))
        bone_errors.append(bone_frame)
    footpos, errors = np.array(footpos), np.rad2deg(errors)
    speed = np.linalg.norm(np.gradient(footpos, 1/fps, axis=0)[:, :, :2], axis=-1)
    interior = np.array([binary_erosion(contacts[:, j], iterations=4 if hard_stance else 2) for j in range(2)]).T
    interior[:2] = interior[-2:] = False
    excess, max_speed_ratio = 0., 0.
    for name, limit in speed_limits.items():
        j = model.joint(name)
        a = qpos[:, j.qposadr[0]]
        excess = max(excess, float(np.maximum(j.range[0]-a, a-j.range[1]).max()))
        max_speed_ratio = max(max_speed_ratio, float(abs(qvel[:, j.dofadr[0]]).max()/limit))
    stance_speed = speed[interior]
    foot_error = np.linalg.norm(footpos-foot_targets, axis=-1)
    root_error = np.linalg.norm(qpos[:, :3]-root_target, axis=-1)
    per_body = {b: {"median_deg": float(np.median(errors[:, i])), "p95_deg": float(np.quantile(errors[:, i], .95)), "max_deg": float(errors[:, i].max())} for i, b in enumerate(target_r)}
    issues = []
    if excess > 1e-6: issues.append("Joint limit violation")
    if min(bottom) < -.001: issues.append("Floor penetration exceeds 1 mm")
    if len(stance_speed) and np.quantile(stance_speed, .95) > .05: issues.append("Inferred stance foot speed p95 exceeds 5 cm/s")
    if max_speed_ratio > 1.001: issues.append("Joint speed exceeds configured limit")
    if min(self_distances) < -.001: issues.append("Selected self-collision hulls penetrate over 1 mm")
    if contacts.any() and np.quantile(foot_error[contacts], .95) > .05:
        issues.append("Source support position error p95 exceeds 5 cm")
    if np.quantile(root_error, .95) > .1:
        issues.append("Source root trajectory error p95 exceeds 10 cm")
    return {"status": "kinematic_sample_needs_review" if not issues else "kinematic_sample_needs_correction", "training_ready": False,
        "issues": issues, "frames": len(qpos), "fps": fps, "duration_s": (len(qpos)-1)/fps,
        "min_collision_mesh_height_m": min(bottom), "max_joint_limit_excess_rad": max(0, excess),
        "min_selected_self_collision_distance_m": min(self_distances), "selected_self_collision_pairs": len(pairs),
        "max_joint_speed_to_limit_ratio": max_speed_ratio, "source_contact_foot_frames": contacts.sum(axis=0).tolist(),
        "stance_foot_speed_m_s": {"median": float(np.median(stance_speed)), "p95": float(np.quantile(stance_speed,.95)), "max": float(stance_speed.max())} if len(stance_speed) else None,
        "stance_foot_target_error_m": {"p95": float(np.quantile(foot_error[contacts],.95)), "max": float(foot_error[contacts].max())} if contacts.any() else None,
        "orientation_tracking": per_body,
        "root_tracking_error_m": {"median": float(np.median(root_error)), "p95": float(np.quantile(root_error, .95)), "max": float(root_error.max())},
        "bone_direction_tracking": {name: {"median_deg": float(np.median(np.asarray(bone_errors)[:, i])),
                 "p95_deg": float(np.quantile(np.asarray(bone_errors)[:, i], .95))}
                 for i, (name, *_) in enumerate(bone_correspondences())},
        "limitations": ["Source contacts are inferred from flat-ground height/speed thresholds, not ground-truth force labels.",
                        "Sequential differential IK with a predictor regularizer; not a whole-trajectory optimum.",
                        "Self-collision screening covers forearm/hand vs torso/head, both arms, and opposing shins/feet; not every link pair.",
                        "No torque limits, coupled motor PD or dynamic tracking validation in this kinematic pass."]}


def render(model, qpos, human, names, fps, path, human_scale=None):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    option, camera, data = mujoco.MjvOption(), mujoco.MjvCamera(), mujoco.MjData(model)
    option.geomgroup[:] = 0
    option.geomgroup[[0, 2]] = 1
    chains = [["Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe"],
              ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToe"],
              ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"],
              ["Spine2", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"],
              ["Spine2", "RightShoulder", "RightArm", "RightForeArm", "RightHand"]]
    edges = [(names.index(a), names.index(b)) for chain in chains for a, b in zip(chain, chain[1:])]
    scale = human_scale if human_scale is not None else json.loads((OUTPUT / "light_profile.json").read_text())["morphology_reference"]["leg_scale"]
    human = human.copy()*scale
    human[:, :, 2] -= np.quantile(human[:, [names.index("LeftToe"), names.index("RightToe")], 2], .05)
    human[:, :, 1] += .85
    with mujoco.Renderer(model, height=600, width=800) as renderer, imageio.get_writer(path, fps=25, codec="libx264", quality=7, macro_block_size=1) as writer:
        for frame in range(0, len(qpos), max(1, round(fps/25))):
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            human_root = human[frame, names.index("Hips")]
            camera.lookat[:] = .5*(qpos[frame, :3]+human_root)
            camera.lookat[2] = .7
            camera.distance, camera.azimuth, camera.elevation = 3.5, 10, -12
            renderer.update_scene(data, camera=camera, scene_option=option)
            for a, b in edges:
                scene = renderer.scene
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                   np.eye(3).ravel(), np.array([1., .55, .12, 1.], dtype=np.float32))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .012, human[frame, a], human[frame, b])
                scene.ngeom += 1
            array = renderer.render()
            image = Image.fromarray(array)
            ImageDraw.Draw(image).text((15, 15), f"{path.stem} | {frame/fps:.2f}s | Orange: original human / Blue: Light IK", fill="white")
            writer.append_data(np.asarray(image))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", action="append", help="BVH stem:start_seconds:duration_seconds (repeatable)")
    parser.add_argument("--fps", type=float, default=50.)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--hard-stance", action="store_true", help="Experimental reached-foot equality anchors; can trap sequential IK, always check source tracking errors")
    args = parser.parse_args()
    mink, extract, utils = prepare()
    asset = ROOT / "outputs/humanoid_light_conversion/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(asset))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    offsets, anatomical = calibrated_offsets(model, data)
    meshes, feet = geometry(model)
    raw_dir = ROOT / "src/humanoid_retarget/datasets/lafan1/raw"
    ref_names, ref_p, _, ref_offsets, _ = load_human(raw_dir / "walk1_subject1.bvh", extract, utils)
    robot_leg = sum(np.linalg.norm(data.xpos[model.body(f"left_{end}_link").id] - data.xpos[model.body(f"left_{first}_link").id]) for first, end in [("hip_pitch", "knee"), ("knee", "ankle_pitch")])
    human_leg = sum(np.linalg.norm(ref_offsets[ref_names.index(name)]) for name in ["LeftLeg", "LeftFoot"])
    scale = robot_leg/human_leg
    reference = (scale, float(data.qpos[2]), float(ref_p[0, ref_names.index("Hips"), 2]),
                 float(np.mean([ref_p[0, ref_names.index(n), 2] for n in ["LeftFoot", "RightFoot"]])))
    controls = json.loads((asset.parent / "control.json").read_text())["groups"]
    joint_names = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    speeds = {}
    for name in joint_names:
        for group in controls.values():
            if any(re.fullmatch(pattern, name) for pattern in group["joint_names_expr"]):
                for pattern, value in group["velocity_limit"].items():
                    if re.fullmatch(pattern, name): speeds[name] = value
    if set(speeds) != set(joint_names): raise ValueError("Missing joint speed limits")
    profile = {"revision": 2, "source": "original LAFAN1 BVH", "algorithm": "Mink sequential QP IK with calibrated GMR BVH axes",
               "driver_sha256": digest(Path(__file__)), "backend_versions": {"mink": "1.3.0", "qpsolvers": "4.13.0", "daqp": "0.10.3", "mujoco": mujoco.__version__},
               "gmr_commit": REVISION, "rotation_offsets": offsets, "anatomical_calibration": anatomical,
               "morphology_reference": {"leg_scale": scale, "robot_leg_m": robot_leg, "human_leg_m": human_leg,
                    "robot_root_z_m": reference[1], "human_root_z_m": reference[2], "human_ankle_z_m": reference[3]},
               "velocity_limits_joint_coordinate_rad_s": speeds, "contact_height_threshold_m": .045,
               "contact_xy_speed_threshold_m_s": .15, "contact_yaw_speed_threshold_rad_s": .8,
               "contact_max_foot_tilt_deg": 20., "contact_model": "inferred full-sole stationary support; toe/heel pivots are not fixed anchors",
               "stance_anchor": "median source foot position per inferred contact interval",
               "no_slip_constraint": "hard reached-foot position equality after 3 entry frames (experimental)" if args.hard_stance else "constant stance pose target with high cost; residual slip is screened",
               "floor_clearance_constraint_m": .0005, "foot_target_clearance_m": .001,
               "self_clearance_constraint_m": .002, "self_collision_geom_pairs": self_collision_pairs(model),
               "max_inner_step_rad": .04, "max_inner_root_translation_m": .015,
               "max_inner_iterations": 24, "initialization_iterations": 150,
               "mesh_distance_backend": "MuJoCo native CCD, 4 cm query cutoff; nonlinear geometry backtracking",
               "segment_direction_cost": 8., "limb_orientation_cost": 1.,
               "foot_orientation_cost_stance": 100., "foot_orientation_cost_swing": 15.,
               "training_ready": False}
    (OUTPUT / "light_profile.json").write_text(json.dumps(profile, indent=2)+"\n")
    reports = {}
    for spec in args.clip or ["walk1_subject1:5:5", "dance1_subject2:5:5", "fight1_subject2:5:5"]:
        name, start, duration = spec.split(":")
        source = raw_dir / f"{name}.bvh"
        names, positions, rotations, _, source_fps = load_human(source, extract, utils)
        root, target_r, target_p, contact, human, times, _ = targets_for_clip(names, positions, rotations, source_fps,
                      float(start), float(duration), offsets, model, reference, feet, args.fps)
        label = f"{name}_{float(start):g}s_{float(duration):g}s"
        print(label, "scale", scale, "stance", contact.sum(axis=0).tolist(), flush=True)
        started = time.perf_counter()
        qpos = run_ik(mink, model, root, target_r, target_p, contact, args.fps, meshes, speeds, human, names, args.hard_stance)
        elapsed = time.perf_counter()-started
        qvel = differentiate(model, qpos, args.fps)
        report = quality(model, qpos, qvel, target_r, target_p, contact, args.fps, meshes, speeds, human, names, root, args.hard_stance)
        report["ik_wall_seconds"] = elapsed
        report.update(source=str(source.relative_to(ROOT)), source_sha256=digest(source), source_start_seconds=float(start),
                      source_fps=source_fps, robot_xml_sha256=digest(asset.parent / "humanoid_light.xml"),
                      profile_sha256=digest(OUTPUT / "light_profile.json"))
        path = OUTPUT / label
        np.savez_compressed(path.with_suffix(".npz"), qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32),
                            frequency=np.asarray(args.fps), joint_names=np.asarray(["root"]+joint_names),
                            split_points=np.array([0, len(qpos)]), source_contact=contact,
                            source_sha256=np.asarray(report["source_sha256"]), training_ready=np.asarray(False))
        np.savez_compressed(OUTPUT / (label+"_targets.npz"), source_time=times, root=root, feet=target_p, contacts=contact,
                            rotation_body_names=np.asarray(list(target_r)), rotations=np.stack(list(target_r.values()), axis=1),
                            human_positions=human, human_joint_names=np.asarray(names))
        path.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
        reports[label] = report
        print(label, json.dumps({k:v for k,v in report.items() if k not in {"orientation_tracking", "bone_direction_tracking", "limitations"}}), flush=True)
        if not args.no_render:
            render(model, qpos, human, names, args.fps, path.with_suffix(".mp4"))
    (OUTPUT / "samples.json").write_text(json.dumps(reports, indent=2)+"\n")


if __name__ == "__main__":
    main()
