"""Audit untrusted Isaac reference motions without modifying or training them.

Compare the stored world link transforms with independent FK in the ORIGINAL
USD link frames. Separately replay root/joint states in the converted MJCF to
measure collision mesh heights. This avoids mistaking URDF frame changes for
retargeting errors. Foot support/sliding thresholds are diagnostics, not proof
of dynamic feasibility or fidelity to an unavailable human source motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from validate_humanoid_light import transform


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "cocelo-rl-isaaclab/lab/cocelo/assets/data/Reference"


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"min": float(a.min()), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max()),
            "rms": float(np.sqrt(np.mean(a*a)))}


def rotations(quaternions):
    q = np.asarray(quaternions)
    return Rotation.from_quat(q.reshape(-1, 4), scalar_first=True).as_matrix().reshape(*q.shape[:-1], 3, 3)


def original_fk(source, names, angles, root_position, root_quaternion):
    n = len(angles)
    root_path = next(b["path"] for b in source["bodies"] if b["name"] == "pelvis_link")
    root = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    root[:, :3, :3] = rotations(root_quaternion)
    root[:, :3, 3] = root_position
    poses = {root_path: root}
    remaining = list(source["joints"])
    while remaining:
        progressed = False
        for j in list(remaining):
            parent, child = j["parent"][0], j["child"][0]
            if parent not in poses:
                continue
            a = transform(j["local_pos0"], j["local_rot0_wxyz"])
            b_inv = np.linalg.inv(transform(j["local_pos1"], j["local_rot1_wxyz"]))
            motion = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            if j["name"] in names:
                axis = np.eye(3)["XYZ".index(j["axis"])]
                motion[:, :3, :3] = Rotation.from_rotvec(angles[:, names.index(j["name"]), None] * axis).as_matrix()
            poses[child] = poses[parent] @ a @ motion @ b_inv
            remaining.remove(j)
            progressed = True
        if not progressed:
            raise ValueError("Unresolved source FK tree")
    return {b["name"]: poses[b["path"]] for b in source["bodies"]}


def qpos_from_reference(model, raw):
    names, bodies = raw["joint_names"].tolist(), raw["body_names"].tolist()
    rid = bodies.index("pelvis_link")
    qpos = np.tile(model.qpos0, (len(raw["joint_pos"]), 1))
    qpos[:, :3], qpos[:, 3:7] = raw["body_pos_w"][:, rid], raw["body_quat_w"][:, rid]
    for i, name in enumerate(names):
        qpos[:, model.joint(name).qposadr[0]] = raw["joint_pos"][:, i]
    return qpos


def mesh_geometry(model):
    result = []
    for gid in range(model.ngeom):
        if model.geom_group[gid] != 3 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        result.append((gid, model.body(model.geom_bodyid[gid]).name,
                       np.asarray(model.mesh_vert[start:start+count], dtype=float)))
    return result


def inspect(path, model, source):
    with np.load(path, allow_pickle=False) as archive:
        raw = {k: archive[k] for k in archive.files}
    names, bodies = raw["joint_names"].tolist(), raw["body_names"].tolist()
    angles = raw["joint_pos"].astype(float)
    n, fps = len(angles), float(raw["fps"])
    dt = 1 / fps
    expected = {model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
    if set(names) != expected or len(names) != len(expected):
        raise ValueError(f"{path}: joint names differ from the robot")
    if set(bodies) != {b["name"] for b in source["bodies"]}:
        raise ValueError(f"{path}: body names differ from USD")
    finite = all(np.isfinite(a).all() for a in raw.values() if a.dtype.kind in "fiu")
    if not finite:
        raise ValueError(f"{path}: non-finite values")
    rid = bodies.index("pelvis_link")
    root_p, root_q = raw["body_pos_w"][:, rid], raw["body_quat_w"][:, rid]
    poses = original_fk(source, names, angles, root_p, root_q)
    p_error, r_error, com_error = [], [], []
    source_bodies = {b["name"]: b for b in source["bodies"]}
    for i, name in enumerate(bodies):
        pose = poses[name]
        p_error.append(np.linalg.norm(pose[:, :3, 3] - raw["body_pos_w"][:, i], axis=-1))
        recorded = rotations(raw["body_quat_w"][:, i])
        difference = pose[:, :3, :3] @ np.swapaxes(recorded, -1, -2)
        r_error.append(Rotation.from_matrix(difference).magnitude())
        com = source_bodies[name]["center_of_mass"] or [0, 0, 0]
        predicted_com = pose[:, :3, 3] + np.einsum("tij,j->ti", pose[:, :3, :3], com)
        com_error.append(np.linalg.norm(predicted_com - raw["body_pos_w"][:, i], axis=-1))
    p_error, r_error = np.array(p_error).T, np.array(r_error).T
    lo, hi = np.array([model.joint(name).range for name in names]).T
    violation = np.maximum(lo - angles, 0) + np.maximum(angles - hi, 0)
    per_joint = {name: {"min_rad": float(angles[:, i].min()), "max_rad": float(angles[:, i].max()),
                       "max_limit_excess_rad": float(violation[:, i].max()),
                       "max_speed_rad_s": float(abs(raw["joint_vel"][:, i]).max())}
                 for i, name in enumerate(names)}
    lin_fd = np.gradient(raw["body_pos_w"].astype(float), dt, axis=0)
    rot = rotations(raw["body_quat_w"])
    rel = rot[1:] @ np.swapaxes(rot[:-1], -1, -2)
    omega = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(n-1, len(bodies), 3) / dt
    stored_omega = (raw["body_ang_vel_w"][1:] + raw["body_ang_vel_w"][:-1]) / 2
    qpos = qpos_from_reference(model, raw)
    data, meshes = mujoco.MjData(model), mesh_geometry(model)
    sole = np.empty((n, 2))
    foot_origins = np.empty((n, 2, 3))
    foot_geom_rot = np.empty((n, 2, 3, 3))
    foot_geom_pos = np.empty((n, 2, 3))
    sole_local_point = np.empty((n, 2, 3))
    all_bottom = np.empty(n)
    feet = [model.body(side + "_ankle_roll_link").id for side in ["left", "right"]]
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        bottom = np.inf
        for gid, name, vertices in meshes:
            height = vertices @ data.geom_xmat[gid].reshape(3, 3)[2] + data.geom_xpos[gid, 2]
            h = float(height.min())
            bottom = min(bottom, h)
            if name in ["left_ankle_roll_link", "right_ankle_roll_link"]:
                side = int(name.startswith("right"))
                sole[frame, side] = h
                foot_geom_rot[frame, side] = data.geom_xmat[gid].reshape(3, 3)
                foot_geom_pos[frame, side] = data.geom_xpos[gid]
                sole_local_point[frame, side] = vertices[height.argmin()]
        all_bottom[frame] = bottom
        foot_origins[frame] = data.xpos[feet]
    foot_velocity = np.gradient(foot_origins, dt, axis=0)
    # This diagnostic assumes a flat floor at world z=0. It may flag intended
    # jumps; report counts rather than assert that every aerial frame is bad.
    stance = (abs(sole) < .02) & (abs(foot_velocity[:, :, 2]) < .1)
    slide = np.linalg.norm(foot_velocity[:, :, :2], axis=-1)
    # Velocity of the material vertex currently nearest the ground, using the
    # derivative of its rigid transform. Differentiating the minimum-height
    # point trajectory itself would introduce errors when the vertex changes.
    vertex_velocity = (np.einsum("tsij,tsj->tsi", np.gradient(foot_geom_rot, dt, axis=0), sole_local_point)
                       + np.gradient(foot_geom_pos, dt, axis=0))
    vertex_speed = np.linalg.norm(vertex_velocity[:, :, :2], axis=-1)
    worst = np.unravel_index(np.argmax(p_error), p_error.shape)
    report = {
        "source": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "frames": n, "fps": fps, "duration_seconds": (n - 1) / fps, "finite": finite,
        "max_quaternion_norm_error": float(abs(np.linalg.norm(raw["body_quat_w"], axis=-1)-1).max()),
        "joint_limits": {"tolerance_rad": .001, "exceeding_frame_count": int((violation > .001).any(axis=1).sum()),
                         "max_excess_rad": float(violation.max()), "per_joint": per_joint},
        "original_usd_frame_fk": {"position_error_m": stats(p_error), "rotation_error_rad": stats(r_error),
             "position_if_recorded_points_were_com_m": stats(com_error),
             "per_body": {b: {"position_error_m": stats(p_error[:, i]), "rotation_error_rad": stats(r_error[:, i])}
                          for i, b in enumerate(bodies)},
             "worst_position_body": bodies[worst[1]], "worst_position_frame": int(worst[0])},
        "temporal": {"joint_velocity_fd_error_rad_s": stats(np.gradient(angles, dt, axis=0)-raw["joint_vel"]),
                     "body_linear_velocity_fd_error_m_s": stats(lin_fd-raw["body_lin_vel_w"]),
                     "body_angular_velocity_fd_error_rad_s": stats(omega-stored_omega),
                     "max_joint_step_rad": float(abs(np.diff(angles, axis=0)).max())},
        "root": {"xyz_min": root_p.min(axis=0).tolist(), "xyz_max": root_p.max(axis=0).tolist(),
                 "xy_path_length_m": float(np.linalg.norm(np.diff(root_p[:, :2], axis=0), axis=-1).sum())},
        "collision_geometry": {"floor_z_m": 0, "left_sole_height_m": stats(sole[:, 0]),
                 "right_sole_height_m": stats(sole[:, 1]), "all_robot_min_height_m": stats(all_bottom),
                 "penetration_over_1cm_frames": int((all_bottom < -.01).sum()),
                 "both_feet_over_3cm_frames": int((sole.min(axis=1) > .03).sum()),
                 "inferred_stance_foot_frames": int(stance.sum()),
                 "inferred_stance_xy_speed_m_s": stats(slide[stance]) if stance.any() else None,
                 "inferred_stance_speed_over_0_1m_s_frames": int(((slide > .1) & stance).sum()),
                 "lowest_material_vertex_stance_xy_speed_m_s": stats(vertex_speed[stance]) if stance.any() else None,
                 "lowest_material_vertex_stance_speed_over_0_1m_s_frames": int(((vertex_speed > .1) & stance).sum())},
        "limitations": ["No original human source or retarget configuration is attached to the four NPZ files.",
             "Checks validate internal kinematics and geometry, not dynamic feasibility or human-motion fidelity.",
             "The sole-height support/sliding test is a flat-ground heuristic; intended jumps require separate labels.",
             "The original USD link frames are used for FK comparison, not the changed URDF/MJCF body frames."],
    }
    reasons = []
    if p_error.max() > .001 or r_error.max() > .01:
        reasons.append("Stored body transforms disagree with source USD forward kinematics")
    if (violation > .001).any():
        reasons.append("Joint position limit excess over 0.001 rad")
    if (all_bottom < -.01).any():
        reasons.append("Collision mesh penetrates flat floor by over 1 cm")
    if (sole.min(axis=1) > .03).mean() > .2:
        reasons.append("Both collision soles above 3 cm for over 20% of frames; review intended floor/jump labels")
    if stance.any() and ((vertex_speed > .1) & stance).sum() / stance.sum() > .1:
        reasons.append("Lowest sole material vertex moves over 0.1 m/s in over 10% of inferred stance samples")
    report["review_reasons"] = reasons
    report["screening_status"] = "hold_for_review" if reasons else "kinematic_checks_passed_dynamic_and_source_review_needed"
    return report, raw, qpos, sole


def render_montage(model, audited, output):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    cell_w, cell_h = 360, 400
    canvas = Image.new("RGB", (cell_w * 5, (cell_h + 35) * len(audited)), "#edf1f4")
    option, camera, data = mujoco.MjvOption(), mujoco.MjvCamera(), mujoco.MjData(model)
    option.geomgroup[:] = 0
    option.geomgroup[[0, 2]] = 1
    with mujoco.Renderer(model, height=cell_h, width=cell_w) as renderer:
        for row, (name, report, qpos, sole) in enumerate(audited):
            frames = np.unique(np.r_[np.linspace(0, len(qpos)-1, 4, dtype=int), np.argmin(sole.min(axis=1))])
            if len(frames) < 5:
                frames = np.linspace(0, len(qpos)-1, 5, dtype=int)
            for col, frame in enumerate(frames):
                data.qpos[:] = qpos[frame]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = [*qpos[frame, :2], .7]
                yaw = Rotation.from_quat(qpos[frame, 3:7], scalar_first=True).as_euler("xyz")[2]
                camera.distance, camera.azimuth, camera.elevation = 2.5, np.rad2deg(yaw) + 35, -10
                renderer.update_scene(data, camera=camera, scene_option=option)
                top = row * (cell_h + 35)
                canvas.paste(Image.fromarray(renderer.render()), (col * cell_w, top + 35))
                label = f"{name} {frame/report['fps']:.2f}s sole:{sole[frame].min()*100:+.1f}cm"
                ImageDraw.Draw(canvas).text((col * cell_w + 8, top + 8), label, fill="#1b2c38", font=font)
    path = output / "motion_montage.png"
    canvas.save(path)
    return path


def plot_heights(audited, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(audited), 1, figsize=(12, 8), constrained_layout=True)
    for axis, (name, report, _, sole) in zip(np.atleast_1d(axes), audited):
        time = np.arange(len(sole)) / report["fps"]
        axis.plot(time, sole[:, 0] * 100, label="Left sole")
        axis.plot(time, sole[:, 1] * 100, label="Right sole", alpha=.8)
        axis.axhline(0, color="black", linewidth=.7)
        axis.set(title=name, ylabel="Height (cm)", xlabel="Time (s)")
        axis.grid(alpha=.25)
    axes[0].legend(loc="upper right")
    fig.savefig(output / "sole_heights.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE)
    parser.add_argument("--asset-dir", type=Path, default=ROOT / "outputs/humanoid_light_conversion")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/humanoid_light_motion_audit")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.asset_dir / "source_properties.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(args.asset_dir / "scene.xml"))
    reports, audited = {}, []
    for name in ["wave", "dance", "stretch", "boxing"]:
        report, raw, qpos, sole = inspect(args.reference_dir / f"{name}.npz", model, source)
        reports[name] = report
        audited.append((name, report, qpos, sole))
        (args.output_dir / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, json.dumps({"status": report["screening_status"], "fk_max_m": report["original_usd_frame_fk"]["position_error_m"]["max"],
            "fk_max_rad": report["original_usd_frame_fk"]["rotation_error_rad"]["max"],
            "geometry": report["collision_geometry"]}), flush=True)
    (args.output_dir / "audit.json").write_text(json.dumps(reports, indent=2) + "\n")
    plot_heights(audited, args.output_dir)
    if not args.no_render:
        print(render_montage(model, audited, args.output_dir), flush=True)


if __name__ == "__main__":
    main()
