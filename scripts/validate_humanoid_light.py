"""Compare the converted Humanoid Light against USD properties and render it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def rotation(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def transform(position, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation(quaternion)
    matrix[:3, 3] = position
    return matrix


def source_forward(source, angles, base_position):
    """Independent FK from USD joint frames, including fixed sensor links."""
    children = {j["child"][0] for j in source["joints"]}
    roots = [b for b in source["bodies"] if b["path"] not in children]
    if len(roots) != 1:
        raise ValueError("Source does not form a single floating-base tree")
    root = roots[0]
    initial = np.asarray(root["world_transform"]).copy()
    initial[:3, 3] += base_position
    poses = {root["path"]: initial}
    pending = list(source["joints"])
    while pending:
        progress = False
        for joint in list(pending):
            parent, child = joint["parent"][0], joint["child"][0]
            if parent not in poses:
                continue
            a = transform(joint["local_pos0"], joint["local_rot0_wxyz"])
            b = transform(joint["local_pos1"], joint["local_rot1_wxyz"])
            motion = np.eye(4)
            if joint["name"] in angles:
                axis = np.eye(3)["XYZ".index(joint["axis"])]
                half = angles[joint["name"]] / 2
                motion[:3, :3] = rotation([np.cos(half), *(axis * np.sin(half))])
            poses[child] = poses[parent] @ a @ motion @ np.linalg.inv(b)
            pending.remove(joint)
            progress = True
        if not progress:
            raise ValueError("Unresolved source joint tree")
    return poses


def compare_source(output: Path, samples: int):
    source = json.loads((output / "source_properties.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(output / "humanoid_light.xml"))
    data = mujoco.MjData(model)
    hinges = [j for j in source["joints"] if j["type"] == "PhysicsRevoluteJoint"]
    expected_names = {j["name"] for j in hinges}
    actual_names = {model.joint(i).name for i in range(model.njnt)
                    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE}
    assert actual_names == expected_names, (actual_names, expected_names)
    assert (model.nbody, model.nq, model.nv, model.nu) == (len(source["bodies"]) + 1, 33, 32, 26)
    assert set(model.actuator_trnid[:, 0]) == {model.joint(n).id for n in expected_names}
    assert np.all(model.actuator_gainprm[:, 0] == 1), "Torque input gain is not unity"
    result = {"mujoco_version": mujoco.__version__, "source_body_count": len(source["bodies"]),
              "source_revolute_joints": len(hinges), "source_fixed_joints": len(source["joints"]) - len(hinges),
              "mujoco_body_count_including_world": model.nbody,
              "nq": model.nq, "nv": model.nv, "nu": model.nu,
              "source_mass_kg": source["total_mass"], "mujoco_mass_kg": float(model.body_mass.sum()),
              "mass_error_kg": abs(float(model.body_mass.sum()) - source["total_mass"])}
    assert result["mass_error_kg"] < 1e-4, result
    limits_error = max(float(np.abs(model.joint(j["name"]).range -
        np.deg2rad([j["lower_degrees"], j["upper_degrees"]])).max()) for j in hinges)
    result["max_joint_limit_error_rad"] = limits_error
    assert limits_error < 1e-6, limits_error
    mass_error, com_error, inertia_error, axis_error, anchor_error = [], [], [], [], []
    rng = np.random.default_rng(27)
    base_position = model.qpos0[:3].copy()
    for sample in range(samples + 1):
        angles = {}
        data.qpos[:] = model.qpos0
        for j in hinges:
            lower, upper = np.deg2rad([j["lower_degrees"], j["upper_degrees"]])
            value = 0.0 if sample == 0 else rng.uniform(max(lower, -0.3), min(upper, 0.3))
            angles[j["name"]] = value
            data.qpos[model.joint(j["name"]).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        poses = source_forward(source, angles, base_position)
        for body in source["bodies"]:
            bid = model.body(body["name"]).id
            mass_error.append(abs(model.body_mass[bid] - (body["mass"] or 0)))
            if not body["mass"]:
                continue
            pose = poses[body["path"]]
            com = pose[:3, :3] @ body["center_of_mass"] + pose[:3, 3]
            com_error.append(float(np.linalg.norm(data.xipos[bid] - com)))
            principal = pose[:3, :3] @ rotation(body["principal_axes_wxyz"])
            expected_inertia = principal @ np.diag(body["diagonal_inertia"]) @ principal.T
            actual_rotation = data.ximat[bid].reshape(3, 3)
            actual_inertia = actual_rotation @ np.diag(model.body_inertia[bid]) @ actual_rotation.T
            inertia_error.append(float(np.abs(actual_inertia - expected_inertia).max()))
        for j in hinges:
            jid = model.joint(j["name"]).id
            pose = poses[j["parent"][0]] @ transform(j["local_pos0"], j["local_rot0_wxyz"])
            axis = pose[:3, :3] @ np.eye(3)["XYZ".index(j["axis"])]
            axis_error.append(float(np.linalg.norm(data.xaxis[jid] - axis)))
            anchor_error.append(float(np.linalg.norm(data.xanchor[jid] - pose[:3, 3])))
    result.update(random_joint_configurations=samples,
                  max_body_mass_error_kg=max(mass_error), max_com_fk_error_m=max(com_error),
                  max_world_inertia_error_kg_m2=max(inertia_error),
                  max_joint_axis_vector_error=max(axis_error), max_joint_anchor_error_m=max(anchor_error))
    assert max(mass_error) < 1e-5, result
    assert max(com_error) < 2e-4, result
    assert max(inertia_error) < 2e-5, result
    assert max(axis_error) < 2e-4, result
    assert max(anchor_error) < 2e-4, result
    return result


def dynamics(output: Path):
    model = mujoco.MjModel.from_xml_path(str(output / "humanoid_light.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[2] += 2
    mujoco.mj_forward(model, data)
    center = lambda: np.sum(data.xipos * model.body_mass[:, None], axis=0) / model.body_mass.sum()
    initial = center().copy()
    for _ in range(100):
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    mujoco.mj_forward(model, data)
    expected = initial + 0.5 * model.opt.gravity * data.time**2
    freefall_error = float(np.linalg.norm(center() - expected))
    assert freefall_error < 0.005, freefall_error
    scene = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    state = mujoco.MjData(scene)
    contacts = 0
    for _ in range(500):
        mujoco.mj_step(scene, state)
        contacts = max(contacts, state.ncon)
        assert np.isfinite(state.qpos).all() and np.isfinite(state.qvel).all()
    bad = {name: int(state.warning[getattr(mujoco.mjtWarning, name)].number)
           for name in ["mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC"]}
    assert not any(bad.values()), bad
    assert contacts > 0, "Robot did not contact the floor"
    return {"freefall_seconds": float(data.time), "freefall_com_error_m": freefall_error,
            "ground_contact_seconds": float(state.time), "max_contacts": int(contacts),
            "numerical_warnings": bad, "control": "zero torque; no standing-policy claim"}


def render(output: Path):
    from PIL import Image, ImageDraw, ImageFont

    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    options = mujoco.MjvOption()
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    outputs = []
    with mujoco.Renderer(model, height=800, width=600) as renderer:
        for collision in [False, True]:
            options.geomgroup[:] = 0
            options.geomgroup[0] = 1
            options.geomgroup[3 if collision else 2] = 1
            if collision:
                model.geom_rgba[model.geom_group == 3] = [0.9, 0.5, 0.18, 1]
            canvas = Image.new("RGB", (1800, 850), "#e9ecef")
            for index, (label, angle) in enumerate([("Front", 0), ("Side", 90), ("Perspective", 35)]):
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [0, 0, 0.7]
                camera.distance, camera.azimuth, camera.elevation = 2.3, angle, -8
                renderer.update_scene(data, camera=camera, scene_option=options)
                canvas.paste(Image.fromarray(renderer.render()), (index * 600, 50))
                ImageDraw.Draw(canvas).text((index * 600 + 20, 12), label, fill="#17242d", font=font)
            path = output / ("collision_preview.png" if collision else "preview.png")
            canvas.save(path)
            outputs.append(str(path))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/humanoid_light_conversion")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    output = args.asset_dir.resolve()
    result = {"status": "failed"}
    try:
        result["comparison"] = compare_source(output, args.samples)
        print("SOURCE_COMPARISON", json.dumps(result["comparison"]), flush=True)
        result["dynamics"] = dynamics(output)
        print("DYNAMICS", json.dumps(result["dynamics"]), flush=True)
        if not args.no_render:
            result["previews"] = render(output)
        result["disk_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        result["limitations"] = ["PhysX and MuJoCo use different contact/hull solvers.",
            "Delayed PD and coupled ankle/torso torque envelopes are recorded in control.json but are not implemented by independent MJCF torque inputs.",
            "No locomotion/tracking policy or RL environment is trained or validated here."]
        result["status"] = "passed"
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("VALIDATION", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
