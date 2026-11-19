"""Reproduce a recorded carry controller using native mjlab G1 and explicit PD.

Only portable configuration/model arrays are read from the teacher episode.
No I-BFM environment, policy, controller module or MuJoCo model runs in training.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


SCHEMA = "g1-reference-controller-v1"
BUNDLE_NAME = "controller.pt"
ARRAY_FIELDS = ("default_joint_pos", "stiffness", "damping", "effort_limit",
                "action_coefficient", "target_scale", "armature", "frictionloss", "viscous_damping")


def fingerprint(profile):
    payload = {key: value for key, value in profile.items() if key != "fingerprint"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_profile(profile):
    if profile.get("schema") != SCHEMA:
        raise ValueError("Unsupported reference controller schema")
    names = profile["joint_names"]
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("Reference controller needs 29 unique joint names")
    for key in ARRAY_FIELDS:
        array = np.asarray(profile[key], dtype=np.float64)
        if array.shape != (29,) or not np.isfinite(array).all():
            raise ValueError(f"Invalid reference controller {key}")
    for key in ("stiffness", "effort_limit", "action_coefficient", "target_scale"):
        if not (np.asarray(profile[key]) > 0).all():
            raise ValueError(f"Reference controller {key} must be positive")
    for key in ("damping", "armature", "frictionloss", "viscous_damping"):
        if not (np.asarray(profile[key]) >= 0).all():
            raise ValueError(f"Reference controller {key} must be nonnegative")
    limits = np.asarray(profile["joint_limits"])
    if limits.shape != (29, 2) or not np.isfinite(limits).all() or not (limits[:, 0] < limits[:, 1]).all():
        raise ValueError("Invalid reference controller joint limits")
    expected = np.asarray(profile["action_coefficient"])*profile["effort_limit"]/np.asarray(profile["stiffness"])
    if not np.allclose(expected, profile["target_scale"], rtol=1e-12, atol=1e-12):
        raise ValueError("Reference controller action scale is inconsistent")
    if not np.isfinite(profile["normalize_action_to"]) or profile["normalize_action_to"] <= 0:
        raise ValueError("Invalid reference action normalization")
    if not np.isfinite(profile["action_clip"]) or profile["action_clip"] <= 0:
        raise ValueError("Invalid reference action clipping")
    if profile.get("fingerprint") != fingerprint(profile):
        raise ValueError("Reference controller fingerprint differs")


def load_episode_controller(root, relative):
    import mujoco
    import yaml

    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or relative.name != "motion.npz":
        raise ValueError("Reference controller requires a relative episode/motion.npz")
    episode = (root/relative).resolve().parent
    blobs = {}
    for name in ("robot_config.yaml", "robot.xml", "physics.npz", "metadata.json"):
        path = (episode/name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Reference controller sidecar leaves the dataset root")
        blobs[name] = path.read_bytes()
    robot = yaml.safe_load(blobs["robot_config.yaml"])
    metadata = json.loads(blobs["metadata.json"])
    names = metadata["joint_names"]
    control = robot["training"]["control"]
    initial = robot["training"]["init_state"]["default_joint_angles"]
    # Compile only to resolve named joint ranges/addresses and validate the saved motors.
    source_model = mujoco.MjModel.from_xml_path(str(episode/"robot.xml"))
    joint_ids = [source_model.joint(name).id for name in names]
    velocity_ids = source_model.jnt_dofadr[joint_ids]
    if velocity_ids.tolist() != metadata["qvel_addresses"]:
        raise ValueError("Recorded physics and joint velocity addresses differ")
    if (source_model.nu != 29 or source_model.actuator_forcelimited.any()
            or source_model.actuator_ctrllimited.any() or source_model.jnt_actfrclimited.any()
            or not np.allclose(source_model.actuator_gear[:, 0], 1)
            or not np.allclose(source_model.actuator_gear[:, 1:], 0)):
        raise ValueError("Reference controller expects unbounded unit-gear torque motors")

    def gains(mapping):
        values = []
        for name in names:
            matches = [value for key, value in mapping.items() if key in name]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous/missing reference gain for {name}")
            values.append(float(matches[0]))
        return values

    kp, kd = gains(control["stiffness"]), gains(control["damping"])
    coefficient = [float(control.get("action_scale_overrides", {}).get(name, control["action_scale"])) for name in names]
    effort = list(map(float, control["effort_limit"]))
    with np.load(episode/"physics.npz", allow_pickle=False) as physics:
        if not np.allclose(physics["actuator_gainprm"][:, 0], 1) or not np.allclose(physics["actuator_biasprm"], 0):
            raise ValueError("Saved reference actuators are not pure torque motors")
        passive = {key: physics[source][velocity_ids].tolist() for key, source in (
            ("armature", "dof_armature"), ("frictionloss", "dof_frictionloss"), ("viscous_damping", "dof_damping"))}
    profile = {
        "schema": SCHEMA, "source_motion": relative.as_posix(), "joint_names": names,
        "source_sha256": {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
        "recorded_controller_sha256": metadata["controller_sha256"],
        "default_joint_pos": [float(initial[name]) for name in names],
        "stiffness": kp, "damping": kd, "effort_limit": effort,
        "action_coefficient": coefficient,
        "target_scale": (np.asarray(coefficient)*effort/np.asarray(kp)).tolist(),
        "joint_limits": source_model.jnt_range[joint_ids].tolist(),
        "normalize_action_to": float(control["normalize_action_to"]),
        "action_clip": float(metadata["controller_args"]["action_clip"]),
        "control_dt": float(metadata["dt"]),
        # The recorded controller imports SIM_DT=0.005 and runs four PD substeps.
        "physics_dt": 0.005,
        "actuator": "mjlab IdealPdActuator; explicit torque recomputed every physics substep",
        "torque_limit_semantics": "clip PD torque to effort_limit; recorded controller does not multiply effort_limit_scale",
        **passive,
    }
    profile["fingerprint"] = fingerprint(profile)
    validate_profile(profile)
    return profile


def load_controller_bundle(checkpoint):
    result = torch.load(Path(checkpoint)/BUNDLE_NAME, map_location="cpu", weights_only=True)
    validate_profile(result)
    return result


def reference_robot_spec(profile):
    from robots.humanoid.g1.config import get_spec
    spec = get_spec()
    for name, limits in zip(profile["joint_names"], profile["joint_limits"]):
        spec.joint(name).range[:] = limits
    return spec
