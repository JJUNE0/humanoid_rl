"""Portable first-frame scene for a single-motion mjlab overfit experiment."""

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import quat_apply

from humanoid_retarget.datasets.carry_box import scene_description
from humanoid_rl.algorithm.amp.features import ExpertFeatures, OBJECT_REFERENCE_DIM


SCHEMA = "g1-carry-fixed-scene-v1"
BUNDLE_NAME = "fixed_scene.pt"


def load_episode_scene(root, relative):
    """Read source data only; the simulated robot always comes from mjlab."""
    import yaml
    from robots.humanoid.g1.config import get_spec

    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or relative.name != "motion.npz":
        raise ValueError("single_motion must be a relative episode/motion.npz path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Single-motion scene leaves dataset root")
    blobs = {}
    for name in ("motion.npz", "box_config.yaml", "robot.xml"):
        source = (path.parent / name).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Single-motion sidecar leaves dataset root")
        blobs[name] = source.read_bytes()
    scene = scene_description(blobs["box_config.yaml"], blobs["robot.xml"])
    model = get_spec().compile()
    names = tuple(model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0)
    with np.load(path, allow_pickle=False) as motion:
        if str(motion["quaternion_order"].item()) != "wxyz":
            raise ValueError("Fixed scene requires wxyz source quaternions")
        features, references = ExpertFeatures(names, True).encode(motion, scene)
        duration_s, frames = float(motion["time"][-1] - motion["time"][0]), len(references)
    xml = ET.fromstring(blobs["robot.xml"])
    half_sizes, support_friction = [], []
    for name in ("initial_box_support", "target_box_support"):
        geom = xml.find(f"./worldbody/geom[@name='{name}']")
        half_sizes.append(np.fromstring(geom.attrib["size"], sep=" "))
        if "friction" not in geom.attrib:
            raise ValueError("Fixed support requires explicit source friction")
        support_friction.append(np.fromstring(geom.attrib["friction"], sep=" "))
    box = yaml.safe_load(blobs["box_config.yaml"])
    bundle = {
        "schema": SCHEMA, "source_motion": relative.as_posix(),
        "source_sha256": {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
        "joint_names": names, "reference": references[0].clone(),
        "expert_initial_features": features[0].clone(),
        "support_half_sizes": torch.tensor(np.asarray(half_sizes), dtype=torch.float32),
        "box_friction": torch.tensor(box["friction"], dtype=torch.float32),
        "support_friction": torch.tensor(np.asarray(support_friction), dtype=torch.float32),
        "frames": frames, "duration_s": duration_s, "reference_frame": 0,
    }
    validate_scene(bundle)
    return bundle


def validate_scene(scene):
    if scene.get("schema") != SCHEMA or scene.get("reference_frame") != 0:
        raise ValueError("Unsupported fixed-scene schema/reference frame")
    shapes = {"reference": (OBJECT_REFERENCE_DIM,), "expert_initial_features": (104,),
              "support_half_sizes": (2, 3), "box_friction": (3,), "support_friction": (2, 3)}
    for name, shape in shapes.items():
        value = scene[name]
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid fixed-scene {name}")
    if not (scene["support_half_sizes"] > 0).all() or not (scene["reference"][84:88] > 0).all():
        raise ValueError("Fixed scene requires positive dimensions/mass")
    if not (scene["box_friction"] >= 0).all() or not (scene["support_friction"] >= 0).all():
        raise ValueError("Fixed scene requires nonnegative friction")


def load_scene_bundle(checkpoint):
    scene = torch.load(Path(checkpoint) / BUNDLE_NAME, map_location="cpu", weights_only=True)
    validate_scene(scene)
    return scene


def scene_metadata(scene):
    ref = scene["reference"]
    return {key: scene[key] for key in ("schema", "source_motion", "source_sha256", "frames", "duration_s", "reference_frame")} | {
        "bundle": BUNDLE_NAME, "randomization": False, "random_time_rsi": False,
        "robot_position": ref[:3].tolist(), "box_geometry_position": ref[71:74].tolist(),
        "box_size_xyz_m": ref[84:87].tolist(), "box_mass_kg": ref[87].item(),
        "support_full_sizes": (2 * scene["support_half_sizes"]).tolist(),
        "source_top": ref[88:91].tolist(), "target_top": ref[91:94].tolist(),
        "goal_geometry_position": ref[94:97].tolist(),
        "robot_dynamics": "native mjlab G1; unchanged from preceding experiments",
        "validation": "single-motion overfit; no held-out expert clip",
    }


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb", "geom_friction",
                       "body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def reset_fixed_scene(env, env_ids, scene):
    """Restore every selected world to frame zero without drawing random values."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    robot, box = env.scene["robot"], env.scene["box"]
    if tuple(robot.joint_names) != tuple(scene["joint_names"]):
        raise ValueError("Fixed scene and mjlab G1 joint order differ")
    ref = scene["reference"].to(env.device)
    n = len(env_ids)
    origins = env.scene.env_origins[env_ids]
    robot.write_root_state_to_sim(torch.cat((
        ref[:3].expand(n, -1) + origins, ref[3:10].expand(n, -1),
        quat_apply(ref[3:7], ref[10:13]).expand(n, -1),
    ), -1), env_ids=env_ids)
    robot.write_joint_state_to_sim(ref[13:42].expand(n, -1), ref[42:71].expand(n, -1), env_ids=env_ids)

    def geometry(entity, half, friction):
        gid = entity.indexing.geom_ids[0]
        env.sim.model.geom_size[env_ids, gid] = half
        env.sim.model.geom_rbound[env_ids, gid] = half.norm()
        env.sim.model.geom_aabb[env_ids, gid, 0] = 0
        env.sim.model.geom_aabb[env_ids, gid, 1] = half
        env.sim.model.geom_friction[env_ids, gid] = friction

    half, mass = ref[84:87] / 2, ref[87]
    geometry(box, half, scene["box_friction"].to(env.device))
    bid = box.indexing.body_ids[0]
    env.sim.model.body_mass[env_ids, bid] = mass
    squared = (2 * half).square()
    env.sim.model.body_inertia[env_ids, bid] = mass / 12 * torch.stack((
        squared[1] + squared[2], squared[0] + squared[2], squared[0] + squared[1],
    ))
    box.write_root_state_to_sim(torch.cat((ref[71:74].expand(n, -1) + origins,
        ref[74:84].expand(n, -1)), -1), env_ids=env_ids)
    for i, (name, top) in enumerate((("source_support", ref[88:91]), ("target_support", ref[91:94]))):
        support = env.scene[name]
        half_size = scene["support_half_sizes"][i].to(env.device)
        geometry(support, half_size, scene["support_friction"][i].to(env.device))
        center = top.expand(n, -1) + origins
        center[:, 2] -= half_size[2]
        identity = torch.zeros(n, 4, device=env.device)
        identity[:, 0] = 1
        support.write_mocap_pose_to_sim(torch.cat((center, identity), -1), env_ids=env_ids)
    if not hasattr(env, "carry_goal_w"):
        env.carry_goal_w = torch.zeros(env.num_envs, 3, device=env.device)
    env.carry_goal_w[env_ids] = ref[94:97] + origins
