"""Adapt the copied H1 MJCF to a mjlab articulated entity with explicit PD."""

import json

from robots.humanoid.h1.config import ASSET_DIR, JOINT_NAMES, get_spec


def load_control():
    return json.loads((ASSET_DIR/"control.json").read_text())


def get_robot_spec():
    spec = get_spec()
    # A mjlab task supplies terrain/lights. Preserve all robot geometry/dynamics.
    for geom in list(spec.geoms):
        if geom.name == "floor":
            spec.delete(geom)
    for light in list(spec.lights):
        spec.delete(light)
    # mjlab creates one explicitly clipped PD torque motor for each joint.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


def get_effort_limits():
    source = get_spec().compile()
    return {source.joint(int(source.actuator_trnid[i, 0])).name:
            float(max(abs(source.actuator_ctrlrange[i]))) for i in range(source.nu)}


def get_action_scale(control):
    scale = float(control["action_scale"])*float(control["normalize_action_to"])
    if not control.get("action_rescale", False):
        return scale  # Existing checkpoints used a uniform angle scale.
    effort = get_effort_limits()
    return {name: scale*effort[name]/float(control["stiffness"][name]) for name in JOINT_NAMES}


def get_robot_cfg(control=None):
    from mjlab.actuator import IdealPdActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    control = load_control() if control is None else control
    limits = get_effort_limits()
    return EntityCfg(
        spec_fn=get_robot_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.045), joint_pos=control["default_joint_pos"], joint_vel={".*": 0.0}),
        articulation=EntityArticulationInfoCfg(actuators=tuple(
            IdealPdActuatorCfg(target_names_expr=(name,),
                               stiffness=float(control["stiffness"][name]),
                               damping=float(control["damping"][name]), effort_limit=limits[name])
            for name in JOINT_NAMES)),
    )
