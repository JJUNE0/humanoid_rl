"""Original BFM-Zero G1 asset and hard-waist controller on native mjlab."""

from functools import partial
import json
from pathlib import Path

from robots.humanoid.g1.config import JOINT_NAMES

ASSET_DIR = Path(__file__).with_name("bfm_zero_assets")


def load_control():
    return json.loads((ASSET_DIR/"control.json").read_text())


def get_spec():
    import mujoco
    return mujoco.MjSpec.from_file(str(ASSET_DIR/"g1_29dof.xml"))


def get_robot_spec(control):
    spec = get_spec()
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for name in JOINT_NAMES:
        joint = spec.joint(name)
        joint.range[:] = control["joint_limits"][name]
        joint.armature = control["armature"][name]
        joint.frictionloss = control["frictionloss"][name]
        joint.damping[:] = 0.0  # Explicit PD supplies the original damping torque.
        joint.actfrcrange[:] = (-control["effort_limit"][name],control["effort_limit"][name])
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.priority = 1  # Select robot friction over the unit-friction floor.
    return spec


def get_action_scale(control):
    return {name: control["action_scale"]*control["normalize_action_to"]
                  *control["effort_limit"][name]/control["stiffness"][name] for name in JOINT_NAMES}


def get_robot_cfg(control=None):
    from mjlab.actuator import IdealPdActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    control = load_control() if control is None else control
    return EntityCfg(
        spec_fn=partial(get_robot_spec,control),
        init_state=EntityCfg.InitialStateCfg(pos=(0.,0.,control["root_height"]),
                                            joint_pos=control["default_joint_pos"], joint_vel={".*":0.}),
        articulation=EntityArticulationInfoCfg(actuators=tuple(
            IdealPdActuatorCfg(target_names_expr=(name,),stiffness=control["stiffness"][name],
                               damping=control["damping"][name],effort_limit=control["effort_limit"][name])
            for name in JOINT_NAMES)),
    )
