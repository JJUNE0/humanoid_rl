"""Joint-target actions and motor-coordinate coupled PD on native mjlab."""

from dataclasses import dataclass
from functools import partial
import json

import torch
from mjlab.actuator import IdealPdActuator, IdealPdActuatorCfg

from robots.humanoid.humanoid_light.config import ASSET_DIR, JOINT_NAMES, get_spec


def coupling_matrix(pair, g1, g2):
    """q_motor = J @ [pitch, roll], copied from the source gear actuator."""
    if pair == "left_ankle":
        return ((-g1, g1), (g2, g2))
    if pair == "right_ankle":
        return ((g1, g1), (-g2, g2))
    if pair == "torso":
        return ((-g1, g1), (-g2, -g2))
    raise ValueError(f"Unknown coupled pair {pair!r}")


def coupled_pd_torque(matrix, kp, kd, target, position, velocity, limit, gamma=1., velocity_target=None):
    """No motor clipping: source limits joint effort after J.T mapping."""
    error = (target-position) @ matrix.T
    rate = -velocity if velocity_target is None else velocity_target-velocity
    motor_torque = kp*error + kd*(rate @ matrix.T)
    return (gamma*(motor_torque @ matrix)).clamp(-limit, limit)


class CoupledPdActuator(IdealPdActuator):
    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.matrix = torch.tensor(coupling_matrix(self.cfg.pair, self.cfg.gear_ratio_1,
                                                  self.cfg.gear_ratio_2),
                                   device=self.stiffness.device, dtype=self.stiffness.dtype)
        expected = [f"{self.cfg.pair}_pitch_joint", f"{self.cfg.pair}_roll_joint"]
        if self.target_names != expected:
            raise ValueError(f"Coupled actuator order {self.target_names} != {expected}")

    def compute(self, cmd):
        torque = coupled_pd_torque(self.matrix, self.stiffness, self.damping,
                                   cmd.position_target, cmd.pos, cmd.vel,
                                   self.force_limit, self.cfg.gamma, cmd.velocity_target)
        # No feedforward effort is issued by this task; retain joint-space units.
        return (torque+cmd.effort_target).clamp(-self.force_limit, self.force_limit)


@dataclass(kw_only=True)
class CoupledPdActuatorCfg(IdealPdActuatorCfg):
    pair: str
    gear_ratio_1: float
    gear_ratio_2: float
    gamma: float = 1.

    def build(self, entity, target_ids, target_names):
        return CoupledPdActuator(self, entity, target_ids, target_names)


def load_control():
    source = json.loads((ASSET_DIR/"control.json").read_text())
    model = get_spec().compile()
    if tuple(model.joint(i).name for i in range(1, model.njnt)) != JOINT_NAMES:
        raise ValueError("Humanoid Light joint order differs from the controller")
    result = {key: {name: source["joints"][name][key] for name in JOINT_NAMES}
              for key in ("stiffness", "damping", "effort_limit", "velocity_limit", "armature")}
    result.update(default_joint_pos={name: 0. for name in JOINT_NAMES},
                  joint_limits={name: model.joint(name).range.tolist() for name in JOINT_NAMES},
                  groups=source["groups"], root_height=.71, action_scale=1.,
                  normalize_action_to=1., clip_position_targets=True,
                  action_parameterization="signed_joint_range",
                  controller="motor-coordinate PD, joint-position target adapter",
                  velocity_limit_semantics="reference QC/diagnostics; no hard qvel projection in MuJoCo")
    return result


def get_robot_spec(control):
    spec = get_spec()
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for name in JOINT_NAMES:
        joint = spec.joint(name)
        joint.armature = control["armature"][name]
        joint.damping[:] = 0.
        joint.actfrcrange[:] = (-control["effort_limit"][name], control["effort_limit"][name])
    # Preserve non-neighbor self contacts; exempt only the recorded assemblies.
    existing = {frozenset((item.bodyname1, item.bodyname2)) for item in spec.excludes}
    for pair in json.loads((ASSET_DIR/"collision_exclusions.json").read_text()):
        if frozenset((pair["first"], pair["second"])) not in existing:
            spec.add_exclude(bodyname1=pair["first"], bodyname2=pair["second"])
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype = geom.conaffinity = 1
            geom.priority = 1
    return spec


def get_action_scale(control):
    if control.get("action_parameterization") == "signed_joint_range":
        return 1.  # JointRangePositionAction uses the physical asymmetric limits.
    # Couple-aware effective stiffness keeps a unit action near .25 rated torque.
    effective = dict(control["stiffness"])
    for pair, group_name in (("left_ankle", "ankles"), ("right_ankle", "ankles"),
                             ("torso", "torso_pitch_roll")):
        group = control["groups"][group_name]
        matrix = coupling_matrix(pair, group["gear_ratio_1"], group["gear_ratio_2"])
        for column, suffix in enumerate(("pitch", "roll")):
            effective[f"{pair}_{suffix}_joint"] *= group["gamma"]*sum(row[column]**2 for row in matrix)
    return {name: control["action_scale"]*control["effort_limit"][name]/effective[name]
            for name in JOINT_NAMES}


def get_robot_cfg(control=None):
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    control = load_control() if control is None else control
    coupled_names, actuators = set(), []
    for pair, group_name in (("left_ankle", "ankles"), ("right_ankle", "ankles"),
                             ("torso", "torso_pitch_roll")):
        names = tuple(f"{pair}_{suffix}_joint" for suffix in ("pitch", "roll"))
        coupled_names.update(names)
        group = control["groups"][group_name]
        if any(control[key][names[0]] != control[key][names[1]]
               for key in ("stiffness", "damping", "effort_limit")):
            raise ValueError("This coupled PD implementation requires equal pair gains/limits")
        actuators.append(CoupledPdActuatorCfg(
            target_names_expr=names, pair=pair, gear_ratio_1=group["gear_ratio_1"],
            gear_ratio_2=group["gear_ratio_2"], gamma=group["gamma"],
            stiffness=control["stiffness"][names[0]], damping=control["damping"][names[0]],
            effort_limit=control["effort_limit"][names[0]]))
    actuators.extend(IdealPdActuatorCfg(target_names_expr=(name,), stiffness=control["stiffness"][name],
                                      damping=control["damping"][name], effort_limit=control["effort_limit"][name])
                     for name in JOINT_NAMES if name not in coupled_names)
    return EntityCfg(spec_fn=partial(get_robot_spec, control),
                     init_state=EntityCfg.InitialStateCfg(pos=(0., 0., control["root_height"]),
                                                         joint_pos=control["default_joint_pos"],
                                                         joint_vel={".*": 0.}),
                     articulation=EntityArticulationInfoCfg(actuators=tuple(actuators)))
