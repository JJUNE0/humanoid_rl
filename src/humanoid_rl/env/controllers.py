"""Apply portable robot control profiles to mjlab environment configs."""

from dataclasses import replace
from functools import partial

from robots.humanoid.g1.control import load_episode_controller, reference_robot_spec, validate_profile
from humanoid_rl.env.actions import ReferenceJointPositionActionCfg


def configure_reference_controller(cfg, imitation_config, profile=None):
    from mjlab.actuator import IdealPdActuatorCfg

    profile = profile if profile is not None else load_episode_controller(
        imitation_config.data_root, imitation_config.single_motion)
    validate_profile(profile)
    if profile["source_motion"] != imitation_config.single_motion:
        raise ValueError("Reference controller and selected motion differ")
    if cfg.sim.mujoco.timestep*cfg.decimation != profile["control_dt"]:
        raise ValueError("Reference controller requires its recorded 50 Hz control period")
    if cfg.sim.mujoco.timestep != profile.get("physics_dt", .005):
        raise ValueError("Reference controller requires its recorded 200 Hz PD/physics period")
    names = profile["joint_names"]
    robot = cfg.scene.entities["robot"]
    # replace() avoids mutating mjlab's shared native G1 configuration constants.
    robot.init_state = replace(robot.init_state, joint_pos=dict(zip(names, profile["default_joint_pos"])))
    robot.spec_fn = partial(reference_robot_spec, profile=profile)
    robot.articulation = replace(robot.articulation, actuators=tuple(
        IdealPdActuatorCfg(target_names_expr=(name,), stiffness=profile["stiffness"][i],
            damping=profile["damping"][i], effort_limit=profile["effort_limit"][i],
            armature=profile["armature"][i], frictionloss=profile["frictionloss"][i],
            viscous_damping=profile["viscous_damping"][i]) for i, name in enumerate(names)))
    cfg.actions["joint_pos"] = ReferenceJointPositionActionCfg(
        entity_name="robot", actuator_names=tuple(names),
        use_default_offset=True, scale=dict(zip(names, profile["target_scale"])),
        normalize_action_to=profile["normalize_action_to"], action_clip=profile["action_clip"])
    cfg.reference_controller = profile
