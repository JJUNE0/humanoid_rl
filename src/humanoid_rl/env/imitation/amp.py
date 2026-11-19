"""AMP mode, versioned robot/object observations and coherent reference resets."""

import torch
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    yaw_quat,
)


def configure_amp_environment(cfg, reference_probability=0.0, *, amp_config=None, fixed_scene=None, controller=None):
    cfg.rewards = {}
    if amp_config is None or not amp_config.terminate_on_success:
        cfg.terminations.pop("success", None)
    if amp_config is not None and not amp_config.terminate_on_fall:
        cfg.terminations.pop("fallen", None)
    cfg.events.pop("amp_reference_pose", None)
    if amp_config is not None and amp_config.controller_profile == "reference":
        from humanoid_rl.env.controllers import configure_reference_controller
        configure_reference_controller(cfg, amp_config, controller)
    elif amp_config is not None and amp_config.action_scale_multiplier != 1:
        from robots.humanoid.g1.config import G1_ACTION_SCALE
        from humanoid_rl.env.actions import LimitedJointPositionActionCfg

        cfg.actions["joint_pos"] = LimitedJointPositionActionCfg(
            entity_name="robot", actuator_names=(".*",), use_default_offset=True,
            scale={name: value * amp_config.action_scale_multiplier for name, value in G1_ACTION_SCALE.items()},
        )
    if amp_config is not None and amp_config.control is not None:
        from mjlab.sensor import ContactMatch, ContactSensorCfg

        existing = {sensor.name for sensor in cfg.scene.sensors}
        cfg.scene.sensors = tuple(cfg.scene.sensors) + tuple(
            ContactSensorCfg(
                name=name, primary=ContactMatch(mode="body", pattern=pattern, entity="robot"),
                fields=("found", "force"), reduce="netforce",
            )
            for name, pattern in (
                ("control_feet", "(left|right)_ankle_roll_link"),
                ("control_undesired", ".*(pelvis|shoulder|hip).*"),
            ) if name not in existing
        )
    if amp_config is not None and amp_config.fixed_initial_state:
        from humanoid_rl.tasks.carry_box.fixed_scene import load_episode_scene, reset_fixed_scene, validate_scene

        if reference_probability:
            raise ValueError("Fixed scene cannot use random-time reference initialization")
        fixed_scene = fixed_scene if fixed_scene is not None else load_episode_scene(
            amp_config.data_root, amp_config.single_motion
        )
        validate_scene(fixed_scene)
        if fixed_scene["source_motion"] != amp_config.single_motion:
            raise ValueError("Fixed-scene bundle and selected AMP motion differ")
        cfg.fixed_scene = fixed_scene
        # Replace size sampling AND the ordinary random robot/scene reset.
        cfg.events = {"fixed_scene": EventTermCfg(
            func=reset_fixed_scene, mode="reset", params={"scene": fixed_scene},
        )}
    elif reference_probability:
        cfg.events["amp_reference_pose"] = EventTermCfg(
            func=reset_reference_scene if amp_config is not None and amp_config.observe_object else reset_reference_pose,
            mode="reset",
            params={"probability": reference_probability},
        )


def reset_reference_pose(env, env_ids, probability):
    """Robot-only RSI. Box/supports retain the randomized scene; no grasp is forced."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    selected = env_ids[torch.rand(len(env_ids), device=env.device) < probability]
    if not len(selected):
        return
    samples = env.amp_dataset.train.sample_references(len(selected), env.device)
    robot = env.scene["robot"]
    source_quat = samples[:, 3:7]
    source_heading = yaw_quat(source_quat)
    yaw = torch.empty(len(selected), device=env.device).uniform_(
        *env.cfg.carry.robot_yaw
    )
    target_heading = quat_from_euler_xyz(
        torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
    )
    quat = quat_mul(
        target_heading, quat_mul(quat_conjugate(source_heading), source_quat)
    )
    position = (
        robot.data.default_root_state[selected, :3].clone()
        + env.scene.env_origins[selected]
    )
    position[:, 2] = samples[:, 2] + env.scene.env_origins[selected, 2]
    linear = quat_apply(
        target_heading, quat_apply_inverse(source_heading, samples[:, 7:10])
    )
    angular_w = quat_apply(source_quat, samples[:, 10:13])
    angular = quat_apply(target_heading, quat_apply_inverse(source_heading, angular_w))
    robot.write_root_state_to_sim(
        torch.cat((position, quat, linear, angular), -1), env_ids=selected
    )
    robot.write_joint_state_to_sim(
        samples[:, 13:42], samples[:, 42:71], env_ids=selected
    )


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb", "body_mass", "body_inertia",
                       recompute=RecomputeLevel.set_const)
def reset_reference_scene(env, env_ids, probability):
    """Restore a matched robot/box/support/goal sample, applying one XY/yaw transform.

    The ordinary reset still samples the configured scene ranges. RSI instead
    restores expert dimensions/mass and support top heights, since changing any
    of those independently would invalidate an expert grasp or resting box.
    Native support widths remain randomized; robot-support collisions stay off.
    """
    from humanoid_rl.algorithm.amp.features import OBJECT_REFERENCE_DIM

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    selected = env_ids[torch.rand(len(env_ids), device=env.device) < probability]
    if not len(selected):
        return
    sample = env.amp_dataset.train.sample_references(len(selected), env.device)
    if sample.shape[-1] != OBJECT_REFERENCE_DIM:
        raise ValueError("Object AMP reset requires matched robot/box/scene references")
    robot, box = env.scene["robot"], env.scene["box"]
    root_pos, root_quat = sample[:, :3], sample[:, 3:7]
    yaw = torch.empty(len(selected), device=env.device).uniform_(*env.cfg.carry.robot_yaw)
    heading = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    rotation = quat_mul(heading, quat_conjugate(yaw_quat(root_quat)))
    destination = robot.data.default_root_state[selected, :3].clone() + env.scene.env_origins[selected]
    destination[:, 2] = root_pos[:, 2] + env.scene.env_origins[selected, 2]

    def transform(position):
        return destination + quat_apply(rotation, position - root_pos)

    robot.write_root_state_to_sim(torch.cat((
        destination, quat_mul(rotation, root_quat),
        quat_apply(rotation, sample[:, 7:10]),
        quat_apply(rotation, quat_apply(root_quat, sample[:, 10:13])),
    ), -1), env_ids=selected)
    robot.write_joint_state_to_sim(sample[:, 13:42], sample[:, 42:71], env_ids=selected)
    half, mass = sample[:, 84:87] / 2, sample[:, 87]
    gid, bid = box.indexing.geom_ids[0], box.indexing.body_ids[0]
    # Native box is a centred primitive. Refresh the collision broad phase as
    # well as dimensions; EventManager recomputes physical constants afterwards.
    env.sim.model.geom_size[selected, gid] = half
    env.sim.model.geom_rbound[selected, gid] = half.norm(dim=-1)
    env.sim.model.geom_aabb[selected, gid, 0] = 0
    env.sim.model.geom_aabb[selected, gid, 1] = half
    env.sim.model.body_mass[selected, bid] = mass
    squared = (2 * half).square()
    env.sim.model.body_inertia[selected, bid] = mass[:, None] / 12 * torch.stack((
        squared[:, 1] + squared[:, 2], squared[:, 0] + squared[:, 2], squared[:, 0] + squared[:, 1],
    ), -1)
    box.write_root_state_to_sim(torch.cat((
        transform(sample[:, 71:74]), quat_mul(rotation, sample[:, 74:78]),
        quat_apply(rotation, sample[:, 78:81]), quat_apply(rotation, sample[:, 81:84]),
    ), -1), env_ids=selected)
    for name, top in (("source_support", sample[:, 88:91]), ("target_support", sample[:, 91:94])):
        support = env.scene[name]
        center = transform(top)
        center[:, 2] -= env.sim.model.geom_size[selected, support.indexing.geom_ids[0], 2]
        support.write_mocap_pose_to_sim(torch.cat((center, rotation), -1), env_ids=selected)
    env.carry_goal_w[selected] = transform(sample[:, 94:97])
