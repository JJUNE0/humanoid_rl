"""Light tracking on the official mjlab fork, retaining native coupled PD."""

import torch
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse

from humanoid_rl.env.actions import JointRangePositionActionCfg
from robots.humanoid.humanoid_light.config import JOINT_NAMES, ROOT_BODY_NAME, get_spec
from robots.humanoid.humanoid_light.mjlab import get_action_scale, get_robot_cfg, load_control
from supermimic.representations import assemble_anchor_window, motion_anchor_window, motion_goal_window

KEY_BODY_NAMES = (
    "pelvis_link",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_roll_link", "head_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_link",
)
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
# Heel, mid-sole, and toe in ankle_roll_link local coordinates (verified against humanoid_light.xml).
SOLE_OFFSETS_LOCAL = (
    (0.0, -0.043, -0.090),
    (0.0, -0.043, 0.000),
    (0.0, -0.043, 0.090),
)
DEFAULT_SOFT_FOOTSTEP_CFG = {
    "enabled": False,
    "impact_weight": 2.0,
    "v_free": 0.15,
    "impact_height": 0.050,
    "impact_tau": 0.012,
    "slip_weight": 0.15,
    "slip_height": 0.010,
    "slip_tau": 0.004,
    "leg_action_rate_weight": 0.10,
}


def resolve_soft_footstep_cfg(raw_cfg):
    merged = dict(DEFAULT_SOFT_FOOTSTEP_CFG)
    if raw_cfg:
        unknown = set(raw_cfg) - set(DEFAULT_SOFT_FOOTSTEP_CFG)
        if unknown:
            raise ValueError(f"Unknown soft_footstep settings: {sorted(unknown)}")
        merged.update(raw_cfg)
    return merged


def compute_soft_footstep_cost(
    foot_pos_w,
    foot_rot_mat_w,
    foot_lin_vel_w,
    foot_ang_vel_w,
    action,
    last_action,
    sole_offsets_local,
    *,
    impact_weight,
    v_free,
    impact_height,
    impact_tau,
    slip_weight,
    slip_height,
    slip_tau,
    leg_action_rate_weight,
):
    """Smooth differentiable foot touchdown impact, ground slip, and leg action-rate cost."""
    r_w = torch.einsum("...fij,sj->...fsi", foot_rot_mat_w, sole_offsets_local)
    sole_z = foot_pos_w[..., :, None, 2] + r_w[..., 2]
    sole_vel = foot_lin_vel_w[..., :, None, :] + torch.linalg.cross(
        foot_ang_vel_w[..., :, None, :], r_w, dim=-1
    )
    down_speed = (-sole_vel[..., 2] - v_free).clamp(min=0.0)
    impact_gate = torch.sigmoid((impact_height - sole_z) / impact_tau)
    impact_cost = (impact_gate * down_speed.square()).mean(dim=-1).sum(dim=-1)

    horiz_speed_sq = sole_vel[..., 0].square() + sole_vel[..., 1].square()
    slip_gate = torch.sigmoid((slip_height - sole_z) / slip_tau)
    slip_cost = (slip_gate * horiz_speed_sq).mean(dim=-1).sum(dim=-1)

    leg_rate = torch.sum((action[..., :12] - last_action[..., :12]).square(), dim=-1)
    return (
        impact_weight * impact_cost
        + slip_weight * slip_cost
        + leg_action_rate_weight * leg_rate
    )


def soft_footstep_env_cost(
    env,
    command_name,
    foot_indices,
    impact_weight,
    v_free,
    impact_height,
    impact_tau,
    slip_weight,
    slip_height,
    slip_tau,
    leg_action_rate_weight,
):
    command = env.command_manager.get_term(command_name)
    idx = list(foot_indices)
    foot_pos_w = command.robot_body_pos_w[:, idx] - env.scene.env_origins[:, None, :]
    foot_rot_mat_w = matrix_from_quat(command.robot_body_quat_w[:, idx])
    foot_lin_vel_w = command.robot_body_lin_vel_w[:, idx]
    foot_ang_vel_w = command.robot_body_ang_vel_w[:, idx]
    sole_offsets = foot_pos_w.new_tensor(SOLE_OFFSETS_LOCAL)
    return compute_soft_footstep_cost(
        foot_pos_w,
        foot_rot_mat_w,
        foot_lin_vel_w,
        foot_ang_vel_w,
        env.action_manager.action,
        env.action_manager.prev_action,
        sole_offsets,
        impact_weight=impact_weight,
        v_free=v_free,
        impact_height=impact_height,
        impact_tau=impact_tau,
        slip_weight=slip_weight,
        slip_height=slip_height,
        slip_tau=slip_tau,
        leg_action_rate_weight=leg_action_rate_weight,
    )


# 21 surface collision spheres across all 3 hardware safety tiers (verified against humanoid_light.xml):
# (tier, body_name, local_offset_xyz, radius_m, impact_rel_w, slip_rel_w, forbid_rel_w, unintended_rel_w)
WHOLE_BODY_SURFACE_SPECS = (
    # Tier 1: Strictly forbidden fragile head + camera (5.5 Nm)
    (1, "head_link",                ( 0.0000, -0.0140, -0.0680), 0.086, 4.00, 0.50, 1.0, 0.0),
    # Tier 2: Conditional support / cushion bodies (scaled inversely to rated joint torque)
    (2, "left_wrist_link",          ( 0.0008,  0.0000,  0.0418), 0.042, 2.50, 0.20, 0.0, 1.0),
    (2, "right_wrist_link",         (-0.0008,  0.0000,  0.0418), 0.042, 2.50, 0.20, 0.0, 1.0),
    (2, "left_elbow_link",          ( 0.0000, -0.0204,  0.0005), 0.036, 0.90, 0.10, 0.0, 0.6),
    (2, "right_elbow_link",         ( 0.0000, -0.0204, -0.0005), 0.036, 0.90, 0.10, 0.0, 0.6),
    (2, "left_elbow_link",          ( 0.0000, -0.0992, -0.0134), 0.033, 0.90, 0.10, 0.0, 0.6),
    (2, "right_elbow_link",         ( 0.0000, -0.0992,  0.0134), 0.033, 0.90, 0.10, 0.0, 0.6),
    (2, "left_shoulder_roll_link",  ( 0.0200, -0.0360, -0.0270), 0.062, 1.50, 0.15, 0.0, 0.8),
    (2, "right_shoulder_roll_link", (-0.0200, -0.0360, -0.0270), 0.062, 1.50, 0.15, 0.0, 0.8),
    (2, "torso_roll_link",          ( 0.0010,  0.2200,  0.0090), 0.130, 1.80, 0.15, 0.0, 1.0),
    (2, "left_knee_link",           (-0.0150, -0.0150, -0.0100), 0.050, 1.00, 0.12, 0.0, 0.8),
    (2, "right_knee_link",          (-0.0150, -0.0150,  0.0100), 0.050, 1.00, 0.12, 0.0, 0.8),
    (2, "pelvis_link",              ( 0.0000,  0.0000, -0.0020), 0.082, 0.70, 0.08, 0.0, 0.6),
    (2, "left_hip_roll_link",       ( 0.0000, -0.0400, -0.0270), 0.062, 0.60, 0.08, 0.0, 0.6),
    (2, "right_hip_roll_link",      ( 0.0000, -0.0400, -0.0270), 0.062, 0.60, 0.08, 0.0, 0.6),
    # Tier 3: Primary bipedal sole points (heel, mid, toe x left/right)
    (3, "left_ankle_roll_link",     ( 0.0000, -0.0430, -0.0900), 0.000, 0.45, 0.04, 0.0, 0.0),
    (3, "left_ankle_roll_link",     ( 0.0000, -0.0430,  0.0000), 0.000, 0.45, 0.04, 0.0, 0.0),
    (3, "left_ankle_roll_link",     ( 0.0000, -0.0430,  0.0900), 0.000, 0.45, 0.04, 0.0, 0.0),
    (3, "right_ankle_roll_link",    ( 0.0000, -0.0430, -0.0900), 0.000, 0.45, 0.04, 0.0, 0.0),
    (3, "right_ankle_roll_link",    ( 0.0000, -0.0430,  0.0000), 0.000, 0.45, 0.04, 0.0, 0.0),
    (3, "right_ankle_roll_link",    ( 0.0000, -0.0430,  0.0900), 0.000, 0.45, 0.04, 0.0, 0.0),
)

DEFAULT_WHOLE_BODY_CONTACT_CFG = {
    "enabled": False,
    "impact_scale": 1.0,
    "slip_scale": 1.0,
    "v_free": 0.18,
    "impact_height": 0.050,
    "impact_tau": 0.012,
    "slip_height": 0.010,
    "slip_tau": 0.004,
    "head_forbid_weight": 400.0,
    "head_safe_height": 0.045,
    "unintended_weight": 150.0,
    "unintended_height": 0.025,
    "ref_air_height": 0.060,
    "ref_air_tau": 0.015,
    "clear_tau": 0.008,
    "extra_action_rate_weight": 0.08,
}


def resolve_whole_body_contact_cfg(raw_cfg):
    merged = dict(DEFAULT_WHOLE_BODY_CONTACT_CFG)
    if raw_cfg:
        unknown = set(raw_cfg) - set(DEFAULT_WHOLE_BODY_CONTACT_CFG)
        if unknown:
            raise ValueError(f"Unknown whole_body_contact settings: {sorted(unknown)}")
        merged.update(raw_cfg)
    return merged


def build_whole_body_contact_tensors(body_names, wb_cfg, device="cpu"):
    body_idx = torch.tensor(
        [body_names.index(spec[1]) for spec in WHOLE_BODY_SURFACE_SPECS],
        dtype=torch.long,
        device=device,
    )
    local_offsets = torch.tensor(
        [spec[2] for spec in WHOLE_BODY_SURFACE_SPECS], dtype=torch.float32, device=device
    )
    radii = torch.tensor(
        [spec[3] for spec in WHOLE_BODY_SURFACE_SPECS], dtype=torch.float32, device=device
    )
    impact_w = torch.tensor(
        [spec[4] * wb_cfg["impact_scale"] for spec in WHOLE_BODY_SURFACE_SPECS],
        dtype=torch.float32,
        device=device,
    )
    slip_w = torch.tensor(
        [spec[5] * wb_cfg["slip_scale"] for spec in WHOLE_BODY_SURFACE_SPECS],
        dtype=torch.float32,
        device=device,
    )
    forbid_w = torch.tensor(
        [spec[6] * wb_cfg["head_forbid_weight"] for spec in WHOLE_BODY_SURFACE_SPECS],
        dtype=torch.float32,
        device=device,
    )
    unintend_w = torch.tensor(
        [spec[7] * wb_cfg["unintended_weight"] for spec in WHOLE_BODY_SURFACE_SPECS],
        dtype=torch.float32,
        device=device,
    )
    return {
        "body_idx": body_idx,
        "local_offsets": local_offsets,
        "radii": radii,
        "impact_w": impact_w,
        "slip_w": slip_w,
        "forbid_w": forbid_w,
        "unintend_w": unintend_w,
    }


def compute_whole_body_contact_cost(
    pred_pos_w,
    pred_rot_mat_w,
    pred_lin_vel_w,
    pred_ang_vel_w,
    target_pos_w,
    target_quat_w,
    action,
    last_action,
    wb_body_idx,
    wb_local_offsets,
    wb_radii,
    wb_impact_w,
    wb_slip_w,
    wb_forbid_w,
    wb_unintend_w,
    *,
    v_free,
    impact_height,
    impact_tau,
    slip_height,
    slip_tau,
    head_safe_height,
    unintended_height,
    ref_air_height,
    ref_air_tau,
    clear_tau,
    extra_action_rate_weight,
):
    """Differentiable 3-tier whole-body surface collision-sphere BPTT cost."""
    p_b = pred_pos_w.index_select(-2, wb_body_idx)
    r_mat = pred_rot_mat_w.index_select(-3, wb_body_idx)
    v_b = pred_lin_vel_w.index_select(-2, wb_body_idx)
    w_b = pred_ang_vel_w.index_select(-2, wb_body_idx)

    r_w = torch.einsum("...sij,sj->...si", r_mat, wb_local_offsets)
    d_pred = p_b[..., 2] + r_w[..., 2] - wb_radii
    v_surf = v_b + torch.linalg.cross(w_b, r_w, dim=-1)

    p_ref = target_pos_w.index_select(-2, wb_body_idx)
    r_ref_mat = matrix_from_quat(target_quat_w.index_select(-2, wb_body_idx))
    r_ref_w = torch.einsum("...sij,sj->...si", r_ref_mat, wb_local_offsets)
    d_ref = p_ref[..., 2] + r_ref_w[..., 2] - wb_radii

    down_speed = (-v_surf[..., 2] - v_free).clamp(min=0.0)
    impact_gate = torch.sigmoid((impact_height - d_pred) / impact_tau)
    impact_cost = torch.sum(wb_impact_w * impact_gate * down_speed.square(), dim=-1)

    horiz_speed_sq = v_surf[..., 0].square() + v_surf[..., 1].square()
    slip_gate = torch.sigmoid((slip_height - d_pred) / slip_tau)
    slip_cost = torch.sum(wb_slip_w * slip_gate * horiz_speed_sq, dim=-1)

    head_pen = torch.nn.functional.softplus((head_safe_height - d_pred) / clear_tau) * clear_tau
    forbid_cost = torch.sum(wb_forbid_w * head_pen.square(), dim=-1)

    ref_airborne = torch.sigmoid((d_ref - ref_air_height) / ref_air_tau)
    unintend_pen = (
        torch.nn.functional.softplus((unintended_height - d_pred) / clear_tau) * clear_tau
    )
    unintend_cost = torch.sum(wb_unintend_w * ref_airborne * unintend_pen.square(), dim=-1)

    extra_rate = extra_action_rate_weight * torch.sum((action - last_action).square(), dim=-1)
    return impact_cost + slip_cost + forbid_cost + unintend_cost + extra_rate


def whole_body_contact_env_cost(env, command_name, wb_cfg):
    command = env.command_manager.get_term(command_name)
    cache_attr = "_wb_contact_tensors"
    tensors = getattr(command, cache_attr, None)
    dev = command.robot_body_pos_w.device
    if tensors is None or tensors["body_idx"].device != dev:
        tensors = build_whole_body_contact_tensors(command.cfg.body_names, wb_cfg, device=dev)
        setattr(command, cache_attr, tensors)
    origins = env.scene.env_origins[:, None, :]
    pred_pos_w = command.robot_body_pos_w - origins
    pred_rot_mat_w = matrix_from_quat(command.robot_body_quat_w)
    target_pos_w = command.body_pos_w - origins
    return compute_whole_body_contact_cost(
        pred_pos_w,
        pred_rot_mat_w,
        command.robot_body_lin_vel_w,
        command.robot_body_ang_vel_w,
        target_pos_w,
        command.body_quat_w,
        env.action_manager.action,
        env.action_manager.prev_action,
        tensors["body_idx"],
        tensors["local_offsets"],
        tensors["radii"],
        tensors["impact_w"],
        tensors["slip_w"],
        tensors["forbid_w"],
        tensors["unintend_w"],
        v_free=wb_cfg["v_free"],
        impact_height=wb_cfg["impact_height"],
        impact_tau=wb_cfg["impact_tau"],
        slip_height=wb_cfg["slip_height"],
        slip_tau=wb_cfg["slip_tau"],
        head_safe_height=wb_cfg["head_safe_height"],
        unintended_height=wb_cfg["unintended_height"],
        ref_air_height=wb_cfg["ref_air_height"],
        ref_air_tau=wb_cfg["ref_air_tau"],
        clear_tau=wb_cfg["clear_tau"],
        extra_action_rate_weight=wb_cfg["extra_action_rate_weight"],
    )


def root_ang_vel(env):
    data = env.scene["robot"].data
    return quat_apply_inverse(data.root_link_quat_w, data.root_link_ang_vel_w)


def full_state(env):
    data = env.scene["robot"].data
    return torch.cat((data.root_link_pos_w - env.scene.env_origins, data.root_link_quat_w,
                      data.joint_pos, data.root_link_lin_vel_w, data.root_link_ang_vel_w, data.joint_vel), -1)


def goal(env):
    command = env.command_manager.get_term("motion")
    target_pos, target_quat = motion_anchor_window(command.motion, command.time_steps)
    pos, ori = assemble_anchor_window(command.robot_anchor_pos_w, command.robot_anchor_quat_w,
                                    target_pos + env.scene.env_origins[:, None], target_quat)
    return torch.cat((motion_goal_window(command.motion, command.time_steps), pos, ori), -1)


def make_cfg(motion_directory, settings, num_envs, seed, *, sampling_mode="uniform"):
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}
    cfg.scene.num_envs, cfg.seed = num_envs, seed
    cfg.scene.sensors = ()
    action_mode = settings.get("action_mode", "joint_range")
    if action_mode == "coupled_scale":
        control = dict(load_control(), action_parameterization="coupled_effort_scale",
                       action_scale=settings.get("action_scale", 0.25))
        cfg.actions = {"joint_pos": JointPositionActionCfg(
            entity_name="robot", actuator_names=JOINT_NAMES, preserve_order=True,
            scale=get_action_scale(control), use_default_offset=True)}
    elif action_mode == "joint_range":
        cfg.actions = {"joint_pos": JointRangePositionActionCfg(entity_name="robot", actuator_names=JOINT_NAMES,
                       preserve_order=True, scale={name: 1. for name in JOINT_NAMES}, use_default_offset=True)}
    else:
        raise ValueError(f"Unknown action_mode {action_mode!r}")
    command = cfg.commands["motion"]
    command.motion_file = str(motion_directory)
    command.anchor_body_name = ROOT_BODY_NAME
    if settings.get("key_bodies_only", False):
        command.body_names = KEY_BODY_NAMES
    else:
        model = get_spec().compile()
        command.body_names = tuple(model.body(i).name for i in range(1, model.nbody))
    command.pose_range, command.velocity_range = {}, {}
    command.joint_position_range = (0., 0.)
    command.sampling_mode = "external" if sampling_mode == "return" else sampling_mode
    command.debug_vis = False
    cfg.observations = {
        "actor": ObservationGroupCfg(terms={
            "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
            "base_ang_vel": ObservationTermCfg(func=root_ang_vel),
            "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
            "actions": ObservationTermCfg(func=mdp.last_action),
        }, concatenate_terms=True, enable_corruption=False),
        "full_state": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=full_state)},
                                         concatenate_terms=True, enable_corruption=False),
        "goal": ObservationGroupCfg(terms={"reference": ObservationTermCfg(func=goal)},
                                   concatenate_terms=True, enable_corruption=False),
    }
    # A fixed-zero context retains the official model's context interface without
    # randomizing the converted robot's physics or observations.
    cfg.events = {"encoder_bias": EventTermCfg(func=mdp.dr.encoder_bias, mode="startup",
                  params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (0., 0.)})}
    cfg.rewards.pop("self_collisions", None)
    cfg.rewards["joint_limit"].weight = -2.
    soft_cfg = resolve_soft_footstep_cfg(settings.get("soft_footstep"))
    wb_cfg = resolve_whole_body_contact_cfg(settings.get("whole_body_contact"))
    if soft_cfg["enabled"] and wb_cfg["enabled"]:
        raise ValueError("Cannot enable both soft_footstep and whole_body_contact simultaneously")
    if soft_cfg["enabled"]:
        foot_indices = tuple(command.body_names.index(name) for name in FOOT_BODY_NAMES)
        reward_params = {k: v for k, v in soft_cfg.items() if k != "enabled"}
        cfg.rewards["soft_footstep"] = RewardTermCfg(
            func=soft_footstep_env_cost,
            weight=-1.0,
            params={"command_name": "motion", "foot_indices": foot_indices, **reward_params},
        )
    if wb_cfg["enabled"]:
        cfg.rewards["whole_body_contact"] = RewardTermCfg(
            func=whole_body_contact_env_cost,
            weight=-1.0,
            params={"command_name": "motion", "wb_cfg": wb_cfg},
        )
    cfg.terminations.pop("ee_body_pos", None)
    cfg.sim.mujoco.timestep = settings["physics_dt"]
    cfg.sim.mujoco.iterations = settings["solver_iterations"]
    cfg.sim.mujoco.ls_iterations = settings["line_search_iterations"]
    cfg.decimation = settings["decimation"]
    cfg.sim.nconmax, cfg.sim.njmax = 128, 512
    if abs(cfg.sim.mujoco.timestep * cfg.decimation - .02) > 1e-9:
        raise ValueError("Light tracking motions require a 50Hz control period")
    cfg.episode_length_s = settings["episode_length_s"]
    cfg.viewer.body_name = ROOT_BODY_NAME
    return cfg
