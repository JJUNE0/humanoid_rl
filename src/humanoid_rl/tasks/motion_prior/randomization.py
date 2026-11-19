"""Native mjlab counterparts of original BFM-Zero's enabled randomizations."""

from mjlab.envs import mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
import torch


@requires_model_fields("body_mass","body_inertia",recompute=RecomputeLevel.set_const)
def randomize_mass_and_inertia(env,env_ids,ranges):
    """Original Isaac mass randomization also scales the inertia by mass ratio."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs,device=env.device)
    bodies = env.scene["robot"].data.indexing.body_ids
    scale = torch.empty(len(env_ids),len(bodies),device=env.device).uniform_(*ranges)
    row, column = env_ids[:,None], bodies[None,:]
    env.sim.model.body_mass[row,column] = env.sim.get_default_field("body_mass")[bodies][None]*scale
    env.sim.model.body_inertia[row,column] = env.sim.get_default_field("body_inertia")[bodies][None]*scale[...,None]


def configure_randomization(cfg, profile, control):
    values = profile.randomization
    robot = SceneEntityCfg("robot")
    cfg.events.update({
        "bfm_mass": EventTermCfg(mode="startup",func=randomize_mass_and_inertia,params={
            "ranges":tuple(values["mass_scale"])}),
        "bfm_com": EventTermCfg(mode="startup",func=dr.body_com_offset,params={
            "asset_cfg":SceneEntityCfg("robot",body_names=("torso_link",)),
            "ranges":{i:tuple(values["com_offset"]) for i in range(3)},"operation":"add"}),
        "bfm_friction": EventTermCfg(mode="startup",func=dr.geom_friction,params={
            "asset_cfg":robot,"ranges":tuple(values["friction"]),"operation":"abs"}),
        "bfm_joint_offset": EventTermCfg(mode="reset",func=dr.encoder_bias,params={
            "asset_cfg":robot,"bias_range":tuple(values["joint_offset"])}),
        "bfm_push": EventTermCfg(mode="interval",func=mdp.push_by_setting_velocity,
            interval_range_s=tuple(values["push_interval_s"]),params={
                "asset_cfg":robot,"velocity_range":{key:(-values["push_velocity"],values["push_velocity"])
                                                       for key in ("x","y","roll","pitch","yaw")}}),
    })
