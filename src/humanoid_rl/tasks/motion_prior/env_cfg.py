"""H1 motion-prior pretraining on native mjlab physics and managers."""

from dataclasses import dataclass
import re

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from humanoid_rl.env.actions import JointRangePositionActionCfg, LimitedJointPositionActionCfg
from humanoid_rl.tasks.motion_prior import mdp as prior_mdp
from humanoid_rl.tasks.motion_prior.agents import get_agent_config


@dataclass
class MotionPriorEnvConfig:
    physics_dt: float = 0.005
    decimation: int = 4
    episode_length_s: float = 10.0
    history_length: int = 4
    lie_down_probability: float = 0.3
    terminate_on_fall: bool = False
    observation_noise: bool = False
    domain_randomization: bool = False

    def __post_init__(self):
        if self.physics_dt <= 0 or self.decimation < 1 or self.history_length < 1 or self.episode_length_s <= 0:
            raise ValueError("Invalid motion-prior timing/history settings")
        if not 0 <= self.lie_down_probability <= 1:
            raise ValueError("lie_down_probability must be in [0,1]")


def make_env_cfg(settings=None, control=None, *, agent_cfg=None):
    settings = settings or MotionPriorEnvConfig()
    agent_cfg = agent_cfg or get_agent_config()
    if agent_cfg.robot == "g1":
        from robots.humanoid.g1.bfm_zero import get_action_scale, get_robot_cfg, load_control
    elif agent_cfg.robot == "humanoid_light":
        from robots.humanoid.humanoid_light.mjlab import get_action_scale, get_robot_cfg, load_control
        if settings.observation_noise or settings.domain_randomization:
            raise ValueError("Humanoid Light currently requires observation_noise=false and domain_randomization=false")
    else:
        from robots.humanoid.h1.mjlab import get_action_scale, get_robot_cfg, load_control
        if settings.observation_noise or settings.domain_randomization:
            raise ValueError("H1 bring-up supports observation_noise=false and domain_randomization=false")
    control = load_control() if control is None else control
    action_type = LimitedJointPositionActionCfg if control.get("clip_position_targets", True) else JointPositionActionCfg
    if control.get("action_parameterization") == "signed_joint_range":
        action_type = JointRangePositionActionCfg
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(entities={"robot": get_robot_cfg(control)},
                       terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1024, env_spacing=4.0,
                       sensors=(
                           ContactSensorCfg(name="prior_feet", primary=ContactMatch(
                               mode="body", pattern="("+"|".join(map(re.escape, agent_cfg.feet_body_names))+")", entity="robot"),
                               fields=("found", "force"), reduce="netforce"),
                           ContactSensorCfg(name="prior_undesired", primary=ContactMatch(
                               mode="body", pattern="("+"|".join(agent_cfg.undesired_contact_patterns)+")", entity="robot"),
                               fields=("found", "force"), reduce="netforce"))),
        observations={"actor": ObservationGroupCfg(
            terms={"proprioception": ObservationTermCfg(func=prior_mdp.actor_state)}, concatenate_terms=True)},
        actions={"joint_pos": action_type(
            entity_name="robot", actuator_names=agent_cfg.joint_names, preserve_order=True,
            use_default_offset=True, scale=get_action_scale(control))},
        rewards={},
        terminations={"time_out": TerminationTermCfg(
            func=prior_mdp.original_time_out if agent_cfg.strict_timeout else mdp.time_out,time_out=True)},
        events={"reference_reset": EventTermCfg(func=prior_mdp.reset_reference, mode="reset")},
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=settings.physics_dt), nconmax=128, njmax=512),
        decimation=settings.decimation, episode_length_s=settings.episode_length_s,
        viewer=ViewerConfig(entity_name="robot", body_name=agent_cfg.root_body_name, distance=4.0),
        auto_reset=False, scale_rewards_by_dt=False,
    )
    if settings.terminate_on_fall:
        cfg.terminations["fallen"] = TerminationTermCfg(func=prior_mdp.fallen)
    cfg.motion_prior = settings
    cfg.robot_control = control
    cfg.motion_prior_agent = agent_cfg
    if settings.domain_randomization:
        from humanoid_rl.tasks.motion_prior.randomization import configure_randomization
        configure_randomization(cfg,agent_cfg,control)
    return cfg
