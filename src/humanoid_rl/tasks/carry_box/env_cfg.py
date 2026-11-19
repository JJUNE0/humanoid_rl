"""Randomized G1 box carry task built entirely with mjlab managers."""

from dataclasses import dataclass, field
from functools import partial

from robots.humanoid.g1.config import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from humanoid_rl.tasks.carry_box.assets import box_spec, support_spec

from humanoid_rl.tasks.carry_box.mdp import observations, resets, rewards, terminations


@dataclass
class CarrySceneCfg:
    # Full dimensions in metres, mass in kg, angles in radians; sampled per reset.
    box_size: tuple[float, float] = (0.2, 0.4)
    box_mass: tuple[float, float] = (0.5, 0.9)
    box_yaw: tuple[float, float] = (-0.1, 0.1)
    box_xy_offset: float = 0.025
    support_width: tuple[float, float] = (0.5, 0.6)
    support_top_height: tuple[float, float] = (0.1, 0.55)
    source_x: tuple[float, float] = (0.8, 1.2)
    source_y: tuple[float, float] = (-0.25, 0.25)
    target_dx: tuple[float, float] = (0.9, 1.6)
    target_dy: tuple[float, float] = (-0.6, 0.6)
    robot_yaw: tuple[float, float] = (-0.15, 0.15)

    def validate(self):
        import math

        for name, value in vars(self).items():
            if isinstance(value, tuple):
                if (
                    len(value) != 2
                    or not all(math.isfinite(x) for x in value)
                    or value[0] > value[1]
                ):
                    raise ValueError(f"Invalid carry range {name}: {value}")
        for name in ("box_size", "box_mass", "support_width", "support_top_height"):
            if getattr(self, name)[0] <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.box_xy_offset < self.support_width[0] / 2:
            raise ValueError("box_xy_offset must fit on a support")
        # Conservative bound for all yaw angles: a square's diagonal.
        if self.support_width[0] < self.box_size[1] * 2**0.5 + 0.01:
            # Small yaw is less restrictive; use a conservative interval bound.
            max_yaw = max(abs(x) for x in self.box_yaw)
            extent = self.box_size[1] * (1 + min(max_yaw, 1.0)) + 0.01
            if self.support_width[0] < extent:
                raise ValueError(
                    "Support minimum width cannot contain the largest yawed box"
                )
        if self.target_dx[0] <= self.support_width[1] + 0.05:
            raise ValueError("target_dx must keep the two supports separated")
        if self.source_x[0] <= self.support_width[1] / 2 + 0.3:
            raise ValueError("source_x must keep the support clear of the robot spawn")


@dataclass(kw_only=True)
class CarryBoxEnvCfg(ManagerBasedRlEnvCfg):
    carry: CarrySceneCfg = field(default_factory=CarrySceneCfg)
    observe_hand_feet: bool = True
    stage_model: object | None = None
    fixed_scene: dict | None = None
    add_tracking: bool = False
    add_reference: object | None = None
    reference_controller: dict | None = None


def make_env_cfg(carry: CarrySceneCfg | None = None, *, observe_hand_feet: bool = True, stage_model=None) -> CarryBoxEnvCfg:
    carry = carry or CarrySceneCfg()
    carry.validate()
    actor = {
        "proprioception": ObservationTermCfg(func=observations.proprioception),
        "box_pose": ObservationTermCfg(func=observations.box_relative_pose),
        "goal_position": ObservationTermCfg(func=observations.goal_relative_position),
        "previous_action": ObservationTermCfg(func=mdp.last_action),
    }
    if observe_hand_feet:
        # Append to preserve the original 108D prefix and its group order.
        actor["hand_foot_position"] = ObservationTermCfg(func=observations.hand_foot_relative_positions)
    if stage_model is not None:
        from humanoid_rl.tasks.carry_box.stage.observations import frozen_stage_context
        actor["stage_context"] = ObservationTermCfg(func=frozen_stage_context)
    events = {}
    for name in ("box", "source_support", "target_support"):
        lo, hi = carry.box_size if name == "box" else carry.support_width
        events[f"{name}_size"] = EventTermCfg(
            func=dr.geom_size,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(name, geom_names=(".*",)),
                "ranges": (lo / 2, hi / 2),
                "operation": "abs",
                "axes": [0, 1, 2] if name == "box" else [0, 1],
            },
        )
    events["reset_scene"] = EventTermCfg(func=resets.reset_carry_scene, mode="reset")
    sensors = (
        ContactSensorCfg(
            name="box_target_contact",
            primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
            secondary=ContactMatch(
                mode="geom", pattern="support_geom", entity="target_support"
            ),
            fields=("found",),
        ),
        ContactSensorCfg(
            name="hand_box_contact",
            primary=ContactMatch(
                mode="geom", pattern=".*hand_collision", entity="robot"
            ),
            secondary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
            fields=("found",),
        ),
    )
    return CarryBoxEnvCfg(
        carry=carry,
        observe_hand_feet=observe_hand_feet,
        stage_model=stage_model,
        scene=SceneCfg(
            num_envs=1,
            env_spacing=6,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": get_g1_robot_cfg(),
                "box": EntityCfg(spec_fn=box_spec),
                "source_support": EntityCfg(spec_fn=support_spec),
                "target_support": EntityCfg(
                    spec_fn=partial(support_spec, rgba=(0.16, 0.82, 0.34, 1))
                ),
            },
            sensors=sensors,
        ),
        observations={"actor": ObservationGroupCfg(actor, enable_corruption=False)},
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=G1_ACTION_SCALE,
                use_default_offset=True,
            )
        },
        events=events,
        rewards={"placement": RewardTermCfg(func=rewards.placement_bonus, weight=1)},
        terminations={
            "success": TerminationTermCfg(func=terminations.placed),
            "fallen": TerminationTermCfg(func=terminations.fallen),
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        },
        sim=SimulationCfg(
            nconmax=100,
            njmax=500,
            contact_sensor_maxmatch=200,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        viewer=ViewerConfig(entity_name="robot", body_name="torso_link", distance=4.5),
        decimation=4,
        episode_length_s=30,
        auto_reset=False,
        scale_rewards_by_dt=False,
    )
