"""Compare the actual mjlab action/actuator path to independent teacher equations."""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from humanoid_rl.algorithm.add.config import config_from_values
from humanoid_rl.env.imitation.add import configure_add_environment
from humanoid_rl.tasks.carry_box.env_cfg import make_env_cfg
from robots.humanoid.g1.control import (
    BUNDLE_NAME, load_controller_bundle, load_episode_controller, validate_profile,
)
from test_object_amp import DATA


MOTION = "episode_00002/motion.npz"


def original_arrays():
    """Read the source separately; do not derive expected outputs from our profile."""
    episode = DATA/"episode_00002"
    raw = yaml.safe_load((episode/"robot_config.yaml").read_text())
    meta = json.loads((episode/"metadata.json").read_text())
    control = raw["training"]["control"]
    names = meta["joint_names"]
    kp = np.asarray([next(v for k,v in control["stiffness"].items() if k in n) for n in names])
    kd = np.asarray([next(v for k,v in control["damping"].items() if k in n) for n in names])
    coefficient = np.asarray([control["action_scale_overrides"].get(n, control["action_scale"]) for n in names])
    effort = np.asarray(control["effort_limit"])
    defaults = np.asarray([raw["training"]["init_state"]["default_joint_angles"][n] for n in names])
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(episode/"robot.xml"))
    limits = model.jnt_range[[model.joint(n).id for n in names]]
    return SimpleNamespace(names=names, kp=kp, kd=kd, coefficient=coefficient, effort=effort,
        defaults=defaults, limits=limits, normalize=control["normalize_action_to"], clip=meta["controller_args"]["action_clip"])


def original_target(raw_action, source):
    scaled = np.clip(raw_action*source.normalize, -source.clip, source.clip)
    return np.clip(source.defaults+scaled*source.coefficient*source.effort/source.kp,
                   source.limits[:, 0], source.limits[:, 1])


@unittest.skipUnless(DATA.exists(), "Recorded episode configuration required")
class ReferenceControllerUnitTests(unittest.TestCase):
    def test_configuration_forbids_accidental_double_action_scaling(self):
        base = dict(data_root=str(DATA), single_motion=MOTION, controller_profile="reference")
        config_from_values(base)
        with self.assertRaisesRegex(ValueError, "action_scale_multiplier"):
            config_from_values(base | {"action_scale_multiplier": 5})
        with self.assertRaisesRegex(ValueError, "controller_profile"):
            config_from_values(base | {"controller_profile": "unknown"})

    def test_saved_episode_profile_and_portable_integrity(self):
        source = original_arrays()
        profile = load_episode_controller(DATA, MOTION)
        self.assertEqual(profile["joint_names"], source.names)
        np.testing.assert_allclose(profile["target_scale"], source.coefficient*source.effort/source.kp, atol=1e-14)
        np.testing.assert_allclose(np.asarray(profile["target_scale"])[12:15]*5,
                                   [.3666666666666667, .2083329678522872, .5], atol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            torch.save(profile, Path(directory)/BUNDLE_NAME)
            self.assertEqual(load_controller_bundle(directory), profile)
        changed = deepcopy(profile)
        changed["damping"][12] = 0
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_profile(changed)

    def test_reference_profile_does_not_mutate_native_mjlab_defaults(self):
        from mjlab.asset_zoo.robots import get_g1_robot_cfg
        from mjlab.actuator import BuiltinPositionActuatorCfg
        cfg = make_env_cfg()
        configure_add_environment(cfg, config_from_values(dict(data_root=str(DATA), single_motion=MOTION,
                                   controller_profile="reference")))
        native = get_g1_robot_cfg()
        self.assertEqual(native.init_state.joint_pos[".*_knee_joint"], .669)
        self.assertTrue(all(isinstance(a, BuiltinPositionActuatorCfg) for a in native.articulation.actuators))
        # A restored controller/scene needs no source configuration files for play.
        with patch("humanoid_rl.env.controllers.load_episode_controller", side_effect=AssertionError("source read")):
            restored = make_env_cfg()
            configure_add_environment(restored,
                config_from_values(dict(data_root="/missing", single_motion=MOTION, controller_profile="reference")),
                fixed_scene=cfg.fixed_scene, controller=cfg.reference_controller)
            self.assertEqual(restored.reference_controller["fingerprint"], cfg.reference_controller["fingerprint"])


@unittest.skipUnless(DATA.exists() and torch.cuda.is_available(), "Controller parity requires episode data and CUDA")
class ReferenceControllerEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.envs import ManagerBasedRlEnv
        cfg = make_env_cfg()
        cfg.scene.num_envs = 8
        config = config_from_values(dict(data_root=str(DATA), single_motion=MOTION,
                                        controller_profile="reference", terminate_on_fall=False))
        configure_add_environment(cfg, config)
        cls.env = ManagerBasedRlEnv(cfg, device="cuda:0")
        cls.env.reset(seed=17)
        cls.term = cls.env.action_manager.get_term("joint_pos")
        cls.source = original_arrays()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_target_and_actual_explicit_pd_torque_match_teacher_equations(self):
        env, source, term = self.env, self.source, self.term
        robot = env.scene["robot"]
        self.assertEqual(term.target_names, source.names)
        self.assertEqual(list(robot.joint_names), source.names)
        rng = np.random.default_rng(621)
        worst_target, worst_torque = 0.0, 0.0
        for _ in range(32):
            actions = rng.uniform(-2, 2, (env.num_envs, 29)).astype(np.float32)
            q = rng.uniform(source.limits[:, 0], source.limits[:, 1], (env.num_envs, 29)).astype(np.float32)
            qd = rng.normal(0, 12, (env.num_envs, 29)).astype(np.float32)
            robot.write_joint_state_to_sim(torch.tensor(q, device=env.device), torch.tensor(qd, device=env.device))
            term.process_actions(torch.tensor(actions, device=env.device))
            term.apply_actions()
            env.scene.write_data_to_sim()
            target = original_target(actions.astype(np.float64), source)
            actual_target = robot.data.joint_pos_target.cpu().numpy()
            expected_torque = np.clip(source.kp*(target-q)-source.kd*qd, -source.effort, source.effort)
            actual_torque = env.sim.data.ctrl[:, robot.indexing.ctrl_ids].cpu().numpy()
            worst_target = max(worst_target, float(np.max(np.abs(target-actual_target))))
            worst_torque = max(worst_torque, float(np.max(np.abs(expected_torque-actual_torque))))
            np.testing.assert_allclose(actual_target, target, atol=7e-7, rtol=1e-6)
            np.testing.assert_allclose(actual_torque, expected_torque, atol=7e-5, rtol=2e-6)
        print(f"Reference controller parity: target max_abs={worst_target:.8g} rad, torque max_abs={worst_torque:.8g} Nm", flush=True)
        env.reset()

    def test_reference_gains_passive_parameters_and_default_angles_are_applied(self):
        env, source = self.env, self.source
        robot = env.scene["robot"]
        profile = env.cfg.reference_controller
        np.testing.assert_allclose(robot.data.default_joint_pos[0].cpu(), source.defaults, atol=1e-7)
        model = env.sim.mj_model
        ids = robot.indexing.joint_v_adr.cpu().numpy()
        for field, model_field in (("armature", "dof_armature"), ("frictionloss", "dof_frictionloss"), ("viscous_damping", "dof_damping")):
            np.testing.assert_allclose(getattr(model, model_field)[ids], profile[field], atol=1e-12)
        motors = robot.indexing.ctrl_ids.cpu().numpy()
        np.testing.assert_allclose(model.actuator_gainprm[motors, 0], 1)
        np.testing.assert_allclose(model.actuator_biasprm[motors], 0)
        np.testing.assert_allclose(model.actuator_forcerange[motors], np.stack((-source.effort, source.effort), -1))

    def test_pd_is_recomputed_using_current_state_at_every_physics_substep(self):
        env, source, term = self.env, self.source, self.term
        env.reset()
        robot = env.scene["robot"]
        actions = np.full((env.num_envs, 29), .05, np.float32)
        target = original_target(actions, source)
        term.process_actions(torch.tensor(actions, device=env.device))
        torques = []
        for _ in range(env.cfg.decimation):
            q, qd = robot.data.joint_pos.cpu().numpy(), robot.data.joint_vel.cpu().numpy()
            term.apply_actions()
            env.scene.write_data_to_sim()
            expected = np.clip(source.kp*(target-q)-source.kd*qd, -source.effort, source.effort)
            actual = env.sim.data.ctrl[:, robot.indexing.ctrl_ids].cpu().numpy().copy()
            np.testing.assert_allclose(actual, expected, atol=7e-5, rtol=2e-6)
            torques.append(actual)
            env.sim.step()
            env.scene.update(dt=env.physics_dt)
        self.assertGreater(np.max(np.abs(torques[0]-torques[-1])), 1e-4)
        env.sim.forward()
        env.reset()


if __name__ == "__main__":
    unittest.main()
