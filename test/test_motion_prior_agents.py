"""Robot settings, saved-run compatibility and parity with original BFM costs."""

import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace as NS
import unittest

import mujoco
import torch
from omegaconf import OmegaConf

from humanoid_rl.algorithm.bfm_zero.config import resolve_agent_config
from humanoid_rl.algorithm.bfm_zero.upstream import BFM_ROOT, enable_bfm_zero
from humanoid_rl.tasks.motion_prior.agents import get_agent_config
from humanoid_rl.tasks.motion_prior.mdp import AuxiliaryRewards


def original_reward_functions():
    """Execute the original function bodies verbatim, avoiding Isaac-only imports."""
    enable_bfm_zero()
    from humanoidverse.utils.torch_utils import quat_rotate_inverse
    namespace = {"torch": torch, "quat_rotate_inverse": quat_rotate_inverse}
    for relative in ("envs/legged_base_task/legged_robot_base.py", "envs/legged_robot_motions/legged_robot_motions.py"):
        path = BFM_ROOT/"humanoidverse"/relative
        tree = ast.parse(path.read_text())
        functions = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
                     for node in cls.body if isinstance(node, ast.FunctionDef)
                     and (node.name.startswith("_reward_") or node.name == "_compute_torques")]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class AgentSettingsTests(unittest.TestCase):
    def test_robot_selections_match_models_and_g1_only_has_ankle_roll_cost(self):
        for robot in ("h1", "g1"):
            profile = get_agent_config(f"{robot.upper()}-Motion-Prior")
            if robot == "h1":
                from robots.humanoid.h1.config import get_spec
            else:
                from robots.humanoid.g1.config import get_spec
            model = get_spec().compile()
            self.assertEqual(profile.joint_names, tuple(model.joint(i).name for i in range(1, model.njnt)))
            for name in profile.feet_body_names:
                self.assertGreater(model.body(name).id, 0)
            bodies = [model.body(i).name for i in range(1, model.nbody)]
            for pattern in profile.undesired_contact_patterns:
                self.assertTrue(any(re.fullmatch(pattern, name) for name in bodies))
            self.assertEqual("penalty_ankle_roll" in profile.rewards.weights, robot == "g1")

    def test_reward_overrides_snapshot_and_legacy_resume(self):
        values = {"auxiliary_rewards": {"penalty_slippage": -3.0},
                  "auxiliary_reward_parameters": {"contact_force_threshold": 2.0}}
        profile = resolve_agent_config(values)
        self.assertEqual(profile.rewards.weights["penalty_slippage"], -3.0)
        self.assertEqual(profile.rewards.weights["penalty_action_rate"], -0.1)
        restored = get_agent_config(snapshot=json.loads(json.dumps(values["motion_prior_agent"])))
        self.assertEqual(restored.to_dict(), profile.to_dict())
        self.assertEqual(get_agent_config().rewards.contact_force_threshold, 1.0)
        self.assertEqual(resolve_agent_config(values, legacy=True).rewards.feet_orientation_contact, "vertical")
        old = {"auxiliary_rewards": {"penalty_action_rate": -0.1}}
        legacy = resolve_agent_config(old, legacy=True)
        self.assertEqual(legacy.rewards.feet_orientation_contact, "magnitude")
        self.assertEqual(legacy.action_observation_units, "normalized")
        self.assertFalse(legacy.history_uses_previous_action)
        self.assertEqual((profile.lying_reset_mode, profile.lying_root_height), ("reference", .5))
        self.assertEqual((legacy.lying_reset_mode, legacy.lying_root_height), ("neutral", .25))
        with self.assertRaisesRegex(ValueError, "no ankle-roll"):
            get_agent_config().with_rewards({"penalty_ankle_roll": -4})
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            get_agent_config().with_rewards({"misspelled_reward": -1})

    def test_all_robot_costs_match_original_functions_including_horizontal_contacts(self):
        functions = original_reward_functions()
        torch.manual_seed(17)
        for robot_name in ("h1", "g1"):
            profile = get_agent_config(f"{robot_name.upper()}-Motion-Prior")
            joints, feet = profile.joint_names, profile.feet_body_names
            n = len(joints)
            orientation = torch.randn(4, 2, 4)
            orientation /= orientation.norm(dim=-1, keepdim=True)
            limits = torch.tensor([-1.0, 1.0]).expand(4, n, 2)
            position = torch.randn(4, n)
            velocity = torch.randn(4, 2, 3)
            data = NS(joint_pos=position, joint_pos_limits=limits,
                      body_link_quat_w=orientation, body_link_lin_vel_w=velocity)
            robot = NS(data=data, joint_names=joints,
                       find_bodies=lambda names, **kwargs: ([feet.index(name) for name in names], names),
                       find_joints=lambda names, **kwargs: ([joints.index(name) for name in names], names))
            # Env 1 touches sideways only. Env 2 has downward force on the foot.
            force = torch.tensor([[[0., 0., -200.], [0., 0., -100.]],
                                  [[100., 0., 0.], [0., 100., 0.]],
                                  [[0., 0., 200.], [0., 0., 100.]],
                                  [[0., 0., 0.], [0., 0., -0.5]]])
            undesired = torch.zeros(4, 1, 3); undesired[0, 0, 0] = 5
            scene = {"robot": robot, "prior_feet": NS(primary_names=list(feet), data=NS(force=force)),
                     "prior_undesired": NS(data=NS(force=undesired))}
            env = NS(scene=scene, num_envs=4, device="cpu",
                     cfg=NS(motion_prior_agent=profile, robot_control={"normalize_action_to": 5.0}))
            action, previous = torch.randn(4, n).clamp(-1, 1), torch.zeros(4, n)
            actual = AuxiliaryRewards(env)(env, action, previous)
            original = NS(num_envs=4, device="cpu", actions=action*5, last_actions=previous*5,
                          use_reward_limits_dof_pos_curriculum=False, feet_indices=[0, 1],
                          gravity_vec=torch.tensor([0., 0., -1.]).expand(4, 3), penalised_contact_indices=[2],
                          left_ankle_dof_indices=[joints.index(profile.ankle_roll_joint_names[0])-1,
                                                 joints.index(profile.ankle_roll_joint_names[0])] if robot_name == "g1" else [],
                          right_ankle_dof_indices=[joints.index(profile.ankle_roll_joint_names[1])-1,
                                                  joints.index(profile.ankle_roll_joint_names[1])] if robot_name == "g1" else [],
                          simulator=NS(dof_pos=position, dof_pos_limits=torch.tensor([-.95, .95]).expand(n, 2),
                                       _rigid_body_rot=orientation[..., [1, 2, 3, 0]], _rigid_body_vel=velocity,
                                       contact_forces=torch.cat((-force, undesired), dim=1)))
            for name, value in actual.items():
                expected = functions[f"_reward_{name}"](original)
                torch.testing.assert_close(value[:, 0], expected, msg=f"{robot_name}/{name}")
            self.assertGreater(actual["penalty_feet_ori"][0].item(), 0)
            torch.testing.assert_close(actual["penalty_feet_ori"][1:], torch.zeros(3, 1))
            self.assertGreater(actual["penalty_slippage"][1].item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for native mjlab action processing")
    def test_h1_action_targets_reproduce_original_pd_rescaling(self):
        from mjlab.envs import ManagerBasedRlEnv
        from humanoid_rl.tasks.motion_prior.env_cfg import make_env_cfg
        from robots.humanoid.h1.mjlab import get_action_scale, get_effort_limits, load_control
        control = load_control()
        cfg = make_env_cfg(); cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg, device="cuda:0")
        try:
            env.reset()
            names = env.scene["robot"].joint_names
            action = torch.linspace(-1, 1, len(names), device=env.device)[None]
            env.action_manager.process_action(action)
            targets = env.action_manager.get_term("joint_pos")._processed_actions.cpu()
            kp = torch.tensor([control["stiffness"][name] for name in names])
            kd = torch.tensor([control["damping"][name] for name in names])
            effort = torch.tensor([get_effort_limits()[name] for name in names])
            default = env.scene["robot"].data.default_joint_pos.cpu()
            position, velocity = default+.02, torch.full_like(default, .03)
            source = NS(config=NS(robot=NS(control=OmegaConf.create(dict(action_scale=.25, action_rescale=True,
                                                                        control_type="P", clip_torques=True)),
                                          dof_effort_limit_list=effort.tolist()),
                                 domain_rand=NS(randomize_torque_rfi=False)),
                        device="cpu", p_gains=kp, d_gains=kd, _kp_scale=1., _kd_scale=1.,
                        default_dof_pos=default, default_dof_pos_offset=0.,
                        simulator=NS(dof_pos=position, dof_vel=velocity),
                        _torque_noise=torch.zeros_like(default), torque_limits=effort)
            expected = original_reward_functions()["_compute_torques"](source, action.cpu()*5)
            actual = ((targets-position)*kp-kd*velocity).clamp(-effort, effort)
            torch.testing.assert_close(actual, expected)
            legacy = dict(control)
            legacy.pop("action_rescale"); legacy.pop("clip_position_targets")
            self.assertEqual(get_action_scale(legacy), 1.25)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
