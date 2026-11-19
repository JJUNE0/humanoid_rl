"""Check the official differentiable FK against Light's MuJoCo conventions."""

import importlib.util
import unittest

import mujoco
import numpy as np
import torch

from humanoid_rl.env.actions import joint_range_targets
from robots.humanoid.humanoid_light.config import get_spec


class LightMappingTests(unittest.TestCase):
    def test_asymmetric_action_mapping_and_its_gradient(self):
        action = torch.tensor([[-.5, .5], [2., -2.]], requires_grad=True)
        limits = torch.tensor([[-1., 2.], [-2., .5]])
        targets = joint_range_targets(action, torch.zeros(2), limits)
        torch.testing.assert_close(targets, torch.tensor([[-.5, .25], [2., -2.]]))
        targets.sum().backward()
        torch.testing.assert_close(action.grad, torch.tensor([[1., .5], [0., 0.]]))


@unittest.skipUnless(importlib.util.find_spec("mjlab.utils.forward_kinematics"), "Official Light environment required")
class LightForwardKinematicsTests(unittest.TestCase):
    def test_link_pose_and_velocity_match_mujoco_with_a_rotated_root(self):
        from mjlab.utils.forward_kinematics import ForwardKinematics
        model = get_spec().compile()
        data = mujoco.MjData(model)
        data.qpos[:3] = [.3, -.4, .8]
        q = np.array([.7, .2, -.1, .4]); q /= np.linalg.norm(q)
        data.qpos[3:7] = q
        data.qpos[7:] = np.linspace(-.07, .09, model.nq - 7)
        data.qvel[:] = np.linspace(-.2, .3, model.nv)
        mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data); mujoco.mj_comVel(model, data)
        bodies = torch.arange(1, model.nbody)
        fk = ForwardKinematics(model, bodies, torch.from_numpy(model.jnt_qposadr[1:].copy()),
                               torch.arange(len(bodies)), dtype=torch.float64)
        def tensor(x):
            return torch.from_numpy(np.asarray(x).copy()).double().unsqueeze(0)
        root_angular = data.xmat[1].reshape(3, 3) @ data.qvel[3:6]
        pos, quat, linear, angular = fk(tensor(data.qpos[:3]), tensor(q), tensor(data.qpos[7:]),
                                       tensor(data.qvel[:3]), tensor(root_angular), tensor(data.qvel[6:]))
        expected_angular = data.cvel[1:, :3]
        expected_linear = data.cvel[1:, 3:] - np.cross(expected_angular, data.subtree_com[1] - data.xpos[1:])
        np.testing.assert_allclose(pos[0], data.xpos[1:], atol=1e-10)
        np.testing.assert_allclose(quat[0].abs(), np.abs(data.xquat[1:]), atol=1e-10)
        np.testing.assert_allclose(angular[0], expected_angular, atol=1e-10)
        np.testing.assert_allclose(linear[0], expected_linear, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
