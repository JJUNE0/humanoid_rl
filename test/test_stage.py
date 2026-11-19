"""Saved-predictor parity, coordinate conversion, frozen weights and portability."""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_mul

from humanoid_rl.tasks.carry_box.stage.config import StageModelConfig
from humanoid_rl.tasks.carry_box.stage.model import FrozenCarryPredictor
from humanoid_rl.tasks.carry_box.stage.observations import state101

SOURCE = Path(__file__).resolve().parents[2] / "I-BFM-Code" / "ibfm"
CHECKPOINT = SOURCE / "analysis/carry_mlp_failure_v1/seed42/best.pt"
REFERENCE = SOURCE / "tasks/carry_box/rollouts/carry_20_40_success500_v1/episode_00004"


def source_module():
    path = SOURCE / "tasks/carry_box/tools/carry_stage_state_v2.py"
    spec = importlib.util.spec_from_file_location("carry_reference_stage_schema", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(CHECKPOINT.exists() and REFERENCE.exists(), "User's reference checkpoint/data unavailable")
class StagePredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = StageModelConfig.from_reference(CHECKPOINT, REFERENCE)

    def test_matches_saved_original_predictions_without_trainable_weights(self):
        predictor = FrozenCarryPredictor(self.cfg, "cpu")
        with np.load(SOURCE / "analysis/carry_stage_state_v3/test.npz") as data:
            raw = torch.tensor(data["x"][:1024])
        with np.load(CHECKPOINT.parent / "test_predictions.npz") as saved:
            actual = predictor(raw)
            self.assertEqual(actual.shape, (1024, 28))
            np.testing.assert_array_equal(actual[:, :9].argmax(-1).numpy(), saved["stage_predictions"][:1024])
            np.testing.assert_array_equal(actual[:, 9:18].argmax(-1).numpy(), saved["skill_predictions"][:1024])
            # Stored predictions used GPU/TF32 in 8192-state batches. CPU and
            # differently shaped GEMMs differ by <1e-3 even with identical weights.
            np.testing.assert_allclose(actual[:, 18:27].numpy(), saved["progress_all_stages"][:1024], atol=1e-3)
            np.testing.assert_allclose(actual[:, 27].numpy(), saved["failure_probability"][:1024], atol=1e-3)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in predictor.parameters()))
        self.assertFalse(predictor.training)

    def test_bundle_restores_without_the_source_files_or_rng_changes(self):
        rng = torch.get_rng_state().clone()
        predictor = FrozenCarryPredictor(self.cfg, "cpu")
        torch.testing.assert_close(torch.get_rng_state(), rng)
        raw = torch.randn(16, 101)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage_model.pt"
            predictor.save_bundle(path)
            bundle = torch.load(path, weights_only=True)
            bundle["schema"]["source_episode"] = "/missing/reference"
            torch.save(bundle, path)
            cfg = StageModelConfig.from_bundle(path)
            # A relocated portable bundle must need no source episode or code.
            restored = FrozenCarryPredictor(cfg, "cpu")
            torch.testing.assert_close(restored(raw), predictor(raw))
            with self.assertRaises(ValueError):
                FrozenCarryPredictor(StageModelConfig(str(path), "incorrect", cfg.schema), "cpu")

    def test_state101_matches_original_numpy_encoder(self):
        reference = source_module()
        motion, metadata, box = reference.read_episode(REFERENCE)
        fk = reference.RobotFK(REFERENCE, metadata, box)
        ids = np.asarray([0, 65, 100, 200, 350, 500])
        points = fk.hand_points(motion, ids)
        expected = reference.actor_state(motion, ids, points, fk.defaults)
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32)
        root_quat = tensor(motion["root_quat"][ids])
        object_quat = tensor(motion["object_quat"][ids])
        object_angular = quat_apply(object_quat, tensor(motion["object_ang_vel_local"][ids]))
        com = quat_apply(object_quat, tensor(motion["object_com_offset"]).expand(len(ids), -1))
        actual = state101(
            tensor(motion["root_pos"][ids]), root_quat,
            quat_apply(root_quat, tensor(motion["root_ang_vel_local"][ids])),
            tensor(motion["dof_pos"][ids]), tensor(motion["dof_vel"][ids]), tensor(fk.defaults),
            tensor(motion["object_pos"][ids]), object_quat,
            tensor(motion["object_lin_vel"][ids]) + torch.cross(object_angular, com, dim=-1),
            object_angular, tensor(motion["goal"]).expand(len(ids), -1), tensor(points), tensor(motion["hand_contact"][ids]),
        )
        torch.testing.assert_close(actual, tensor(expected), atol=2e-5, rtol=2e-5)


@unittest.skipUnless(CHECKPOINT.exists() and REFERENCE.exists() and torch.cuda.is_available(), "Stage/mjlab integration requires reference data and CUDA")
class StageEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.envs import ManagerBasedRlEnv
        from humanoid_rl.tasks.carry_box.env_cfg import make_env_cfg
        from humanoid_rl.env.mjlab_adapter import MjlabAdapter
        cls.cfg = StageModelConfig.from_reference(CHECKPOINT, REFERENCE)
        cfg = make_env_cfg(stage_model=cls.cfg)
        cfg.scene.num_envs = 2
        cls.env = ManagerBasedRlEnv(cfg, device="cuda:0")
        cls.adapter = MjlabAdapter(cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_context_is_148d_and_terminal_predictions_precede_partial_reset(self):
        env = self.env
        self.assertEqual(self.adapter.observation_dim, 148)
        weights = {k: v.clone() for k, v in env.carry_stage_observer.predictor.state_dict().items()}
        env.cfg.episode_length_s = env.step_dt
        self.adapter.observations = env.reset()[0]["actor"].clone()
        transition, _ = self.adapter.step(torch.full((2, 29), .2, device=env.device))
        self.assertTrue(bool(transition["truncated"].all()))
        context = transition["next_observation"][:, -28:]
        self.assertTrue(bool(torch.isfinite(context).all()))
        torch.testing.assert_close(context[:, :9].sum(-1), torch.ones(2, device=env.device))
        self.assertFalse(torch.equal(context, self.adapter.observations[:, -28:]))
        for key, value in env.carry_stage_observer.predictor.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        env.cfg.episode_length_s = 30

    def test_virtual_box_frame_and_environment_origin_match_reference_conventions(self):
        env = self.env
        env.reset()
        observer = env.carry_stage_observer
        reference = source_module()
        motion, metadata, box = reference.read_episode(REFERENCE)
        fk = reference.RobotFK(REFERENCE, metadata, box)
        ids = np.asarray([100, 100])
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=env.device)
        robot, obj = env.scene["robot"], env.scene["box"]
        root_quat = tensor(motion["root_quat"][ids])
        root_state = torch.cat((tensor(motion["root_pos"][ids]) + env.scene.env_origins, root_quat,
                               tensor(motion["root_lin_vel"][ids]), quat_apply(root_quat, tensor(motion["root_ang_vel_local"][ids]))), -1)
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(tensor(motion["dof_pos"][ids]), tensor(motion["dof_vel"][ids]), joint_ids=observer.joints)
        object_quat = tensor(motion["object_quat"][ids])
        angular = quat_apply(object_quat, tensor(motion["object_ang_vel_local"][ids]))
        offset = quat_apply(object_quat, observer.geom_position.expand(2, -1))
        geometry_quat = quat_mul(object_quat, observer.geom_quat.expand(2, -1))
        geometry_vel = tensor(motion["object_lin_vel"][ids]) + torch.cross(angular, offset, dim=-1)
        obj.write_root_state_to_sim(torch.cat((tensor(motion["object_pos"][ids]) + offset + env.scene.env_origins,
                                               geometry_quat, geometry_vel, angular), -1))
        goal_offset = quat_apply(quat_conjugate(observer.geom_quat.expand(2, -1)), observer.geom_position.expand(2, -1))
        env.carry_goal_w[:] = tensor(motion["goal"])[None] + goal_offset + env.scene.env_origins
        env.scene.write_data_to_sim()
        env.sim.forward()
        actual = observer.raw_state(env)
        points = fk.hand_points(motion, ids)
        expected = tensor(reference.actor_state(motion, ids, points, fk.defaults))
        # Contact flags may differ between collision models. Other original
        # channels must agree; physical wrist kinematics differ by about 5 mm.
        for section in (slice(0, 82), slice(96, 101)):
            torch.testing.assert_close(actual[:, section], expected[:, section], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(actual[:, 84:96], expected[:, 84:96], atol=.006, rtol=0)
        torch.testing.assert_close(actual[0, :82], actual[1, :82], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(actual[0, 98:], actual[1, 98:], atol=3e-5, rtol=3e-5)


if __name__ == "__main__":
    unittest.main()
