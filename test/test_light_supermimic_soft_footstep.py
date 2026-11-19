"""Unit tests for soft-footstep BPTT reward math, gradient flow, and running fine-tune config."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class SoftFootstepBpttTests(unittest.TestCase):
    def test_run_soft_plan_and_manifest_split_isolation(self) -> None:
        from humanoid_rl.launchers.config import load_named, resolve

        exp = load_named("light_supermimic_run_soft", "experiments", REPO_ROOT)
        rt = load_named("local", "runtimes", REPO_ROOT)
        plan = resolve(exp, rt, root=REPO_ROOT, run_id="unit_test_run_soft")
        self.assertTrue(plan.settings["environment"]["soft_footstep"]["enabled"])
        self.assertTrue(Path(plan.settings["training"]["init_from"]).exists())

        manifest = json.loads(Path(plan.settings["motion"]["manifest"]).read_text())
        train_subjects = {r["subject"] for r in manifest["splits"]["train"]}
        val_subjects = {r["subject"] for r in manifest["splits"]["validation"]}
        self.assertTrue(train_subjects.isdisjoint(val_subjects))
        self.assertEqual(val_subjects, {5})

    def test_soft_footstep_physics_gates_and_bptt_gradients(self) -> None:
        from humanoid_rl.tasks.motion_tracking.light_supermimic import (
            SOLE_OFFSETS_LOCAL,
            compute_soft_footstep_cost,
            resolve_soft_footstep_cfg,
        )

        cfg = resolve_soft_footstep_cfg({"enabled": True})
        params = {k: v for k, v in cfg.items() if k != "enabled"}
        sole_offsets = torch.tensor(SOLE_OFFSETS_LOCAL, dtype=torch.float32)

        # Exact upright ankle_roll_link rotation matrix from humanoid_light.xml
        r_upright = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        foot_rot = r_upright.view(1, 1, 3, 3).expand(2, 2, 3, 3).clone().requires_grad_(True)

        # Case 0: high aerial swing (ankle_z = 0.20m => sole_z = 0.157m), fast forward v_x=4.0m/s, downward v_z=-1.0m/s
        # Case 1: touchdown slam (ankle_z = 0.050m => sole_z = 0.007m), downward v_z=-1.3m/s, slip v_x=1.2m/s
        foot_pos = torch.tensor(
            [
                [[0.0, 0.1, 0.20], [0.0, -0.1, 0.20]],
                [[0.0, 0.1, 0.05], [0.0, -0.1, 0.20]],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        foot_lin_vel = torch.tensor(
            [
                [[4.0, 0.0, -1.0], [4.0, 0.0, 0.2]],
                [[1.2, 0.0, -1.3], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        foot_ang_vel = torch.zeros((2, 2, 3), dtype=torch.float32, requires_grad=True)
        action = torch.full((2, 26), 0.3, dtype=torch.float32, requires_grad=True)
        last_action = torch.zeros((2, 26), dtype=torch.float32)

        cost = compute_soft_footstep_cost(
            foot_pos,
            foot_rot,
            foot_lin_vel,
            foot_ang_vel,
            action,
            last_action,
            sole_offsets,
            **params,
        )
        # Aerial swing cost (minus identical leg action_rate) should be near zero, while touchdown slam cost is large
        leg_rate_only = params["leg_action_rate_weight"] * (12 * 0.3**2)
        self.assertLess(float(cost[0].item() - leg_rate_only), 0.01)
        self.assertGreater(float(cost[1].item() - leg_rate_only), 2.5)

        # Verify smooth analytic BPTT gradients flow cleanly to pos, rot, lin_vel, ang_vel, and action
        loss = cost.sum()
        loss.backward()
        for name, tensor in (
            ("foot_pos", foot_pos),
            ("foot_rot", foot_rot),
            ("foot_lin_vel", foot_lin_vel),
            ("foot_ang_vel", foot_ang_vel),
            ("action", action),
        ):
            self.assertIsNotNone(tensor.grad, msg=f"Missing grad for {name}")
            self.assertTrue(torch.isfinite(tensor.grad).all(), msg=f"Nonfinite grad for {name}")
        # Gradient wrt touchdown vertical velocity must push v_z UPWARD (dCost / dv_z < 0 => -grad increases v_z!)
        self.assertLess(float(foot_lin_vel.grad[1, 0, 2].item()), -1.0)

    def test_whole_body_3tier_contact_gates_and_bptt_gradients(self) -> None:
        from humanoid_rl.launchers.config import load_named, resolve
        from humanoid_rl.tasks.motion_tracking.light_supermimic import (
            KEY_BODY_NAMES,
            build_whole_body_contact_tensors,
            compute_whole_body_contact_cost,
            resolve_whole_body_contact_cfg,
        )

        exp = load_named("light_supermimic_lafan_cmu_wb_soft", "experiments", REPO_ROOT)
        rt = load_named("local", "runtimes", REPO_ROOT)
        plan = resolve(exp, rt, root=REPO_ROOT, run_id="unit_test_wb_soft")
        self.assertTrue(plan.settings["environment"]["whole_body_contact"]["enabled"])
        self.assertTrue(Path(plan.settings["training"]["init_from"]).exists())

        wb_cfg = resolve_whole_body_contact_cfg({"enabled": True})
        t = build_whole_body_contact_tensors(KEY_BODY_NAMES, wb_cfg, device="cpu")
        scalar_keys = (
            "v_free",
            "impact_height",
            "impact_tau",
            "slip_height",
            "slip_tau",
            "head_safe_height",
            "unintended_height",
            "ref_air_height",
            "ref_air_tau",
            "clear_tau",
            "extra_action_rate_weight",
        )
        scalars = {k: wb_cfg[k] for k in scalar_keys}

        # Batch of 4 scenarios across all 15 key bodies:
        # 0: Normal upright high pose (all bodies high z=0.8m) -> near-zero cost
        # 1: Head near floor (head z=0.02m, even if ref head is also z=0.02m!) -> Tier-1 forbid penalty fires!
        # 2: Left wrist intentionally on floor (pred z=0.04m, ref z=0.04m, v=0) -> static support allowed (low cost)
        # 3: Left wrist slamming floor at v_z=-1.5m/s while ref is airborne (ref z=0.50m) -> large impact + unintended cost!
        pred_pos = torch.zeros((4, 15, 3), dtype=torch.float32)
        pred_pos[..., 2] = 0.80
        target_pos = pred_pos.clone()

        head_idx = KEY_BODY_NAMES.index("head_link")
        lwrist_idx = KEY_BODY_NAMES.index("left_wrist_link")

        pred_pos[1, head_idx, 2] = 0.02
        target_pos[1, head_idx, 2] = 0.02

        pred_pos[2, lwrist_idx, 2] = 0.045
        target_pos[2, lwrist_idx, 2] = 0.045

        pred_pos[3, lwrist_idx, 2] = 0.01
        target_pos[3, lwrist_idx, 2] = 0.50

        pred_pos.requires_grad_(True)
        pred_rot = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).expand(4, 15, 3, 3).clone().requires_grad_(True)
        pred_lin_vel = torch.zeros((4, 15, 3), dtype=torch.float32)
        pred_lin_vel[3, lwrist_idx, 2] = -1.5
        pred_lin_vel.requires_grad_(True)
        pred_ang_vel = torch.zeros((4, 15, 3), dtype=torch.float32, requires_grad=True)
        target_quat = torch.zeros((4, 15, 4), dtype=torch.float32)
        target_quat[..., 0] = 1.0
        action = torch.zeros((4, 26), dtype=torch.float32, requires_grad=True)
        last_action = torch.zeros((4, 26), dtype=torch.float32)

        costs = compute_whole_body_contact_cost(
            pred_pos,
            pred_rot,
            pred_lin_vel,
            pred_ang_vel,
            target_pos,
            target_quat,
            action,
            last_action,
            t["body_idx"],
            t["local_offsets"],
            t["radii"],
            t["impact_w"],
            t["slip_w"],
            t["forbid_w"],
            t["unintend_w"],
            **scalars,
        )
        self.assertLess(float(costs[0].item()), 1e-3)
        self.assertGreater(float(costs[1].item()), 1.0)  # Head strictly protected even if ref is low
        self.assertLess(float(costs[2].item()), 0.05)    # Intentional static hand support allowed!
        self.assertGreater(float(costs[3].item()), 4.0)  # Hand slam + unintended contact strongly penalized!

        costs.sum().backward()
        self.assertTrue(torch.isfinite(pred_pos.grad).all())
        self.assertTrue(torch.isfinite(pred_lin_vel.grad).all())
        self.assertLess(float(pred_pos.grad[1, head_idx, 2].item()), -5.0)  # Pushes head UP away from floor!

    def test_velocity_tasks_cfg_and_stratified_sampler(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from humanoid_rl.algorithm.supermimic.train import (
            build_locomotion_frame_sampler,
            parse_tasks_cfg,
        )

        tasks_cfg = parse_tasks_cfg(
            {
                "velocity": {
                    "lin_vel_x_range": [-1.5, 2.8],
                    "lin_vel_y_range": [-0.8, 0.8],
                    "ang_vel_z_range": [-3.14, 3.14],
                    "window_frames": 25,
                    "command_noise": [0.15, 0.12, 0.25],
                }
            }
        )
        self.assertEqual(tasks_cfg.velocity.lin_vel_x_range, (-1.5, 2.8))
        self.assertEqual(tasks_cfg.velocity.command_noise, (0.15, 0.12, 0.25))

        with tempfile.TemporaryDirectory() as td:
            pdir = Path(td)
            (pdir / "motion_names.json").write_text(
                json.dumps(["cmu/90_01/0-50", "cmu/49_01/0-50", "lafan/run1_subject2/0-50"])
            )
            allow = pdir / "allow.json"
            allow.write_text(
                json.dumps({"version": 1, "sequence_prefixes": ["cmu/49_01", "lafan/run1_subject2"]})
            )
            vel = torch.zeros((150, 3), dtype=torch.float32)
            vel[50:65, 0] = 0.02   # stand in cmu/49_01
            vel[65:100, 0] = 0.90  # walk in cmu/49_01
            vel[100:150, 0] = 2.30 # run in lafan/run1_subject2
            fake_motion = SimpleNamespace(
                motion_ends=torch.tensor([50, 100, 150], dtype=torch.long),
                root_lin_vel_w=vel,
            )
            f_ids, probs = build_locomotion_frame_sampler(pdir, fake_motion, allow, "cpu")
            self.assertTrue((f_ids >= 50).all())  # cmu/90_01 backflip completely excluded!
            self.assertAlmostEqual(float(probs.sum().item()), 1.0, places=5)

    def test_s2r_goal_twist_dr_and_adapter_parity(self) -> None:
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        test_dir = str(Path(__file__).resolve().parent)
        if test_dir not in sys.path:
            sys.path.insert(0, test_dir)
        from humanoid_rl.tasks.motion_tracking.light_supermimic import (
            motion_goal_window_with_twist,
            resolve_domain_randomization_cfg,
        )
        from robots.humanoid.humanoid_light.mjlab import get_robot_spec, load_control
        from humanoid_rl.sim2sim.sim2sim_supermimic_adapter import build_motion_actor_goal, build_proprio_obs
        from supermimic.representations import (
            _assemble_goal,
            extract_actor_goal,
            motion_anchor_window,
        )

        # 1. Foot collision geoms named in memory without touching humanoid_light.xml
        mj_model = get_robot_spec(load_control()).compile()
        geom_names = {mj_model.geom(i).name for i in range(mj_model.ngeom)}
        self.assertIn("left_ankle_roll_link_collision", geom_names)
        self.assertIn("right_ankle_roll_link_collision", geom_names)

        dr = resolve_domain_randomization_cfg({"enabled": True, "foot_friction_range": [0.4, 1.25]})
        self.assertTrue(dr["enabled"])
        self.assertEqual(dr["foot_friction_range"], (0.4, 1.25))

        # 2. Goal root twist (670-D wm_goal -> 640-D actor_goal) parity & mat-height invariance
        T = 60
        torch.manual_seed(7)
        root_pos_w = torch.randn((T, 3))
        q_raw = torch.randn((T, 4))
        root_quat_w = q_raw / torch.linalg.vector_norm(q_raw, dim=-1, keepdim=True)
        root_lin_vel_w = torch.randn((T, 3))
        root_ang_vel_w = torch.randn((T, 3))
        joint_pos = torch.randn((T, 26))
        joint_vel = torch.randn((T, 26))
        motion_ends = torch.tensor([T], dtype=torch.long)

        fake_mot = SimpleNamespace(
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            root_lin_vel_w=root_lin_vel_w,
            root_ang_vel_w=root_ang_vel_w,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            motion_ends=motion_ends,
        )
        t_idx = torch.tensor([0, 12], dtype=torch.long)
        r_pos = torch.zeros((2, 3))
        r_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        content = motion_goal_window_with_twist(fake_mot, t_idx)
        self.assertEqual(content.shape, (2, 580))
        t_pos, t_quat = motion_anchor_window(fake_mot, t_idx)
        wm_goal, actor_goal = _assemble_goal(content, r_pos, r_quat, t_pos, t_quat)
        self.assertEqual(wm_goal.shape, (2, 670))
        self.assertEqual(actor_goal.shape, (2, 640))
        self.assertTrue(torch.allclose(extract_actor_goal(wm_goal), actor_goal, atol=1e-6))

        # Raising clip height by +0.30 m (platform/mat) must NOT change actor_goal at all!
        fake_mot_elevated = SimpleNamespace(
            **{**fake_mot.__dict__, "root_pos_w": root_pos_w + torch.tensor([0.0, 0.0, 0.30])}
        )
        content_elev = motion_goal_window_with_twist(fake_mot_elevated, t_idx)
        self.assertTrue(torch.allclose(content, content_elev, atol=1e-6))

        # Sim2Sim adapter parity
        adapter_goal = build_motion_actor_goal(
            fake_mot, t_idx, r_pos, r_quat, goal_root_twist=True
        )
        self.assertTrue(torch.allclose(actor_goal, adapter_goal, atol=1e-5))

        prop = build_proprio_obs(
            r_quat, torch.zeros((2, 26)), torch.zeros((2, 3)), torch.zeros((2, 26)), torch.zeros((2, 26))
        )
        self.assertEqual(prop.shape, (2, 84))


if __name__ == "__main__":
    unittest.main()
