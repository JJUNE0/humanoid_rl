"""CPU parity test verifying Sim-to-Sim adapter math against official SuperMimic & Coupled PD."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TEST_DIR = REPO_ROOT / "test"
for p in (SRC_DIR, TEST_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CKPT = (
    REPO_ROOT
    / "runs/light_supermimic_walk1_t4/t4_walk1_fixed_20261118/training/checkpoints/step_3072000/last.pt"
)


class Sim2SimAdapterParityTests(unittest.TestCase):
    def test_coupled_motor_slot_targets_match_isaaclab_and_mjlab_torque_identically(self):
        from sim2sim_supermimic_adapter import (
            CONTROL_JSON_PATH,
            JOINT_NAMES,
            coupling_matrix,
            joint_targets_mj_to_isaac_actuator_slots,
        )

        torch.manual_seed(7)
        batch = 16
        q_target = torch.randn(batch, 26) * 0.4
        q_pos = torch.randn(batch, 26) * 0.3
        q_vel = torch.randn(batch, 26) * 1.2

        slot_targets = joint_targets_mj_to_isaac_actuator_slots(q_target)
        source = json.loads(CONTROL_JSON_PATH.read_text())
        name_to_idx = {n: i for i, n in enumerate(JOINT_NAMES)}

        for pair, group_name in (
            ("left_ankle", "ankles"),
            ("right_ankle", "ankles"),
            ("torso", "torso_pitch_roll"),
        ):
            grp = source["groups"][group_name]
            g1, g2, gamma = float(grp["gear_ratio_1"]), float(grp["gear_ratio_2"]), float(grp["gamma"])
            p_i = name_to_idx[f"{pair}_pitch_joint"]
            r_i = name_to_idx[f"{pair}_roll_joint"]
            kp = float(source["joints"][f"{pair}_pitch_joint"]["stiffness"])
            kd = float(source["joints"][f"{pair}_pitch_joint"]["damping"])
            lim = float(source["joints"][f"{pair}_pitch_joint"]["effort_limit"])

            # 1) mjlab coupled_pd_torque equation
            mat = torch.tensor(coupling_matrix(pair, g1, g2), dtype=torch.float32)
            tgt_pair = torch.stack([q_target[:, p_i], q_target[:, r_i]], dim=-1)
            pos_pair = torch.stack([q_pos[:, p_i], q_pos[:, r_i]], dim=-1)
            vel_pair = torch.stack([q_vel[:, p_i], q_vel[:, r_i]], dim=-1)
            mj_motor_tau = kp * ((tgt_pair - pos_pair) @ mat.T) + kd * ((-vel_pair) @ mat.T)
            mj_joint_tau = (gamma * (mj_motor_tau @ mat)).clamp(-lim, lim)

            # 2) cocelo-rl-isaaclab CoupledDelayedPDActuator equation (gear_actuator.py)
            pitch, roll = q_pos[:, p_i], q_pos[:, r_i]
            vp, vr = q_vel[:, p_i], q_vel[:, r_i]
            if pair == "left_ankle":
                m_pos = torch.stack([g1 * (roll - pitch), g2 * (roll + pitch)], dim=-1)
                m_vel = torch.stack([g1 * (vr - vp), g2 * (vr + vp)], dim=-1)
            elif pair == "right_ankle":
                m_pos = torch.stack([g1 * (roll + pitch), g2 * (roll - pitch)], dim=-1)
                m_vel = torch.stack([g1 * (vr + vp), g2 * (vr - vp)], dim=-1)
            else:
                m_pos = torch.stack([g1 * (roll - pitch), -g2 * (roll + pitch)], dim=-1)
                m_vel = torch.stack([g1 * (vr - vp), -g2 * (vr + vp)], dim=-1)

            cmd_m = torch.stack([slot_targets[:, p_i], slot_targets[:, r_i]], dim=-1)
            isaac_motor_tau = kp * (cmd_m - m_pos) + kd * (0.0 - m_vel)
            tm1, tm2 = isaac_motor_tau[:, 0], isaac_motor_tau[:, 1]
            if pair == "left_ankle":
                j_tau = torch.stack([-g1 * tm1 + g2 * tm2, g1 * tm1 + g2 * tm2], dim=-1)
            elif pair == "right_ankle":
                j_tau = torch.stack([g1 * tm1 - g2 * tm2, g1 * tm1 + g2 * tm2], dim=-1)
            else:
                j_tau = torch.stack([-g1 * tm1 - g2 * tm2, g1 * tm1 - g2 * tm2], dim=-1)
            isaac_joint_tau = (j_tau * gamma).clamp(-lim, lim)

            torch.testing.assert_close(isaac_joint_tau, mj_joint_tau, atol=1e-5, rtol=1e-5)

    def test_standalone_actor_and_obs_match_supermimic_upstream_when_available(self):
        from sim2sim_supermimic_adapter import (
            PackedMotionLibrary,
            build_motion_actor_goal,
            build_rep_state,
            build_standalone_actor_from_saved,
            load_checkpoint_compat,
            pack_manifest_pure_numpy,
        )

        saved, _, _ = load_checkpoint_compat(CKPT, device="cpu")
        standalone = build_standalone_actor_from_saved(saved, device="cpu")
        packed = pack_manifest_pure_numpy(
            REPO_ROOT / "configs/motions/light_supermimic_walk1.json",
            repo_root=REPO_ROOT,
            split="train",
        )
        motion = PackedMotionLibrary(packed, device="cpu")

        torch.manual_seed(11)
        bsz = 8
        times = torch.randint(0, motion.time_step_total - 20, (bsz,))
        root_pos = torch.randn(bsz, 3) * 0.2
        root_pos[:, 2] += 0.71
        root_quat = torch.randn(bsz, 4)
        root_quat = root_quat / root_quat.norm(dim=-1, keepdim=True)
        qj = torch.randn(bsz, 26) * 0.25
        lin_vel = torch.randn(bsz, 3) * 0.5
        ang_vel = torch.randn(bsz, 3) * 0.5
        vj = torch.randn(bsz, 26) * 1.0

        rep = build_rep_state(root_pos, root_quat, qj, lin_vel, ang_vel, vj)
        goal = build_motion_actor_goal(motion, times, root_pos, root_quat)
        with torch.no_grad():
            out_standalone = standalone(rep, goal)
        self.assertEqual(tuple(out_standalone.shape), (bsz, 26))
        self.assertTrue(torch.isfinite(out_standalone).all())

        # If running inside .venv_supermimic_light, also cross-check against upstream MultiEncoderAE
        try:
            from tensordict import TensorDict
            from supermimic.models import MultiEncoderAE
            from supermimic.representations import (
                assemble_anchor_window,
                extract_actor_goal,
                motion_anchor_window,
                motion_goal_window,
            )
            from supermimic.tasks.task_spec import TasksCfg, resolve_task_specs
            from supermimic.utils.gencoord_helpers import compute_gencoord_state
        except ImportError:
            return

        full_state = torch.cat([root_pos, root_quat, qj, lin_vel, ang_vel, vj], dim=-1)
        upstream_rep = compute_gencoord_state(full_state)
        torch.testing.assert_close(rep, upstream_rep, atol=1e-6, rtol=1e-6)

        tgt_pos, tgt_quat = motion_anchor_window(motion, times)
        rel_pos, rel_ori = assemble_anchor_window(root_pos, root_quat, tgt_pos, tgt_quat)
        wm_goal = torch.cat([motion_goal_window(motion, times), rel_pos, rel_ori], dim=-1)
        upstream_goal = extract_actor_goal(wm_goal)
        torch.testing.assert_close(goal, upstream_goal, atol=1e-6, rtol=1e-6)

        # Dummy rep object with properties needed by resolve_task_specs
        class _DummyRep:
            num_joints = 26

        specs = resolve_task_specs(TasksCfg(), _DummyRep(), 610, "cpu")
        runner_cfg = saved["settings"]["runner"]
        upstream_actor = MultiEncoderAE(
            obs_dim=62,
            action_dim=26,
            task_specs=specs,
            slot_mix={"motion": 1.0, "velocity": 0.0},
            latent_dim=32,
            hidden_dims=tuple(runner_cfg["actor_hidden_dims"]),
            latent_bc_weight=0.1,
            history_length=1,
        )
        upstream_actor.load_state_dict(saved["actor"])
        upstream_actor.eval()
        td = TensorDict(
            {"goal_motion": upstream_goal, "history": upstream_rep.unsqueeze(1)},
            batch_size=[bsz],
        )
        with torch.no_grad():
            out_upstream = upstream_actor(td)
        torch.testing.assert_close(out_standalone, out_upstream, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
