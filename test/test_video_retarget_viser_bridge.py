"""Unit and integration tests for Viser Video-to-Robot Dance Studio bridge (CPU-only)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TEST_DIR = Path(__file__).resolve().parent
for _p in (SRC_DIR, TEST_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from humanoid_rl.sim2sim.video_retarget_viser_bridge import (
    DEFAULT_SMPLX_NPZ,
    VideoDanceStudioPanel,
    append_clip_to_motion_loader,
    download_and_decode_video,
    load_motion_prediction_file,
    retarget_smplx_to_light_50hz,
    synced_video_frame_index,
)


class _DummyMotion:
    def __init__(self, initial_frames: int = 20) -> None:
        self.root_pos_w = torch.zeros((initial_frames, 3), dtype=torch.float32)
        self.root_quat_w = torch.zeros((initial_frames, 4), dtype=torch.float32)
        self.root_quat_w[:, 0] = 1.0
        self.root_lin_vel_w = torch.zeros((initial_frames, 3), dtype=torch.float32)
        self.root_ang_vel_w = torch.zeros((initial_frames, 3), dtype=torch.float32)
        self.joint_pos = torch.zeros((initial_frames, 26), dtype=torch.float32)
        self.joint_vel = torch.zeros((initial_frames, 26), dtype=torch.float32)
        self.time_step_total = initial_frames
        self.motion_ends = torch.tensor([initial_frames], dtype=torch.long)
        self.motion_names = ["init/clip/0-20"]


class _DummyMotionCmd:
    def __init__(self) -> None:
        self.motion = _DummyMotion(20)
        self.time_steps = torch.tensor([0], dtype=torch.long)
        self.bin_count = 3
        self.bin_failed_count = torch.zeros(3, dtype=torch.float32)
        self._current_bin_failed = torch.zeros(3, dtype=torch.float32)
        self._external_probs = torch.full((3,), 1.0 / 3.0)


class _DummyCmdMgr:
    def __init__(self, cmd) -> None:
        self._cmd = cmd

    def get_term(self, name: str):
        assert name == "motion"
        return self._cmd


class _DummySimData:
    def __init__(self) -> None:
        self.qpos = torch.zeros((1, 33), dtype=torch.float32)
        self.qpos[:, 2] = 0.72
        self.qpos[:, 3] = 1.0


class _DummySim:
    def __init__(self) -> None:
        self.data = _DummySimData()


class _DummyBaseEnv:
    def __init__(self) -> None:
        self.step_dt = 0.02
        self.command_manager = _DummyCmdMgr(_DummyMotionCmd())
        self.sim = _DummySim()


class _DummyEnvWrapper:
    def __init__(self) -> None:
        self.unwrapped = _DummyBaseEnv()
        self.device = "cpu"


class _DummyViewer:
    def __init__(self) -> None:
        self._sim_lock = threading.Lock()
        self.env = _DummyEnvWrapper()
        self.policy = torch.nn.Linear(2, 2)
        self.task_specs = {}
        self._replay_goal_tables = {}
        self._ordered_playlist = [("00: init", 0)]
        self.motion_names = ["init/clip/0-20"]
        self._playlist_pos = 0
        self._selected_clip_idx = 0
        self._suppress_dd_cb = False
        self._clip_dropdown = None

    def _apply_selected_clip(self) -> None:
        cmd = self.env.unwrapped.command_manager.get_term("motion")
        idx = self._selected_clip_idx
        start = 0 if idx == 0 else int(cmd.motion.motion_ends[idx - 1].item())
        cmd.time_steps[0] = start

    def reset_wm_visualizer(self) -> None:
        pass


class TestVideoRetargetViserBridge(unittest.TestCase):
    def test_smplx_to_light_50hz_retarget_from_gvhmr_pt(self) -> None:
        self.assertTrue(DEFAULT_SMPLX_NPZ.is_file(), f"Missing {DEFAULT_SMPLX_NPZ}")
        fps = 30.0
        n_frames = 37  # ~1.2s dance snippet @ 30 Hz
        t = np.arange(n_frames, dtype=np.float32) / fps

        # Gentle idol dance arm wave + knee bounce in GVHMR Y-up SMPL-X coordinates
        global_orient = np.zeros((n_frames, 3), dtype=np.float32)
        body_pose = np.zeros((n_frames, 21, 3), dtype=np.float32)
        # Bring arms softly down from T-pose and wave forearms
        body_pose[:, 15, 2] = -0.75 + 0.15 * np.sin(2.0 * np.pi * t)  # left_shoulder (idx 16 -> body_pose 15)
        body_pose[:, 16, 2] = 0.75 - 0.15 * np.sin(2.0 * np.pi * t)   # right_shoulder (idx 17 -> body_pose 16)
        body_pose[:, 17, 1] = -0.35 + 0.20 * np.sin(2.0 * np.pi * t)  # left_elbow
        body_pose[:, 18, 1] = 0.35 - 0.20 * np.sin(2.0 * np.pi * t)   # right_elbow
        transl = np.zeros((n_frames, 3), dtype=np.float32)
        betas = np.zeros((n_frames, 10), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pt_path = tmp_dir / "hmr4d_results.pt"
            torch.save(
                {
                    "smpl_params_global": {
                        "global_orient": torch.from_numpy(global_orient),
                        "body_pose": torch.from_numpy(body_pose.reshape(n_frames, 63)),
                        "transl": torch.from_numpy(transl),
                        "betas": torch.from_numpy(betas),
                    }
                },
                pt_path,
            )

            names, pos, rot, _, loaded_fps = load_motion_prediction_file(pt_path, override_fps=fps)
            self.assertEqual(len(names), 22)
            self.assertEqual(pos.shape, (n_frames, 22, 3))
            self.assertEqual(rot.shape, (n_frames, 22, 3, 3))
            self.assertAlmostEqual(loaded_fps, 30.0)
            # Head must be well above pelvis in Z-up
            self.assertGreater(
                float(pos[0, names.index("Head"), 2] - pos[0, names.index("Hips"), 2]),
                0.45,
            )

            out_npz = tmp_dir / "dance_light_50hz.npz"
            qpos, qvel, report = retarget_smplx_to_light_50hz(
                pt_path,
                out_npz,
                source_fps_override=fps,
                start_s=0.0,
                duration_s=1.0,
                target_fps=50.0,
            )
            self.assertEqual(qpos.shape, (51, 33))
            self.assertEqual(qvel.shape, (51, 32))
            self.assertLessEqual(report["max_joint_limit_excess_rad"], 1e-5)
            self.assertGreaterEqual(report["min_collision_mesh_height_m"], -0.0015)
            self.assertGreaterEqual(report["min_selected_self_collision_distance_m"], -0.0015)
            self.assertTrue(out_npz.is_file())

    def test_video_decode_frame_sync_and_viser_panel_hot_swap(self) -> None:
        import viser

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            mp4 = tmp_dir / "sample.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=duration=2.0:size=320x180:rate=30",
                    "-pix_fmt",
                    "yuv420p",
                    str(mp4),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _, frames, fps = download_and_decode_video(
                str(mp4), start_s=0.2, duration_s=1.0, work_dir=tmp_dir / "work", max_height=180
            )
            self.assertAlmostEqual(fps, 30.0, places=2)
            self.assertGreaterEqual(len(frames), 28)
            self.assertEqual(frames.shape[1:], (180, 320, 3))

            # 50 Hz robot step 25 (0.50s) -> 30 fps video frame 15
            self.assertEqual(synced_video_frame_index(125, 100, 30, 30.0, 50.0), 15)
            # Reset back to clip_start -> frame 0
            self.assertEqual(synced_video_frame_index(100, 100, 30, 30.0, 50.0), 0)

            # Ephemeral Viser server + hot-swap + 3D screen sync check
            viewer = _DummyViewer()
            server = viser.ViserServer(host="127.0.0.1", port=0)
            try:
                viewer._clip_dropdown = server.gui.add_dropdown(
                    "Showcase Clip", options=["00: init"], initial_value="00: init"
                )
                panel = VideoDanceStudioPanel(viewer, server)
                qpos = np.zeros((51, 33), dtype=np.float32)
                qpos[:, 2] = 0.72
                # 90-deg Z yaw quaternion wxyz = [cos(45), 0, 0, sin(45)]
                qpos[:, 3] = np.sqrt(0.5)
                qpos[:, 6] = np.sqrt(0.5)
                qvel = np.zeros((51, 32), dtype=np.float32)
                qvel[:, 3] = 1.0  # +1 rad/s body-X -> +1 rad/s world-Y after +90 deg Z yaw

                new_idx = panel.register_and_activate_clip(
                    qpos=qpos,
                    qvel=qvel,
                    label="TestIdol",
                    clip_key="studio/TestIdol/0-51",
                    video_frames=frames,
                    video_fps=fps,
                )
                self.assertEqual(new_idx, 1)
                cmd = viewer.env.unwrapped.command_manager.get_term("motion")
                self.assertEqual(cmd.motion.time_step_total, 71)
                # Verify body->world angular velocity rotation parity
                np.testing.assert_allclose(
                    cmd.motion.root_ang_vel_w[20].numpy(),
                    np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    atol=1e-5,
                )
                panel.update_synced_video_frame()
                self.assertIsNotNone(panel._screen_handle)
                cmd.time_steps[0] = 45  # +25 frames into clip 1 -> video frame 15
                panel.update_synced_video_frame()
                self.assertEqual(panel._last_rendered_key, (1, 15))
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
