"""The copied H1 and numeric mocap states load without upstream repositories."""

from pathlib import Path
import shutil
import tempfile
import unittest

import mujoco
import numpy as np

from robots.humanoid.h1.config import ASSET_DIR, JOINT_NAMES, KEY_SITE_NAMES, get_spec
from humanoid_retarget.datasets.lafan1 import H1_DIR
from humanoid_retarget.datasets.robot_motion import load_robot_motion


class H1AssetTests(unittest.TestCase):
    def test_model_loads_after_copying_only_the_h1_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            isolated = Path(directory)/"h1"
            shutil.copytree(ASSET_DIR, isolated, ignore=shutil.ignore_patterns("__pycache__"))
            model = mujoco.MjModel.from_xml_path(str(isolated/"h1.xml"))
        self.assertEqual((model.nq, model.nv, model.nu), (26, 25, 19))
        self.assertEqual(model.nmesh, 21)
        self.assertEqual(tuple(model.joint(i).name for i in range(1, model.njnt)), JOINT_NAMES)

    @unittest.skipUnless((H1_DIR/"walk1_subject1.npz").is_file(), "Local H1 motion data not installed")
    def test_copied_motion_matches_model_and_forward_kinematics(self):
        motion = load_robot_motion(H1_DIR/"walk1_subject1.npz")
        model = get_spec().compile()
        self.assertEqual(motion.joint_names, ("root",)+JOINT_NAMES)
        self.assertEqual(motion.fps, 40)
        self.assertEqual(motion.qpos.shape, (10451, model.nq))
        self.assertEqual(motion.qvel.shape, (10451, model.nv))
        data = mujoco.MjData(model)
        sites = [model.site(name).id for name in KEY_SITE_NAMES]
        for index in (0, 100, motion.num_frames-1):
            data.qpos[:], data.qvel[:] = motion.qpos[index], motion.qvel[index]
            mujoco.mj_forward(model, data)
            self.assertTrue(np.isfinite(data.site_xpos[sites]).all())

    def test_numeric_reader_ignores_object_metadata_and_rejects_invalid_quaternions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"motion.npz"
            qpos = np.zeros((2, 26), dtype=np.float32)
            qpos[:, 3] = 1
            values = dict(qpos=qpos, qvel=np.zeros((2, 25)), frequency=40,
                          joint_names=np.asarray(("root",)+JOINT_NAMES),
                          split_points=np.asarray([0, 2]), metadata=np.asarray(None, dtype=object))
            np.savez(path, **values)
            self.assertEqual(load_robot_motion(path).num_frames, 2)
            qpos[:, 3] = 0
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "Non-unit"):
                load_robot_motion(path)


if __name__ == "__main__":
    unittest.main()
