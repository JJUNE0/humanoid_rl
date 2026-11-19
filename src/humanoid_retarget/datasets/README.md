# Local motion datasets

All files listed below are independent local copies, not symlinks, Git LFS
pointers, Hugging Face cache references or runtime downloads.

| Dataset/representation | Local directory | Contents |
|---|---|---|
| Original human LAFAN1 | `lafan1/raw/` | 77 BVH sequences, 496,672 frames, approximately 30 Hz |
| LAFAN1 retargeted to H1 | `lafan1/retargeted/h1/` | 40 NPZ motions, 352,835 frames, 40 Hz |
| LAFAN1 retargeted to G1 | `lafan1/retargeted/g1/` | Existing BFM-Zero full and 10-second-clipped PKL export variants |
| LocoMuJoCo default mocap, H1 | `loco_mujoco/mocap/h1/` | 19 NPZ motions, 83,963 frames, 40 Hz |

The G1 PKLs are two exports of the same motion collection, not two individual
clips or two new datasets. Carry-box rollouts are robot-generated demonstrations
and remain separate from these human mocap datasets.

## Inspect and load

```bash
.venv/bin/python -m humanoid_retarget.datasets
```

```python
from humanoid_retarget.datasets.lafan1 import H1_DIR, raw_sequences
from humanoid_retarget.datasets.robot_motion import load_robot_motion

print(raw_sequences()[0])
motion = load_robot_motion(H1_DIR / "walk1_subject1.npz")
print(motion.qpos.shape, motion.qvel.shape, motion.fps)
```

H1 NPZ files store MuJoCo `qpos` (26D) and `qvel` (25D), joint names, frame
frequency and trajectory split points. The floating-root quaternion is wxyz.
The numeric reader does not load pickle/object metadata or require JAX or
LocoMuJoCo. No resampling or physical correction is performed here. The 40 Hz
H1 motions are aligned to 50 Hz by the BFM-Zero dataset adapter during training.
Its resampled/encoded cache lives in `outputs/bfm_zero_cache`; these source
archives remain unchanged.

## Sources and reproducibility

- Human BVH: https://github.com/ubisoft/ubisoft-laforge-animation-dataset
- H1 retargeted/default motions: https://huggingface.co/datasets/robfiras/loco-mujoco-datasets
- G1 exports: https://github.com/LeCAR-Lab/BFM-Zero

Each collection has a manifest with file sizes, SHA-256 hashes and source
revision/URLs. Upstream license notices and source documentation are retained.
The original LAFAN1 archive is retained in `lafan1/.downloads/lafan1.zip`.
The H1 and dataset copies can be reconstructed using:

```bash
.venv/bin/python scripts/prepare_h1_data.py
```

This setup script copies the already-downloaded G1 PKLs from the BFM-Zero
checkout. Runtime readers use only the resulting files in this directory.
Large BVH/NPZ/PKL/archive files are excluded from Git and Python distributions;
they remain physically present here for the repository's editable installation.
Code, manifests, licenses and robot assets are packaged/versioned normally.

AMASS is another supported source in the upstream project, but is not copied
here: it requires an account and separate dataset access. Human-to-H1
retargeting for additional sources is not implemented by this data-copy step.
The current BFM-Zero integration reads the copied H1 NPZ files directly and
constructs expert features with the same model and conventions as mjlab.
