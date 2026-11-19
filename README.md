# Humanoid RL

Unified humanoid motion retargeting, reinforcement learning, and Sim-to-Sim / Sim-to-Real validation framework for `Humanoid-Light`, `H1`, and `G1`.

---

## Repository Layout

```text
humanoid_rl/
├── src/
│   ├── humanoid_rl/
│   │   ├── algorithm/       # Learning cores & adapters (see algorithm/README.md)
│   │   ├── configs/         # Experiment JSONs, runtime profiles, and scene/AMP YAMLs
│   │   ├── environments/    # Method-specific Python/uv environment locks
│   │   ├── env/             # mjlab vector-env wrappers, controllers, and imitation helpers
│   │   ├── launchers/       # Unified local & Colab execution / bundling engine
│   │   ├── methods/         # Method registry & CLI adapters
│   │   ├── sim2sim/         # Viser 3D viewers, Isaac Sim2Sim validator, TorchScript & W&B tools
│   │   └── tasks/           # Motion tracking, motion prior, and carry-box task definitions
│   ├── humanoid_retarget/
│   │   ├── configs/motions/ # Motion manifests, showcase playlists, and velocity allowlists
│   │   └── datasets/        # Provenance manifests & local mocap storage (LAFAN1, CMU)
│   └── robots/
│       └── humanoid/        # Shared MJCF, meshes, and control profiles (humanoid_light, h1, g1)
├── scripts/                 # Dataset download, Mink/DAQP IK retargeting, and screening scripts
├── test/                    # Unified unit & integration test suite
├── docs/                    # Guides and dated experiment reports (docs/exp/YYYY-MM-DD/)
└── third_party/             # Git submodules (BFM-Zero, FlashSAC, loco-mujoco, mjlab_research, supermimic)
```

For details on each learning algorithm (`SuperMimic`, `BFM-Zero`, `FlashSAC`, `AMP`, `ADD`), see [`src/humanoid_rl/algorithm/README.md`](src/humanoid_rl/algorithm/README.md).

---

## Setup & SuperMimic Training (`uv run`)

```bash
# 1. Initialize submodules and base CLI environment
git submodule update --init --recursive
uv sync

# 2. Install the locked Humanoid-Light SuperMimic environment (.venv_supermimic_light)
uv run humanoid setup --method supermimic --task Humanoid-Light-Motion-Tracking

# 3. List experiments and train Humanoid-Light SuperMimic
uv run humanoid list
uv run humanoid train --experiment light_supermimic_s2r_full --runtime local

# 4. Interactive Viser 3D playback
.venv_supermimic_light/bin/python src/humanoid_rl/sim2sim/play_light_supermimic_viser.py --port 8080
```

---

## Motion Datasets (LAFAN1 & CMU) & Humanoid-Light Retargeting

Raw mocap archives (`.zip`, `.bvh`, `.amc`, `.npz`, `.pkl`) and generated `outputs/` are excluded from Git; provenance manifests and SHA-256 checksums are tracked under [`src/humanoid_retarget/datasets/`](src/humanoid_retarget/datasets/).

### 1. Download Original LAFAN1 & CMU Datasets

- **LAFAN1 (Ubisoft La Forge Animation Dataset — 77 BVH sequences, ~30 Hz)**
  - Official repository: <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>
  - Direct archive (`lafan1.zip`, 144,051,503 bytes): <https://github.com/ubisoft/ubisoft-laforge-animation-dataset/raw/master/lafan1/lafan1.zip>
  - Download & extract to `src/humanoid_retarget/datasets/lafan1/raw/`:
    ```bash
    uv run python scripts/prepare_h1_data.py
    ```

- **CMU Graphics Lab Motion Capture Database (2,548 ASF/AMC sequences, 60/120 Hz)**
  - Official database: <http://mocap.cs.cmu.edu/>
  - Bulk archive (`allasfamc.zip`, 1,083,549,385 bytes, SHA-256 `4b0fc3a7ba3cbb2aa23948b4cefe7efa45edf864b2b5e00713659a4f2144a591`): <http://mocap.cs.cmu.edu/allasfamc.zip>
  - Download & unpack with verified C3D frame rates:
    ```bash
    mkdir -p src/humanoid_retarget/datasets/cmu/full/archives
    curl -L http://mocap.cs.cmu.edu/allasfamc.zip \
      -o src/humanoid_retarget/datasets/cmu/full/archives/allasfamc.zip
    uv run python scripts/prepare_cmu_full.py \
      --archive src/humanoid_retarget/datasets/cmu/full/archives/allasfamc.zip \
      --index src/humanoid_retarget/datasets/cmu/full/metadata/official_index.json
    ```

### 2. Retarget to 26-DoF Humanoid-Light (50 Hz)

1. Stage the `Humanoid-Light` assets and generate the calibrated Mink/DAQP IK profile:
   ```bash
   mkdir -p outputs/humanoid_light_conversion
   cp -r src/robots/humanoid/humanoid_light/assets/* outputs/humanoid_light_conversion/
   uv run python scripts/retarget_humanoid_light_lafan.py --no-render
   ```
2. Retarget all LAFAN1 and CMU sequences to 50 Hz `qpos`/`qvel` archives:
   ```bash
   uv run python scripts/retarget_humanoid_light_lafan_full.py --workers 6
   uv run python scripts/summarize_humanoid_light_lafan_full.py
   uv run python scripts/retarget_humanoid_light_cmu_full.py --workers 12
   uv run python scripts/summarize_humanoid_light_cmu_full.py
   ```
3. Screen joint limits, velocity limits, floor clearance, and self-collisions to build `outputs/humanoid_light_bfm/manifest.json`:
   ```bash
   OPENBLAS_NUM_THREADS=1 uv run python scripts/prepare_humanoid_light_bfm.py --workers 4
   ```

---

## Tests

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:test uv run --no-sync python -m unittest discover -s test -v
```
