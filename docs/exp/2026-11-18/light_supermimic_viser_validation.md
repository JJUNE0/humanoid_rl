# Experiment Log: Humanoid Light SuperMimic Viser Playback Validation

- **Date**: 2026-11-18
- **Execution Domain**: `VALIDATION_SIM`
- **Policy Checkpoint / Hash**: `runs/light_supermimic_t4_long/free_t4_fp16_20261117/training/checkpoints/step_2050048/last.pt` (`5ae45294e9254994f9cecbe1973c8a2837b643317a6e6f8c55a7f745a22b02a3`)
- **Evidence Status**: `PASS`

---

## 1. Experiment Objective
- Verify local checkpoint loading, motion manifest/cache remapping from Colab paths, `LightDynamics` coupled-PD world-model rollout, and interactive `Viser` visualization for the trained `Humanoid-Light-Motion-Tracking` (`supermimic`, 2,050,048 env steps) policy via `test/play_light_supermimic_viser.py`.

## 2. Configuration & Setup
- **Environment / Simulator**: `Humanoid-Light-Motion-Tracking` (`mjlab` / `mujoco_warp` on `NVIDIA GeForce RTX 3090`)
- **Policy Rate / Decimation**: 50 Hz (`physics_dt = 0.0025`, `decimation = 8`)
- **Action Scale & Gain**: `JointRangePositionActionCfg` + `LightDynamics` (`joint_range_targets`, `control_hash = 13ab7cabefc5506362aad0066d3632c9bf02cc8cc41d05ca4051e32b610cd250`)
- **Packed Motion Dataset**: `93541cdc3f4473d6ca4cb95fa8ecdfa9fe8a7facfda2540d74019ba2d87566d5` (140 clips, 49,747 frames; exact fingerprint match with checkpoint)

## 3. Results & Log Evidence
- **Key Metrics**:
  - 100-step deterministic headless rollout (`--headless --steps 100`): `mean_step_reward = 0.0428`, `resets = 1`, zero non-finite actions / GT states / WM predictions.
  - Unit test suite (`tests/test_supermimic_light.py`): 2/2 passed in `0.277s`.
  - Live `Viser` HTTP/WebSocket server (`--port 8080`): listening on `http://localhost:8080` (`HTTP 200 OK`), serving ground-truth physics robot + closed-loop World Model ghost + reference motion ghost + 140-clip selector.
- **Observed Behavior / Findings**:
  - Colab `/content/...` motion root/manifest/cache paths in `last.pt` cleanly remap to local repository paths with bit-exact packed motion fingerprint verification.

## 4. Conclusions & Next Steps
- **Summary of results**: Local `Viser` playback for `Humanoid-Light-Motion-Tracking` SuperMimic checkpoints is verified (`PASS`) in `/test/play_light_supermimic_viser.py`.
- **Action items for next iteration**: After interactive inspection in `Viser`, optionally promote the loader into `src/humanoid_rl/algorithm/supermimic/play.py` and `src/humanoid_rl/play.py` if unified CLI playback is desired.
