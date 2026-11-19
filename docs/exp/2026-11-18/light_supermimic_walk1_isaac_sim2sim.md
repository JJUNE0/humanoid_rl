# Experiment Log: Humanoid Light SuperMimic `walk1` Training & MuJoCo-to-Isaac USD Sim-to-Sim Validation

- **Date**: 2026-11-18
- **Execution Domain**: `TRAINING_SIM` & `VALIDATION_SIM`
- **Policy Checkpoint / Hash**:
  - `last.pt` (`step_8192000`): `runs/light_supermimic_walk1_t4/t4_walk1_fixed_20261118/training/checkpoints/step_8192000/last.pt` (`9c5f014de73b9e02370ed6a9084e89030e6197621d3565691d04d82c0daa17f6`)
  - Exported TorchScript (`walk1_step_8192000_jit.pt`): `outputs/sim2sim/walk1_step_8192000_jit.pt` (`2f450dfad17f987bbfde0e5d84023ee6c5ef1d1182587586514c3a4cfef6664e`)
- **W&B Run URL**: `https://forge.coreweave.com/wandb/bjy6614-postech/humanoid-rl/runs/3u7cyhlj`
- **Evidence Status**: `PASS` (verified both quantitatively under strict 3D fall detection and visually via rendered MP4 + live Isaac-to-Viser streaming)

---

## 1. Experiment Objective
1. Resolve the three SuperMimic bottlenecks identified in `light_supermimic_failure_diagnosis.md`:
   - Replace hard-clamped `JointRangePositionAction` (`clamp(-1, 1)`) with unclamped couple-aware `JointPositionActionCfg(scale=0.25 * effort_limit / effective_kp)` (`action_mode="coupled_scale"`).
   - Enable upstream `ReturnSampler` (`sampling_mode="return"`) and restrict tracked bodies to the 15 structural links (`key_bodies_only=true`).
   - Scale batches (`WM: 2×1024`, `Policy: 1×1024`) and network width (`1024` hidden) on `walk1_subject1` (`13,041` frames, 29 chunks).
2. Export the trained `step_8192000` policy and validate **zero-shot Sim-to-Sim transfer** onto the original USD (`humanoid_light_rev_1_0_12_45_merge_joints.usd`) with `HUMANOID_LIGHT_CFG` (`DelayedPDActuatorCfg` + `CoupledDelayedPDActuatorCfg`) in **Isaac Sim 5.1 / IsaacLab v2.3.0**, plus browser-based **Viser** live inspection.

## 2. Critical Sim-to-Sim GUI Bug Caught & Fixed (`sim.step(render=True)`)
- **Symptom**: Initial headless `--all-clips` runs (`render=False`) walked 500 steps cleanly, whereas launching interactive GUI (`not args.headless`) or `--video` caused the USD robot to fall at step 41 (`0.82 s`).
- **Root Cause**: In Isaac Sim 5.1 (`SimulationContext` with `physics_dt = 0.0025 s`, `render_interval = 8`, `rendering_dt = 0.02 s`), calling `sim.step(render=True)` on the 8th substep invokes `self._app.update()` while `/app/player/playSimulations=True`, advancing PhysX by an entire `rendering_dt` (`0.020 s` = **8 extra substeps at once without 400 Hz PD damping updates**). That stretched each policy step to `7 × 0.0025 + 0.020 = 0.0375 s` (`26.7 Hz` instead of `50 Hz`) and destabilized the coupled ankles.
- **Fix**: Aligned `test/validate_light_supermimic_isaac.py` and `test/play_light_supermimic_isaac_viser.py` with `ManagerBasedRLEnv.step()`: always step physics via `sim.step(render=False)` across all 8 substeps and invoke `sim.render()` separately (`playSimulations=False` during Hydra render). Also upgraded fall detection to check 3D projected-gravity norm error (`> 0.70`), pelvis height error (`> 0.25 m`), and absolute pelvis/torso/head/knee ground collapse thresholds.

## 3. Results & Direct Evidence

### 3.1 Training Convergence (`mjlab` MuJoCo Warp — `TRAINING_SIM`)
- **`Train/mean_episode_length`**: `46.2` (step 696k) → `167.7` (step 1.70M) → **`500.0 / 500.0`** (from step 3.62M through final step 8.192M; 100% survival over 10.0s episodes).
- **`Train/mean_reward` (10s return)**: `1.41` → **`39.21`** (`3.92 / 5.0` per second = `78.4%` of theoretical ceiling).
- **`Policy/reward` (imagined return)**: `2.83` → **`12.49`**.
- **World Model `Loss/state`**: `0.065` → **`0.0057 – 0.0072`** (`Loss/joint_pos = 3.95e-5 rad²` ≈ `0.36°` RMS error).

### 3.2 Zero-Shot MuJoCo → Isaac Sim 5.1 (PhysX USD) Validation (`VALIDATION_SIM`)
1. **Visual Rollout Verification (`outputs/sim2sim/isaac_walk1_clip0_fixed.mp4`, `outputs/sim2sim/isaac_walk1_clip0_fixed.json`)**:
   - Clip 0 (`lafan/walk1_subject1/0-501`, 250 steps = `5.0 s` with RTX camera rendering active): **`250 / 250` steps survived (`fell = false`)**, **`9.56°` 12-leg RMSE**, **`2.98 cm` pelvis XY error**, **`1.18 cm` pelvis Z error**. Visually verified upright balanced walking across the ground plane.
2. **Full `train` Split (`walk1_subject1`, 29 clips — `outputs/sim2sim/isaac_walk1_step_8192000_train_strict.json`)**:
   - **Full Clip Completion Rate**: **`24 / 29` clips (`82.76%`)**
   - **Mean Clip Survival Ratio**: **`92.19%`** (`mean_survived_steps = 409.24`)
   - **Mean 12-Leg Joint RMSE**: **`10.89°`** (`5.8° – 8.6°` on straight walking clips; `30.05°` all-26-joint RMSE due to unconstrained `±180°` axial wrist twist)
   - **Mean Pelvis Position Error**: **`0.2469 m` XY** over 10s open-loop trajectory, **`0.0133 m` (`1.33 cm`) Z height**.
3. **Zero-Shot Unseen Subject (`validation` Split `walk1_subject5`, 31 clips — `outputs/sim2sim/isaac_walk1_step_8192000_val_strict.json`)**:
   - **Full Clip Completion Rate**: **`12 / 31` clips (`38.71%`)**
   - **Mean Clip Survival Ratio**: **`64.13%`** (`mean_survived_steps = 240.90`)
   - **Mean 12-Leg Joint RMSE**: **`16.00°`**, **Mean Pelvis Height Error**: **`0.0194 m` (`1.94 cm`)**.
4. **Live Isaac Sim PhysX → Viser Streaming (`test/play_light_supermimic_isaac_viser.py`)**:
   - Streams headless Isaac Sim 5.1 USD PhysX states + foot contact forces over localhost TCP into Viser (`http://localhost:8081`) with green reference ghost overlay and clip controls, avoiding Omniverse GUI overhead.

## 4. Conclusions & Next Steps
- **Summary**: Both the SuperMimic training bottleneck (`coupled_scale` + `ReturnSampler`) and the Isaac Sim rendering-step bug (`sim.step(render=False)` + `sim.render()`) are resolved and visually verified.
- **Unresolved Risks / Next Steps**:
  1. Sharp turn transitions (5 of 29 `subject1` clips, and unseen `subject5` gait variations) still experience occasional Sim-to-Sim foot-contact timing slips due to convex-hull vs capsule/mesh contact differences and single-subject training diversity.
  2. Train multi-subject / multi-skill locomotion with `actor_obs="proprio"` (`history_length > 1`) and domain randomization before staged Sim-to-Real evaluation.
