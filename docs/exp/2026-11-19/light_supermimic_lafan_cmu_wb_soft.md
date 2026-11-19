# Whole-Body 3-Tier Soft-Contact BPTT Fine-Tuning (`light_supermimic_lafan_cmu_wb_soft`)

## 1. Metadata (`AGENTS.md` Required)

- **Date**: `2026-11-19`
- **Execution Domains**:
  - `TRAINING_SIM` (`MuJoCo Warp` 400 Hz physics / 50 Hz coupled-PD control, `rtx3090_lafan_cmu_wb_soft_20261119`, `+20,481,024` steps warm-started from `step_196608000` generalist base = `282.6M` cumulative transitions)
  - `VALIDATION_SIM` (`Isaac Sim 5.1 / IsaacLab v2.3.0 PhysX` zero-shot USD `400 Hz` whole-body contact force profiling on `434` unseen `LAFAN1 subject5` clips + zero-shot tracking evaluation on `624` unseen `CMU` validation clips)
- **Checkpoints & Hashes**:
  - Source Base Checkpoint (`step_196608000`): `runs/light_supermimic_lafan_cmu_full/rtx3090_lafan_cmu_full_20261119/training/checkpoints/step_196608000/last.pt` (`696d28c8a09fb362b081aaae467dfe6c9decd64e99c6641b2055f65628c87af3`)
  - Fine-Tuned Checkpoint (`step_20481024`): `runs/light_supermimic_lafan_cmu_wb_soft/rtx3090_lafan_cmu_wb_soft_20261119/training/checkpoints/step_20481024/last.pt`
    - **SHA256**: `4a46518ab1ba128e976cd6a8e5c0202d766fa3701d593b30f87afcf4cc446c6b`
  - Standalone TorchScript Policy: `outputs/sim2sim/lafan_cmu_wb_soft_step_20481024_jit.pt`
    - **SHA256**: `02e0326c5ebe5bfc7bfc4c8b97ea6a8a58fe6c92d94b7e310abc8f76076b1b60`
- **W&B Run URL**: `https://forge.coreweave.com/wandb/bjy6614-postech/humanoid-rl/runs/d4qec2pw`
- **Evidence Status**: `PASS`

---

## 2. Training Summary (`TRAINING_SIM`)

- **Configuration**: `configs/experiments/light_supermimic_lafan_cmu_wb_soft.json` (`num_envs=1536`, `wm.batch_size=1536`, `policy.batch_size=1536`, `policy.lr=1e-4`, `whole_body_contact` 21 surface collision spheres across 3 hardware safety tiers).
- **Throughput & Convergence**:
  - Completed `20,481,024` steps (`13,303` WM & BPTT updates) in `5,255.2 s` (`1h 27m`) alongside two live Viser 3D servers (`4,224.8` steady `env_steps/s`).
  - Final training metrics: `Loss/state = 0.01216` (best `0.00968`), `Policy/reward = 11.92`, `Train/mean_episode_length = 189.79` (peak `243.76`).

---

## 3. Zero-Shot Isaac Sim 5.1 USD Validation (`VALIDATION_SIM`)

### 3.1 Unseen CMU Validation (`624` Clips — `outputs/sim2sim/isaac_lafan_cmu_wb_soft_cmu_val_report.json`)

| Metric | Hard-Contact Base (`step_196608000`) | 3-Tier Soft-Contact (`step_20481024`) |
| :--- | :---: | :---: |
| Completed Clips | `564 / 624` (`90.38%`) | **`560 / 624` (`89.74%`)** |
| Mean Survival Ratio | `94.04%` | **`93.76%`** |
| Mean 12-DoF Leg Joint RMSE | `9.25°` | **`9.79°`** |
| Mean Root Height Error | `0.0218 m` | **`0.0231 m`** |

### 3.2 Unseen LAFAN1 `subject5` (`434` Clips) Whole-Body `400 Hz` Contact Force Comparison (`1 BW = 354.5 N`)

Source reports: `outputs/sim2sim/isaac_lafan_cmu_full_contact_forces_val.json` (Before) vs `outputs/sim2sim/isaac_lafan_cmu_wb_soft_contact_forces_val.json` (After).

| Category | 1-Foot `400 Hz` P99 Before -> After (`Δ%`) | `dF/dt` P99 (`kN/s`) Before -> After | Notable Non-Foot Contact Peak Changes (Alive Steps) |
| :--- | :---: | :---: | :--- |
| `fallAndGetUp1` | `2,321.2 N` -> **`1,622.0 N` (`-30.1%`)** | `519.4` -> **`405.4`** | **`head`: `6,253.7 N -> 0.0 N` (`-100%`)**, `hips_pelvis`: `11,136.9 N -> 4,765.2 N` (`-57.2%`), `hands_wrists`: `12,310.5 N -> 9,147.5 N` (`-25.7%`), `elbows`: `216.6 N -> 0.0 N` (`-100%`) |
| `fight1` | `2,212.3 N` -> **`1,908.1 N` (`-13.8%`)** | `538.8` -> **`440.0`** | **`hands_wrists`: `8,254.7 N -> 0.0 N` (`-100%`)**, **`shoulders`: `3,004.0 N -> 0.0 N` (`-100%`)** |
| `obstacles2` | `2,949.9 N` -> **`2,135.6 N` (`-27.6%`)** | `711.2` -> **`569.3`** | Clean bipedal contact (`<0.5%` non-foot) |
| `obstacles6` | `3,396.8 N` -> **`2,675.6 N` (`-21.2%`)** | `1,071.5` -> **`779.2`** | `shoulders`: `6,398.8 N -> 4,006.4 N` (`-37.4%`), `hands_wrists`: `11,740.0 N -> 10,978.6 N` (`-6.5%`) |
| `dance2` | `1,608.1 N` -> **`1,333.0 N` (`-17.1%`)** | `340.5` -> **`283.2`** | Clean bipedal contact (`<0.5%` non-foot) |
| `pushAndStumble1` | `2,412.3 N` -> **`2,018.5 N` (`-16.3%`)** | `612.6` -> **`478.1`** | `hands_wrists`: `17,546.2 N -> 13,856.6 N` (`-21.0%`) *(1 pre-termination stumble spike on `head` remains)* |
| `run1` | `3,565.7 N` -> **`3,036.5 N` (`-14.8%`)** | `951.2` -> **`735.0`** (`-22.7%`) | Clean bipedal contact (`<0.5%` non-foot) |
| `walk1` | `2,345.1 N` -> **`2,020.7 N` (`-13.8%`)** | `576.8` -> **`488.3`** (`-15.3%`) | Clean bipedal contact (`<0.5%` non-foot) |
| `aiming2` | `2,874.7 N` -> **`2,535.9 N` (`-11.8%`)** | `728.9` -> **`620.2`** | Clean bipedal contact (`<0.5%` non-foot) |
| `walk3` | `3,207.8 N` -> **`2,917.4 N` (`-9.1%`)** | `901.6` -> **`846.9`** | `hands_wrists`: `19,267.6 N -> 10,521.6 N` (`-45.4%`) |
| `ground1` | `542.3 N` -> `594.0 N` (`+9.5%`) | `144.8` -> `158.6` | `hands_wrists`: `5,814.6 N -> 4,140.7 N` (`-28.8%`), `shoulders`: `6,666.3 N -> 2,655.1 N` (`-60.2%`) |

---

## 4. Residual Risks

- Pre-fall uncontrolled stumble frames in `pushAndStumble1` and `ground1` can still produce single-substep upper-body spikes prior to episode termination; hardware deployment still requires an independent IMU tilt/damping safety supervisor.
