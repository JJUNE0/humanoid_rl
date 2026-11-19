# Experiment Log: Humanoid Light SuperMimic Soft-Footstep BPTT Running Fine-Tuning (`run1`, `run2`, `sprint1`) & Isaac Sim 400Hz Impact Force Reduction

- **Date**: 2026-11-18
- **Execution Domain**: `TRAINING_SIM` & `VALIDATION_SIM`
- **Policy Checkpoint / Hash**:
  - Pretrained Base (`step_65536000`): `runs/light_supermimic_lafan_all/rtx3090_lafan_all_stage2_20261118/training/checkpoints/step_65536000/last.pt` (`560f737d841f674b8cdf501b84caecc5a426feff1fe8af2a96e114c5e0c653f2`)
  - Fine-Tuned Checkpoint (`step_10240000`): `runs/light_supermimic_run_soft/rtx3090_run_soft_ft_20261118/training/checkpoints/step_10240000/last.pt` (`d70b55eb9e285c1919296a4a3ac383a8eb8d94fa0d06d68c72592db07f65eb4c`)
  - Exported TorchScript (`run_soft_step_10240000_jit.pt`): `outputs/sim2sim/run_soft_step_10240000_jit.pt` (`ff267ae1a7c9a2d96703d35b0fd4e0c10601a22055bad0fa1fd154599b8bf990`)
- **W&B Run URL**: `https://forge.coreweave.com/wandb/bjy6614-postech/humanoid-rl/runs/ydjfe9nb`
- **Evidence Status**: `PASS`

---

## 1. Objective & BPTT Reward Design
- **Goal**: Reduce transient foot-strike ground reaction forces (`400 Hz` impact peak `N` and loading rate `dF/dt`) during running and sprinting on `Humanoid-Light` (`36.14 kg`, `1.0 × BW = 354.5 N`) via analytic 32-step World-Model BPTT without adding contact-force sensors to the World Model.
- **Exact 3-Point Sole Kinematics (`src/humanoid_rl/tasks/motion_tracking/light_supermimic.py`)**:
  - Verified exact local coordinates of heel `(0.0, -0.043, -0.090)`, mid-sole `(0.0, -0.043, 0.0)`, and toe `(0.0, -0.043, +0.090)` in `left/right_ankle_roll_link` frames against `humanoid_light.xml` mesh vertices.
  - Added identical differentiable penalty in both `SoftFootstepGenCoordRepresentation.compute_wm_reward_from_reference` (`src/humanoid_rl/algorithm/supermimic/train.py`) and `mjlab` simulator `cfg.rewards["soft_footstep"]`:
    1. **Pre-Landing Vertical Impact (`impact_weight = 2.0`, `v_free = 0.15 m/s`, `impact_height = 0.050 m`, `impact_tau = 0.012 m`)**: Smoothly decelerates downward sole velocity (`v_f + ω_f × r_s`) in the final `5 cm` before touchdown while leaving aerial swing unpenalized.
    2. **Ground Contact Slip (`slip_weight = 0.15`, `slip_height = 0.010 m`, `slip_tau = 0.004 m`)**: Suppresses horizontal scuffing right at ground contact (`<= 1 cm`) without penalizing low-clearance forward swing during running.
    3. **Coupled Leg Smoothness (`leg_action_rate_weight = 0.10`)**: Damps 50 Hz PD target jumps on the 12 leg joints.
- **Dataset (`configs/motions/light_supermimic_run.json`)**:
  - `train`: `run1_subject2`, `run2_subject1`, `run2_subject4`, `sprint1_subject2`, `sprint1_subject4` (`217` intervals, `60,157` frames = `20.1 min`).
  - `validation`: Held-out unseen `run1_subject5` (`45` intervals, `52` clips, `11,016` frames = `3.7 min`).
- **Video Logging Note**: Headless CUDA-graph local CLI run prioritized throughput; full scalar curves and Sim-to-Sim evaluation summary were synced to W&B (`ydjfe9nb`), and live visualization is served on Viser (`8080` MuJoCo / `8081` Isaac USD).

---

## 2. Fine-Tuning Convergence (`TRAINING_SIM`)
- **Throughput & Wall-Clock**: **`7,469.1 env_steps/s` steady** (`1,582.1 s` = `26m 22s` for `10,240,000` steps on `RTX 3090`).
- **Reward & Episode Survival**:
  - `Train/mean_reward`: `-1.84` (initial step before footstrike adaptation) → **`+23.92 – +26.25`**
  - `Train/mean_episode_length`: `20.0` → **`500.0 / 500.0` steps** (`100%` full-horizon running survival in `mjlab`)
  - `Loss/state`: **`0.01406`**, `Loss/joint_pos`: **`8.45e-5`**, `Policy/reward`: **`9.05`**

---

## 3. Zero-Shot Isaac Sim 5.1 (PhysX USD) Before vs After Comparison (`VALIDATION_SIM`)

| Running Split | Metric | Stage-2 Baseline (`65.5M`) | Soft-Footstep Fine-Tuned (`+10.24M`) | Change |
|---|---|---:|---:|---:|
| **Unseen `run1_subject5` (`val`, 52 clips)** | **`400 Hz` Impact Peak (`P99`)** | `3,672.6 N` (`10.36× BW`) | **`2,809.4 N` (`7.92× BW`)** | **`-23.5%` (`-863 N`)** |
| | **`400 Hz` Max Spike** | `10,094.9 N` | **`7,942.0 N`** | **`-21.3%` (`-2,153 N`)** |
| | **`50 Hz` Step-Mean `P99`** | `1,185.8 N` | **`1,028.8 N`** | **`-13.2%`** |
| | **Impact Loading Rate `dF/dt` (`P99`)** | `934.3 kN/s` | **`682.5 kN/s`** | **`-27.0%`** |
| | `50 Hz` Stance Mean (1×BW check) | `365.6 N` | `374.3 N` | `+2.4%` (normal momentum) |
| | Clip Survival Ratio / Completion | `99.1%` / `51/52` (`98.1%`) | **`96.6%`** / `47/52` (`90.4%`) | `-2.5%p` survival |
| **`run1_subject2` (`train`)** | **`400 Hz` Impact Peak (`P99`)** | `3,488.5 N` (`9.84× BW`) | **`2,729.7 N` (`7.70× BW`)** | **`-21.8%` (`-759 N`)** |
| | **`400 Hz` Max Spike** | `11,128.9 N` | **`6,354.8 N`** | **`-42.9%` (`-4,774 N`)** |
| | **Impact Loading Rate `dF/dt` (`P99`)** | `830.3 kN/s` | **`710.0 kN/s`** | **`-14.5%`** |
| **`run2_subject1+4` (`train`)** | **`400 Hz` Impact Peak (`P99`)** | `3,797.3 N` (`10.71× BW`) | **`2,854.0 N` (`8.05× BW`)** | **`-24.8%` (`-943 N`)** |
| | **`400 Hz` Max Spike** | `13,202.4 N` | **`9,521.7 N`** | **`-27.9%` (`-3,681 N`)** |
| | **Impact Loading Rate `dF/dt` (`P99`)** | `973.7 kN/s` | **`693.7 kN/s`** | **`-28.8%`** |
| **`sprint1_subject2+4` (`train`)** | **`400 Hz` Impact Peak (`P99`)** | `2,295.9 N` (`6.48× BW`) | **`1,940.8 N` (`5.47× BW`)** | **`-15.5%` (`-355 N`)** |
| | **Impact Loading Rate `dF/dt` (`P99`)** | `562.8 kN/s` | **`476.7 kN/s`** | **`-15.3%`** |

- **Saved Reports**:
  - `outputs/sim2sim/isaac_run_soft_val_eval.json`
  - `outputs/sim2sim/isaac_run_soft_contact_forces_val.json`
  - `outputs/sim2sim/isaac_run_soft_contact_forces_train.json`
  - `outputs/sim2sim/isaac_stage2_baseline_run_contact_forces_train.json`
