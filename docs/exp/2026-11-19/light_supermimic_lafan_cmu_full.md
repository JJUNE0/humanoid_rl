# Experiment Log: Humanoid Light SuperMimic Full LAFAN1 + CMU (`12.98h`, `2,325` Files, `196.6M` Steps) Generalist Foundation Training & Isaac Sim USD Validation

- **Date**: 2026-11-19
- **Execution Domain**: `TRAINING_SIM` & `VALIDATION_SIM`
- **Policy Checkpoint / Hash**:
  - Pretrained Warm-Start (`step_65536000`): `runs/light_supermimic_lafan_all/rtx3090_lafan_all_stage2_20261118/training/checkpoints/step_65536000/last.pt` (`560f737d841f674b8cdf501b84caecc5a426feff1fe8af2a96e114c5e0c653f2`)
  - Final Foundation Checkpoint (`step_196608000`, `262.1M` cumulative steps): `runs/light_supermimic_lafan_cmu_full/rtx3090_lafan_cmu_full_20261119/training/checkpoints/step_196608000/last.pt` (`696d28c8a09fb362b081aaae467dfe6c9decd64e99c6641b2055f65628c87af3`)
  - Exported TorchScript (`lafan_cmu_full_step_196608000_jit.pt`): `outputs/sim2sim/lafan_cmu_full_step_196608000_jit.pt` (`8ce0e9656d9ea896c994f4c9be14264842918fe78a3641fa847497442a8e342d`)
- **W&B Run URL**: `https://forge.coreweave.com/wandb/bjy6614-postech/humanoid-rl/runs/2t9ongbv`
- **Evidence Status**: `PASS`

---

## 1. Objective & Setup
- **Dataset (`outputs/humanoid_light_bfm/manifest.json`)**:
  - **`train` split**: `64` LAFAN1 files (`901` records, `668,209` frames = `3.71 h`) + **`2,261` CMU files** (`2,636` records, `1,668,774` frames = `9.27 h`) → **`2,325` files, `6,707` packed clips, `2,336,980` frames (`12.98 hours`)**.
  - **`validation` split**: Subject-disjoint held-out `13` LAFAN1 `subject5` files (`434` clips, `147,315` frames = `0.82 h`) + **`219` unseen CMU files** (`265` records, `624` clips, `241,427` frames = `1.34 h`).
- **Configuration (`configs/experiments/light_supermimic_lafan_cmu_full.json`)**:
  - Warm-started (`init_from`) from LAFAN Stage-2 (`65.54M` steps) and trained for **`+196,608,000` steps** (`262,144,000` cumulative transitions) on local `RTX 3090` (`num_envs = 2048`, `batch_size = 2048`, `compile = true`, `cuda_graphs = true`, `muon_dtype = "bfloat16"`, `sampling_mode = "return"`).
  - Video logging note: Headless CUDA-graph overnight run prioritized throughput; all `3,840` scalar intervals and CMU Sim-to-Sim metrics were uploaded to W&B (`2t9ongbv`), with live interactive playback served on Viser (`8080` MuJoCo / `8081` Isaac USD).

---

## 2. Training Convergence (`TRAINING_SIM`)
- **Throughput & Wall-Clock**: **`7,504.3 env_steps/s` steady**, completing `196,608,000` steps in **`26,304.4 s` (`7h 18m 24s`)** with `11,469 MiB` free VRAM (`1,459.1 MiB` peak PyTorch allocation).
- **World Model & Policy**:
  - `Loss/state`: `0.0382` (warm-start step `102k`) → **`0.01173`** (`-55%` lower than LAFAN-only Stage-2 `0.02618`, even across `3.5×` larger LAFAN+CMU dataset!)
  - `Loss/joint_pos`: **`7.96e-5 rad²`** (`~0.51°` 1-step RMS joint prediction error), `Loss/joint_vel`: **`0.1031`**
  - `Policy/reward` (32-step imagined horizon): `10.19` (Stage-2) → **`12.24 – 12.33`** (`+20%`)
  - `Train/mean_reward`: **`12.06 – 15.38`**, `Train/mean_episode_length`: **`157.7 – 200.9` steps** under return-prioritized hard-clip sampling across all `6,707` clips.

---

## 3. Zero-Shot Isaac Sim 5.1 (PhysX USD) Evaluation (`VALIDATION_SIM`)

### 3.1 Summary Across Held-Out CMU & LAFAN Suites

| Evaluation Suite | Clips | Full Clip Completion | Mean Survival Ratio | 12-Leg Joint RMSE | Root Height Error | Report File |
|---|---:|---:|---:|---:|---:|---|
| **Held-Out Unseen CMU (`val`)** | **`624`** | **`564 / 624` (`90.38%`)** | **`94.04%`** | **`9.25°`** | **`2.29 cm`** | `outputs/sim2sim/isaac_lafan_cmu_full_cmu_val_eval.json` |
| `walk1_subject1` (`train`) | `29` | **`29 / 29` (`100.00%`)** | **`100.00%`** | **`7.71°`** (vs `8.48°` Stage-2) | **`1.06 cm`** | `outputs/sim2sim/isaac_lafan_cmu_full_walk1_eval.json` |
| Unseen `walk1_subject5` (`val`) | `31` | **`30 / 31` (`96.77%`)** | **`98.63%`** | **`11.21°`** (vs `11.89°` Stage-2) | **`1.30 cm`** | `outputs/sim2sim/isaac_lafan_cmu_full_walk1_val_eval.json` |
| Unseen LAFAN `subject5` All (`val`) | `434` | **`311 / 434` (`71.66%`)** | **`80.28%`** | **`14.62°`** (vs `15.60°` Stage-2) | **`4.27 cm`** | `outputs/sim2sim/isaac_lafan_cmu_full_lafan_val_eval.json` |

### 3.2 Per-Skill Progression on Unseen LAFAN `subject5` (LAFAN Stage-2 `65.5M` → LAFAN+CMU `262.1M`)
- **`run1_subject5` (달리기)**: `51/52` (`98.1%`, `14.55°`) → **`52 / 52` (`100.0%` 완주, `13.76°`)**
- **`aiming2_subject5` (조준 이동)**: `39/47` (`83.0%`, `15.11°`) → **`44 / 47` (`93.6%`, `13.61°`)** (`+10.6%p`)
- **`dance2_subject5` (댄스)**: `26/27` (`96.3%`, `12.00°`) → **`26 / 27` (`96.3%`, `10.13°`)**
- **`jumps1_subject5` (점프)**: `26/26` (`100.0%`, `14.14°`) → **`25 / 26` (`96.2%`, `12.52°`)**
- **`fight1_subject5` (격투)**: `45/47` (`95.7%`, `15.30°`) → **`45 / 47` (`95.7%`, `13.81°`)**
- **`walk1_subject5` (보행)**: `29/31` (`93.5%`, `11.98°`) → **`30 / 31` (`96.8%`, `11.16°`)**
- **`pushAndStumble1_subject5` (밀침 복원)**: `32/42` (`76.2%`, `16.12°`) → **`32 / 42` (`76.2%`, `14.17°`)**
- **`fallAndGetUp1_subject5` (일어나기)**: `6/31` (`19.4%`, `20.71°`) → **`8 / 31` (`25.8%`, `19.03°`)** (`+6.4%p`)
- **Contact Force Baseline (`outputs/sim2sim/isaac_lafan_cmu_full_contact_forces_val.json`)**: Even prior to soft-contact fine-tuning, tighter kinematic tracking naturally reduced `400 Hz` P99 impact peaks across `walk1` (`2,531 → 2,345 N`), `jumps1` (`2,006 → 1,802 N`), `fight1` (`2,477 → 2,212 N`), and `ground1` (`hands 541 → 462 N, shoulders 1,061 → 660 N`), while `fallAndGetUp1` wrist spikes (`5,077 N`) and `ground1` head brushes (`16.9%`, `778 N`) confirm the clear value of whole-body 3-tier soft-contact fine-tuning on top of this generalist foundation checkpoint.
