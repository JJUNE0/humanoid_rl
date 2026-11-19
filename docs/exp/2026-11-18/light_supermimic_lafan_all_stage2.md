# Experiment Log: Humanoid Light SuperMimic Full-LAFAN1 (`64` Sequences) Stage-2 Local RTX 3090 Training & Multi-Skill Isaac USD Sim-to-Sim Evaluation

- **Date**: 2026-11-18
- **Execution Domain**: `TRAINING_SIM` & `VALIDATION_SIM`
- **Policy Checkpoint / Hash**:
  - `last.pt` (`step_65536000`): `runs/light_supermimic_lafan_all/rtx3090_lafan_all_stage2_20261118/training/checkpoints/step_65536000/last.pt` (`560f737d841f674b8cdf501b84caecc5a426feff1fe8af2a96e114c5e0c653f2`)
  - Exported TorchScript (`lafan_all_step_65536000_jit.pt`): `outputs/sim2sim/lafan_all_step_65536000_jit.pt` (`8f25121a9892ab8e4d27f419a4c7d24d9909273d71ebaed54364863966389f9b`)
- **W&B Run URL**: `https://forge.coreweave.com/wandb/bjy6614-postech/humanoid-rl/runs/mat09vfv`
- **Evidence Status**: `PASS`

---

## 1. Objective & Setup
- **Goal**: Resume full-LAFAN1 (`64` sequences, `1,888` train clips, `668,208` frames = `3.71 h`) `Humanoid-Light` SuperMimic training on local `NVIDIA GeForce RTX 3090` (`24 GB`) from Stage-1 (`step_5632000`) to **`65,536,000` steps**, and validate zero-shot Sim-to-Sim transfer in **Isaac Sim 5.1 / IsaacLab v2.3.0 (PhysX USD)** across both `walk1` and the entire held-out unseen `subject5` multi-skill suite (`434` clips).
- **Throughput-Tuned Configuration (`configs/experiments/light_supermimic_lafan_all.json`)**:
  - Selected from grid benchmark (`outputs/sim2sim/bench_3090_grid/summary.json`): `num_envs = 2048`, `wm.batch_size = 2048` (`num_batches = 2`, `horizon = 8`), `policy.batch_size = 2048` (`num_batches = 1`, `horizon = 32`), `updates_per_step = 1` (preserving exact `16:1` sample-to-env UTD), `compile = true`, `cuda_graphs = true`, `muon_dtype = "bfloat16"`.
  - Networks: Actor `[1024, 1024, 512, 256]`, Dynamics `[1024, 1024, 1024]`, Critic `[512, 256]`, `action_mode = "coupled_scale"` (`scale = 0.25`), `sampling_mode = "return"`, `key_bodies_only = true`.
  - Video logging note: Headless local CLI run prioritized throughput (`cuda_graphs=true`); full scalar history (`1,170` log intervals) and Isaac Sim-to-Sim summary were uploaded to W&B (`mat09vfv`), and live interactive visualization is served on Viser (`8080` MuJoCo / `8081` Isaac USD).

---

## 2. Stage-2 Training Results (`mjlab` MuJoCo Warp — `TRAINING_SIM`)
- **Throughput & Wall-Clock**: **`7,203.14 env_steps/s` steady** (`8.6×` faster than Colab T4 `835.68 env_steps/s`), completing `59,904,000` additional transitions (`5.63M -> 65.54M`) in **`8,332.5 s` (`2h 18m 52s`)** with `12,214 MiB` free VRAM (`1,021.7 MiB` peak PyTorch tensor allocation).
- **World Model Convergence**:
  - `Loss/state`: `0.0548` (Stage-1 end) → **`0.02618`** (`-52.2%`)
  - `Loss/joint_pos`: `3.75e-4` → **`1.79e-4 rad²`** (`~0.77°` 1-step RMS joint prediction error across all 64 diverse LAFAN1 sequences)
  - `Loss/joint_vel`: `0.46` → **`0.2086`**
  - `Loss/root_pos`: **`4.59e-6`**, `Loss/root_ori`: **`2.32e-5`**
- **Policy Optimization**:
  - `Policy/reward` (32-step imagined horizon): `7.59` (Stage-1) → **`10.19 – 10.35`**
  - `Train/mean_reward`: **`12.58 – 14.44`**, `Train/mean_episode_length`: **`198.7 – 218.2` steps** under return-prioritized hard-clip sampling (`ReturnSampler`).

---

## 3. Zero-Shot Isaac Sim 5.1 (PhysX USD) Sim-to-Sim Evaluation (`VALIDATION_SIM`)

### 3.1 `walk1` Direct Comparison (Stage-1 `5.6M` vs Stage-2 `65.5M`)

| Split | Policy Checkpoint | Completion Rate | Mean Survival Ratio | 12-Leg Joint RMSE | Pelvis Height Error | Report File |
|---|---|---:|---:|---:|---:|---|
| `walk1_subject1` (`29` clips) | Stage-1 (`5.63M`) | `29 / 29` (`100.0%`) | `100.0%` | `15.74°` | `2.34 cm` | `outputs/sim2sim/isaac_lafan_all_stage1_walk1_eval.json` |
| `walk1_subject1` (`29` clips) | **Stage-2 (`65.54M`)** | **`29 / 29` (`100.0%`)** | **`100.0%`** | **`8.48°`** (`-46%`) | **`1.08 cm`** (`-54%`) | `outputs/sim2sim/isaac_lafan_all_stage2_walk1_eval.json` |
| Unseen `walk1_subject5` (`31` clips) | Stage-1 (`5.63M`) | `30 / 31` (`96.8%`) | `97.0%` | `17.46°` | `2.45 cm` | `outputs/sim2sim/isaac_lafan_all_stage1_walk1_val_eval.json` |
| Unseen `walk1_subject5` (`31` clips) | **Stage-2 (`65.54M`)** | **`30 / 31` (`96.8%`)** | **`97.0%`** | **`11.89°`** (`-32%`) | **`1.46 cm`** (`-40%`) | `outputs/sim2sim/isaac_lafan_all_stage2_walk1_val_eval.json` |

### 3.2 Full Held-Out Unseen `subject5` Multi-Skill Suite (`434` Clips — `outputs/sim2sim/isaac_lafan_all_stage2_full_val_eval.json`)
- **Overall**: **`309 / 434` clips completed (`71.20%`)**, **`80.40%` mean survival ratio**, **`15.60°` mean 12-leg RMSE** zero-shot in Isaac Sim PhysX USD.
- **Per-Sequence Breakdown (`subject5`, completely unseen during training)**:
  - `jumps1_subject5`: **`26 / 26` (`100.0%`)** | survival `100.0%` | leg RMSE `14.14°`
  - `run1_subject5`: **`51 / 52` (`98.1%`)** | survival `99.1%` | leg RMSE `14.55°`
  - `dance2_subject5`: **`26 / 27` (`96.3%`)** | survival `97.5%` | leg RMSE `12.00°`
  - `fight1_subject5`: **`45 / 47` (`95.7%`)** | survival `97.7%` | leg RMSE `15.30°`
  - `walk1_subject5`: **`29 / 31` (`93.5%`)** (`30/31 = 96.8%` in 31-env eval) | survival `94.5%` | leg RMSE `11.98°`
  - `aiming2_subject5`: **`39 / 47` (`83.0%`)** | survival `89.2%` | leg RMSE `15.11°`
  - `pushAndStumble1_subject5`: **`32 / 42` (`76.2%`)** | survival `82.5%` | leg RMSE `16.12°`
  - `walk3_subject5`: **`19 / 28` (`67.9%`)** | survival `77.9%` | leg RMSE `20.04°`
  - `ground1_subject5`: **`11 / 17` (`64.7%`)** | survival `89.9%` | leg RMSE `15.21°`
  - `obstacles2_subject5`: `19 / 37` (`51.4%`), `obstacles1_subject5`: `4 / 22` (`18.2%`), `obstacles6_subject5`: `2 / 27` (`7.4%`) — expected failure mode on flat ground (`terrain_type="plane"`), because LAFAN1 `obstacles*` clips step onto elevated physical boxes/platforms (`+0.3m` to `+0.6m` in mid-air) that are not spawned in the flat-plane scene.
  - `fallAndGetUp1_subject5`: `6 / 31` (`19.4%` full completion, `46.7%` survival) — full-body ground-contact get-up transitions remain sensitive to capsule-vs-USD-mesh self/ground collision differences and represent the primary benefit target for longer (`200M+`) training + domain randomization.

---

## 4. Next Steps
1. **Soft-Footstep BPTT Fine-Tuning**: Fine-tune from `step_65536000` with smooth height-gated downward foot velocity (`Impact`) + horizontal `Foot Slip` penalties in `tracking_reward.py` (synchronized with `mjlab` env reward) and compare peak ground reaction forces (`N`) in Isaac Sim USD.
2. **Optional Full-Horizon Scaling (`65.5M -> 200M+`)**: Extend training with contact/PD domain randomization if higher get-up (`fallAndGetUp1`) recovery in USD mesh contact is desired.
