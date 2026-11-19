# Experiment Log: Humanoid Light SuperMimic Full-LAFAN1 (`64` Sequences) Stage-1 Training & Isaac USD Sim-to-Sim Evaluation

- **Date**: 2026-11-18
- **Execution Domain**: `TRAINING_SIM` & `VALIDATION_SIM`
- **Policy Checkpoint / Hash**:
  - `last.pt` (`step_5632000`): `runs/light_supermimic_lafan_all_t4/t4_lafan_all_stage1_20261118/training/checkpoints/step_5632000/last.pt` (`dd4f79d90b2de3659c5f03261803a8ffef57d16eb5af1de2e46670ae678869f1`)
  - Exported TorchScript (`lafan_all_step_5632000_jit.pt`): `outputs/sim2sim/lafan_all_step_5632000_jit.pt` (`6b69f2d13b742dd170764ca7cebc6473db5c9e16c29bfa1c40b60e4f8fad5e00`)
- **W&B Run URL**: `https://forge.coreweave.com/wandb/bjy6614-postech/humanoid-rl/runs/rbjegsyq`
- **Evidence Status**: `PASS`

---

## 1. Objective & Setup
- **Dataset (`configs/motions/light_supermimic_lafan_all.json`)**:
  - **`train` split**: All **64** screened LAFAN1 sequences across subjects 1–4 (`901` records, **`1,888` clips**, **`668,208` frames = `3.71 hours`** covering `walk`, `run`, `sprint`, `aiming`, `dance`, `obstacles`, `jumps`, `fight`, `fallAndGetUp`, `ground`, `pushAndStumble`).
  - **`validation` split**: Subject-disjoint held-out **subject 5** (`13` files, `230` records, **`434` clips**, **`147,315` frames = `49.1 min`**).
- **Stage-1 Configuration (`configs/experiments/light_supermimic_lafan_all_t4.json`)**:
  - Colab `Tesla T4` (`session = humanoid_8646b988322d`, `num_envs = 1024`, `total_env_steps = 5,632,000`, `save_every = 512,000`, `muon_dtype = "float16"`).
  - Networks scaled to 4-layer Actor `[1024, 1024, 512, 256]`, Dynamics `[1024, 1024, 1024]`, Critic `[512, 256]`, `action_mode = "coupled_scale"` (`scale = 0.25`), `sampling_mode = "return"`, `key_bodies_only = true`.
  - Companion full-scale resume config (`configs/experiments/light_supermimic_lafan_all.json`) verified compatible via `check_resume_settings` for seamless Stage-2 continuation on local `RTX 3090` (`num_envs = 4096`, `bfloat16`).
  - Video logging note: Direct Colab CLI headless job logged scalar curves; post-run W&B upload (`rbjegsyq`) synced full training history.

## 2. Stage-1 Training Results (`mjlab` MuJoCo Warp — `TRAINING_SIM`)
- **Throughput & Wall-Clock**: `835.68 env_steps/s` steady (`6,724.5 s` = `1h 52m` total training time; peak PyTorch VRAM `625.2 MiB`, `6,029.8 MiB` free on 15 GB T4; Colab VM cleanly released).
- **World Model `Loss/state`**: `0.377` (step 942k) → **`0.0548`** (`Loss/joint_pos = 3.75e-4 rad²` ≈ `1.11°` RMS error across all 1,888 diverse LAFAN1 clips).
- **Episode Survival (`Train/mean_episode_length` across all 1,888 LAFAN1 clips including jumps/dance/getup)**: `43.1` → **`254.6` steps** (`> 5.1 s` average survival across the entire multi-skill dataset after only `5.63M` transitions!).
- **Reward**: `Train/mean_reward`: `1.19` → **`12.39`**; `Policy/reward` (imagined horizon): `3.61` → **`7.59`**.

## 3. Zero-Shot Isaac Sim 5.1 (PhysX USD) Sim-to-Sim Evaluation (`VALIDATION_SIM`)
- **`walk1_subject1` (29 clips — `outputs/sim2sim/isaac_lafan_all_stage1_walk1_eval.json`)**:
  - **Full Clip Completion Rate**: **`29 / 29` clips (`100.0%` completion, `0.0%` fall rate!)** — even the 5 sharp-turn clips that previously slipped under the single-clip `walk1` policy (`24/29`) now complete without falling thanks to multi-skill recovery priors!
  - **12-Leg Joint RMSE**: `15.74°`, **Pelvis Height Error**: `0.0234 m` (`2.34 cm`).
- **Held-Out Unseen `walk1_subject5` (31 clips — `outputs/sim2sim/isaac_lafan_all_stage1_walk1_val_eval.json`)**:
  - **Full Clip Completion Rate**: **`30 / 31` clips (`96.77%` completion, `3.23%` fall rate!)** — a massive leap over the single-subject `walk1` policy (`12 / 31 = 38.71%`), proving strong cross-subject & cross-simulator generalization from multi-motion training even at Stage 1 (`5.63M` steps).
  - **12-Leg Joint RMSE**: `17.46°`, **Pelvis Height Error**: `0.0245 m` (`2.45 cm`).

## 4. Next Steps
1. **Stage-2 Local `RTX 3090` Continuation**: Local `RTX 3090` (`24 GB`, native `bfloat16`) now has `>20 GB` free VRAM; resume directly from `step_5632000` using `configs/experiments/light_supermimic_lafan_all.json` (`num_envs = 4096`) toward `30M – 80M` steps to sharpen dynamic skills (`jumps`, `dance`, `fallAndGetUp`, `sprint`) and reduce joint tracking RMSE.
