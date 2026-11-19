# Experiment Log: Humanoid Light SuperMimic Training Failure Root-Cause Diagnosis

- **Date**: 2026-11-18
- **Execution Domain**: `VALIDATION_SIM`
- **Policy Checkpoint / Hash**: `runs/light_supermimic_t4_long/free_t4_fp16_20261117/training/checkpoints/step_2050048/last.pt` (`5ae45294e9254994f9cecbe1973c8a2837b643317a6e6f8c55a7f745a22b02a3`)
- **Evidence Status**: `SUSPICIOUS` (Training execution succeeded with finite tensors, but policy quality fails whole-body tracking due to severe sample-budget, dataset-subset, and action-mapping bottlenecks)

---

## 1. Experiment Objective
- Diagnose why the `light_supermimic_t4_long` (`step_2050048`) policy exhibits poor motion tracking in `Viser`, separating:
  1. **SuperMimic paper (`arXiv:2610.28480`) & official upstream hyperparameter/scale gaps**
  2. **Code/architecture bottlenecks in the `Humanoid-Light` SuperMimic integration**
  3. **Retargeted motion quality & `Humanoid-Light` coupled-PD physical characteristics**

## 2. Configuration & Diagnostic Setup
- **Diagnostic Script**: `test/diagnose_light_supermimic_gap.py`
- **Evaluation Corpus**: All 140 training clips (`49,747` frames at 50 Hz) from `configs/motions/light_supermimic_t4_trial.json` evaluated in parallel (`num_envs = 140` on `RTX 3090`), starting deterministically from frame 0 of each clip up to 500 steps.
- **Baselines Compared**:
  1. `open_loop_ref_pd`: Feeding retargeted reference `motion.joint_pos[t+1]` directly into `Humanoid-Light` coupled PD without feedback.
  2. `trained_policy_step_2050048`: Closed-loop `MultiEncoderAE` actor trained in `free_t4_fp16_20261117`.

## 3. Results & Quantitative Evidence

### 3.1 Paper / Upstream (`third_party/supermimic`) vs `free_t4_fp16_20261117` Scale Comparison

| Parameter | SuperMimic Paper / Official Default | `free_t4_fp16_20261117` | Gap |
| --- | --- | --- | --- |
| `wm` batch per update | `10 × 2048` = **20,480** windows (`H=8`) | `2 × 256` = **512** windows (`H=8`) | **40× smaller** |
| `policy` batch per update | `1 × 4096` = **4,096** rollouts (`H=32`) | `1 × 128` = **128** rollouts (`H=32`) | **32× smaller** |
| Parallel rollout iterations | `10,000 – 250,000+` iterations (`41M – 1B+` env steps) | `2,050,048 / 4096` = **~500 iterations** | **20× – 500× fewer** |
| Total policy rollouts trained | `40M – 1B+` imagined rollouts | `8,081 × 128` = **1.03M** rollouts (= **252.5** upstream updates) | **Under-trained by >40×** |
| WM / Actor MLP hidden dims | `(2048, 2048, 2048)` / `(2048, 1024, 512, 256)` | `(512, 512, 512)` / `(512, 256, 128)` | **8× – 14× fewer params** |
| Reset sampling (`sampling_mode`) | `"return"` (`ReturnSampler` focusing on hard bins) | `"uniform"` | No failure curriculum |

### 3.2 140-Clip Rollout Comparison (`open_loop_ref_pd` vs `trained_policy_step_2050048`)

| Category | Clips | Open-Loop PD Fall Rate | Policy Fall Rate | Open-Loop PD Mean Steps | Policy Mean Steps | Open-Loop PD Reward (`/5.0`) | Policy Reward (`/5.0`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **All Clips** | 140 | 76.43% | 67.14% | 113.3 | 138.2 | **2.875** | **1.395** |
| `cmu` (climb/punch/dance/etc.) | 26 | 96.15% | 96.15% | 42.8 | 52.2 | 3.010 | 1.062 |
| `lafan_dance` (`dance2_subject1`) | 33 | 93.94% | 93.94% | 31.9 | 35.5 | 2.769 | 0.681 |
| `lafan_jumps` (`jumps1_subject2`) | 32 | 71.88% | 71.88% | 84.8 | 90.1 | 2.957 | 1.401 |
| `lafan_ground` (`ground2_subject2/3`) | 49 | 57.14% | 30.61% | 224.0 | 284.3 | 2.823 | 2.048 |

- **Action Clamping Saturation**: `7.86%` of active policy action dimensions exceed `|a| > 1.0` (`max |a| = 2.078`), where `joint_range_targets`'s `action.clamp(-1, 1)` produces **exact zero BPTT gradient** through `LightDynamics.proc`.
- **World Model 1-step Joint Velocity MSE**: `4.345 rad²/s²` (`~2.08 rad/s` RMS error per 20 ms step on policy rollouts; `22.42 rad²/s²` out-of-distribution on open-loop PD).
- **Retargeted Motion Kinematics**:
  - `0.0%` of frames violate hard or soft joint position limits.
  - `99th percentile |joint_vel| = 7.286 rad/s`, `max |joint_vel| = 38.837 rad/s`.
  - `root_z` ranges from `0.0705 m` (floor crawling in `ground2`) to `1.3629 m` (high jumps in `jumps1`), with **0 standard LAFAN walking/running files** in `light_supermimic_t4_trial.json`.

## 4. Conclusions & Recommended Fixes
1. **Primary Cause — Severe Under-Training & Downscaled Batches/Networks**: `8,081` updates at `policy.batch_size=128` equals only **252** upstream SuperMimic updates. Local `RTX 3090` (24 GB VRAM, native `bfloat16`) can run full or near-full upstream batch sizes (`wm: 2048`, `policy: 2048–4096`) and larger MLPs (`1024–2048` hidden) for `20k–50k+` updates without Colab T4 constraints.
2. **Dataset Subset Bias**: Replace the T4 transfer subset (`81.4%` dance/crawling/jumps, `0` LAFAN walk) with either a locomotion-first curriculum (`walk` / `run` clips first to verify clean tracking) or the full screened train split (`outputs/humanoid_light_bfm/manifest.json`), and enable `sampling_mode = "return"` (`ReturnSampler`).
3. **Action Mapping & Tracked Bodies**:
   - Replace hard-clamped `JointRangePositionAction` (`clamp(-1, 1)` on unbounded `MultiEncoderAE` outputs) with un-clamped couple-aware `JointPositionActionCfg(scale=get_action_scale(control))` (already implemented in `robots/humanoid/humanoid_light/mjlab.py:99-111`) or smooth `tanh` bounding before `action_rate_l2` so BPTT gradients never vanish.
   - Exclude the 4 massless sensor frames (`lower_imu`, `upper_imu`, `head_cam_link`, `lower_cam_link`) from `command.body_names` in `light_supermimic.py`.
