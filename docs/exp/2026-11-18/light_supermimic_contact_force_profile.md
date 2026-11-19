# Experiment Log: Whole-Body Ground Contact Force (`N`) Profiling Across LAFAN1 Tasks in Isaac Sim 5.1 (PhysX USD)

- **Date**: 2026-11-18
- **Execution Domain**: `VALIDATION_SIM`
- **Policy Checkpoint / Hash**:
  - `runs/light_supermimic_lafan_all/rtx3090_lafan_all_stage2_20261118/training/checkpoints/step_65536000/last.pt` (`560f737d841f674b8cdf501b84caecc5a426feff1fe8af2a96e114c5e0c653f2`)
- **Saved JSON Evidence**:
  - Held-out `subject5` (`434` clips, `13` sequences): `outputs/sim2sim/isaac_lafan_all_stage2_contact_forces_val.json`
  - Train reference tasks (`walk1_subject1`, `sprint1_subject2`, `ground1_subject1`, `fallAndGetUp2_subject2`): `outputs/sim2sim/isaac_lafan_all_stage2_contact_forces_train_special.json`
- **Evidence Status**: `PASS`

---

## 1. Robot Physical Baseline (`Humanoid-Light`)
- **Total Mass**: `36.140 kg` → **1 Body Weight (`1.0 × BW`) = `354.5 N`**
- **Rated Joint Effort Limits (`src/robots/humanoid/humanoid_light/assets/control.json`)**:
  - Hip pitch / Knee: `120.0 Nm` | Hip roll/yaw: `60.0 Nm`
  - Shoulder pitch/roll: `60.0 Nm` | Shoulder yaw: `17.0 Nm`
  - Elbow: `36.0 Nm` | Torso pitch/roll (coupled): `36.0 Nm`
  - Ankle pitch/roll (coupled `2:1`): `28.0 Nm`
  - Wrist: **`14.0 Nm`** | Head: **`5.5 Nm`** (carries `head_cam_link`)

---

## 2. Measurement Methodology (`test/profile_light_supermimic_contact_forces.py`)
- Attached `ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=8)` across all rigid bodies at `400 Hz` (`sim_dt = 0.0025 s`, `decimation = 8`).
- Excluded post-fall crash steps (`bad_fall`) from normal tracking statistics, and gated each body group by link-geometry ground proximity (`wrist < 0.10 m`, `foot < 0.13 m`, `elbow < 0.14 m`, `knee < 0.16 m`, `pelvis/shoulder/head < 0.18 m`, `torso < 0.24 m`) with upward vertical reaction `F_z > 1.0 N` so mid-air self-collisions do not contaminate floor reaction measurements.
- Reported both:
  1. **`50 Hz` Step-Mean Force (`N`)**: Impulse-averaged force over the `20 ms` control step.
  2. **`400 Hz` Substep Peak Force (`N` & `× BW`)**: True `2.5 ms` rigid-body impact transient peak.

---

## 3. Observed Ground Reaction Forces by Task Category

### 3.1 Upright Locomotion & Whole-Body Skills (Non-Foot Ground Contact `< 0.5%`)

| Task Sequence | Split | Single-Foot `50 Hz` Mean | Single-Foot `50 Hz` P99 | Single-Foot `400 Hz` P95 | Single-Foot `400 Hz` P99 (`× BW`) | Non-Foot Floor Contacts |
|---|---|---:|---:|---:|---:|---|
| `walk1_subject1` | `train` | `329 N` (`0.93×`) | `601 N` (`1.70×`) | `615 N` | **`1,371 N` (`3.87×`)** | None (`0.0%`) |
| `walk1_subject5` | `val` | `342 N` (`0.96×`) | `956 N` (`2.70×`) | `1,019 N` | **`2,531 N` (`7.14×`)** | None (`0.0%`) |
| `dance2_subject5` | `val` | `326 N` (`0.92×`) | `685 N` (`1.93×`) | `677 N` | **`1,674 N` (`4.72×`)** | None (`0.0%`) |
| `jumps1_subject5` | `val` | `327 N` (`0.92×`) | `853 N` (`2.41×`) | `808 N` | **`2,006 N` (`5.66×`)** | None (`0.0%`) |
| `fight1_subject5` | `val` | `348 N` (`0.98×`) | `897 N` (`2.53×`) | `902 N` | **`2,477 N` (`6.99×`)** | None (`0.0%`) |
| `aiming2_subject5` | `val` | `346 N` (`0.98×`) | `984 N` (`2.78×`) | `979 N` | **`3,030 N` (`8.55×`)** | None (`0.0%`) |
| `sprint1_subject2` | `train` | `317 N` (`0.89×`) | `871 N` (`2.46×`) | `774 N` | **`2,258 N` (`6.37×`)** | Dive clip (`4.9%` wrist) |
| `run1_subject5` | `val` | `366 N` (`1.03×`) | `1,186 N` (`3.35×`) | `1,352 N` | **`3,673 N` (`10.36×`)** | None (`0.0%`) |

### 3.2 Multi-Contact / Ground / Recovery Tasks (Active Non-Foot Floor Contacts)

| Task Sequence | Split | Feet `400 Hz` P99 | Hands/Wrists (`14 Nm`) | Knees (`120 Nm`) | Elbows (`36 Nm`) | Hips/Pelvis | Shoulders / Torso | Head (`5.5 Nm`) |
|---|---|---:|---|---|---|---|---|---|
| `ground1_subject1` | `train` | `450 N` | **`92.3%` time**, P99 `375 N` | `5.1%`, P99 `152 N` | `5.4%`, P99 `333 N` | `79.6%`, P99 `697 N` | Shld `54.3%`, P99 `827 N` | **`2.6%`, P99 `1,493 N`** |
| `ground1_subject5` | `val` | `630 N` | **`90.0%` time**, P99 `541 N` | `<0.5%` | `1.8%`, P99 `361 N` | `28.7%`, P99 `975 N` | Shld `26.3%`, P99 `1,061 N` | **`0.7%`, P99 `777 N`** |
| `fallAndGetUp2_s2` | `train` | `1,417 N` | **`57.0%` time**, **P99 `2,787 N`** | `18.6%`, P99 `560 N` | `3.6%`, P99 `590 N` | `32.0%`, P99 `1,392 N` | Shld `28.5%` (`1,523 N`), Torso `8.3%` (`1,695 N`) | **`7.3%`, P99 `2,187 N`** |
| `fallAndGetUp1_s5` | `val` | `1,976 N` | **`42.4%` time**, **P99 `4,292 N`** | `6.4%`, P99 `1,600 N` | `<0.5%` | `19.7%`, P99 `1,925 N` | Shld `3.7%`, P99 `2,473 N` | `<0.5%` |
| `walk3_subject5` (low crawl) | `val` | `3,004 N` | **`47.9%` time**, **P99 `3,996 N`** | `<0.5%` | `<0.5%` | `12.1%`, P99 `2,284 N` | `<0.5%` | `<0.5%` |
| `pushAndStumble1_s5` | `val` | `2,490 N` | `1.4%` catch, **P99 `9,721 N`** | `<0.5%` | `<0.5%` | `<0.5%` | `<0.5%` | `<0.5%` |

---

## 4. Key Takeaways for Hardware Safety & Soft-Contact BPTT Design
1. **Stance vs Transient Impact (`50 Hz` vs `400 Hz`)**: Normal walking stance averages `~330–345 N` (`~0.95 × BW`), which is completely nominal, **but** rigid heel-strike `2.5 ms` transients (`400 Hz` P99) reach `1,371 N` (`3.9× BW` on `subject1`) to `2,531 N` (`7.1× BW` on `subject5`), and `3,673 N` (`10.4× BW`) in `run1`. Pre-landing downward foot velocity (`Impact`) + `Slip` BPTT penalties directly target this transient impulse.
2. **Wrist/Hand Overload in Ground & Recovery Tasks**: `left/right_wrist_link` (`14 Nm` rated torque) supports the body `42%–92%` of the time in `ground` and `fallAndGetUp`, experiencing **`2,787 – 4,292 N`** `400 Hz` hand-slam spikes during `fallAndGetUp` and `9,721 N` during stumble catches.
3. **Head Strike Risk**: During `ground1` (`0.7%–2.6%`) and `fallAndGetUp2` (`7.3%`), `head_link` (`5.5 Nm` + camera) touches the ground at `777 – 2,187 N`, requiring an unconditional ground-clearance & impact-avoidance term on the head.
