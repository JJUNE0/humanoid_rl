# Learning Algorithms (`src/humanoid_rl/algorithm/`)

This package contains the learning adapters, configurations, and auxiliary discriminators/world-models used across `humanoid_rl`. Each algorithm preserves its upstream learning core while sharing robot definitions (`src/robots/`) and motion pipelines (`src/humanoid_retarget/`).

```text
src/humanoid_rl/algorithm/
├── supermimic/   # World-Model + Differentiable BPTT motion tracking (Humanoid-Light)
├── bfm_zero/     # Forward-Backward + Conditional Policy Regularization (FB-CPR-Aux)
├── flashsac/     # Off-policy distributional C51 SAC core + mjlab integration patch
├── amp/          # Adversarial Motion Priors (LSGAN style discriminator + replay relabeling)
└── add/          # Adversarial Differential Discriminator (single-clip residual tracking)
```

---

## 1. `supermimic/` — World Model + BPTT Motion Tracking & Sim-to-Real

Connects `third_party/supermimic` (`WorldModelLearner` + `ActorCriticPolicyLearner` with Muon/AdamW BPTT) and `third_party/mjlab_research` (`MuJoCo Warp` 400 Hz physics / 50 Hz coupled-PD control) to the 26-DoF `Humanoid-Light` robot.

- **Entry point**: `uv run humanoid train --experiment light_supermimic_s2r_full --runtime local`
- **Defaults**: [`supermimic/configs/light.json`](supermimic/configs/light.json), layered with experiment presets under [`../configs/experiments/`](../configs/experiments/).
- **Key capabilities**:
  - Generalized-coordinate (`LightGenCoordRepresentation`) 32-step imagined rollouts synchronized with `SimContext` (`foot_friction`, `base_com`, `encoder_bias`).
  - Hardware-deployable proprioceptive history (`actor_obs="proprio"`, `5 × 84 = 420-D`) + body-frame reference linear/angular twist goals (`640-D`).
  - Differentiable 3-tier whole-body soft-contact regularization (head protection, conditional wrist/elbow/knee/pelvis cushioning, and foot impact/slip smoothing).
  - Zero-shot Sim-to-Sim validation & interactive Viser tools in [`../sim2sim/`](../sim2sim/).

---

## 2. `bfm_zero/` — Unsupervised Zero-Shot Behavior Foundation Model (`FB-CPR-Aux`)

Connects `third_party/BFM-Zero` (`FBcprAuxAgent`: Forward-Backward successor representations $F/B$, latent-conditioned discriminator $D$, style critic $Q_D$, and auxiliary control critic $Q_R$) to `H1-Motion-Prior`, `G1-Motion-Prior`, and `Humanoid-Light-Motion-Prior`.

- **Defaults**: [`bfm_zero/configs/train.json`](bfm_zero/configs/train.json) (`z_dim=256`, residual width `2048`, `192M` transitions).
- **Detailed architecture, memory sizing & usage**: see [`bfm_zero/README.md`](bfm_zero/README.md).

---

## 3. `flashsac/` — Distributional Off-Policy SAC Core

Wraps `third_party/FlashSAC` (distributional C51 twin critic, delayed actor update, reward-scale normalization, optional auxiliary scalar control-cost critic, and n-step returns with timeout bootstrapping) for native `mjlab` vector environments (`G1-Carry-Box`, `Mjlab-Cartpole-Balance`).

- **Defaults**: [`flashsac/configs/train.json`](flashsac/configs/train.json) (`batch_size=2048`, `updates_per_step=2`, `n_step=3`, `discount=0.99`, `101` C51 bins).
- **Submodule patch**: [`flashsac/patches/flashsac-mjlab.patch`](flashsac/patches/flashsac-mjlab.patch) (applied idempotently via `python3 scripts/apply_flashsac_patch.py`).

---

## 4. `amp/` — Adversarial Motion Priors

Layers a least-squares style discriminator ([Peng et al., 2021](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)) over `FlashSAC` with aligned GPU ring-buffer features (`AmpReplay`) and on-the-fly batch reward relabeling before each SAC update.

- **Defaults & presets**: [`amp/configs/train.json`](amp/configs/train.json) and YAML presets under [`../configs/`](../configs/) (`amp.yaml`, `amp_control.yaml`, `amp_object_control.yaml`, `amp_single_episode_control.yaml`).
- **Feature schemas**:
  - `robot83` (`166-D` transition pair): root height, heading-free 6D orientation, heading-frame root twist, 29-DoF joint positions/velocities, and root-relative wrist/ankle positions.
  - `g1-body-object104-v1` (`208-D` transition pair): extends `robot83` with heading-frame box pose, relative box twist, and wrist-to-box vectors for coherent robot-object RSI.
- **Objective**: LSGAN loss with expert gradient penalty ($w_{\text{gp}} = 10$) and bounded style reward $r_{\text{AMP}} = s \cdot \text{clamp}(1 - 0.25 (D(x) - 1)^2, 0, 1)$ (`n_step=1`).

---

## 5. `add/` — Adversarial Differential Discriminator

Implements single-trajectory phase-aligned tracking ([MimicKit ADD](https://github.com/xbpeng/MimicKit/blob/main/mimickit/learning/add_agent.py)) on top of `FlashSAC`.

- **Defaults & presets**: [`add/configs/train.json`](add/configs/train.json) and [`../configs/add_single_episode_control.yaml`](../configs/add_single_episode_control.yaml) (or `add_single_episode_reference_control.yaml`).
- **Residual formulation**: At each 50 Hz frame, computes $\Delta_t = x_{\text{ref}}[t] - x_{\text{sim}}[t]$ (`109-D` robot + box state error, `218-D` consecutive residual pair). Positive target is the exact zero residual $\mathbf{0}$, normalized by running MAE without mean centering so zero error remains invariant; reward is $\text{softplus}(\text{logit})$.
