# Humanoid control: mjlab, learning algorithms and motion data

## Common experiment launcher

Choose the experiment separately from the execution location. BFM-Zero,
FlashSAC and the existing G1 SuperMimic-style trainer share a public launcher;
each keeps its own learning code and compatible Python environment.

```bash
python3 humanoid.py list
python3 humanoid.py launch --experiment light_bfm_zero --runtime local --dry-run
python3 humanoid.py setup --method bfm_zero
python3 humanoid.py launch --experiment light_bfm_zero_smoke --runtime local
python3 humanoid.py launch --experiment light_supermimic_smoke --runtime colab \
  --submit --colab-gpu T4
python3 humanoid.py launch --experiment light_bfm_zero_smoke --runtime colab \
  --bundle-data --output runs/colab/light_bfm_zero_smoke
```

See [the training guide](docs/training.md) for code ownership, runtime profiles,
Colab notebooks, data packaging, result persistence and checkpoint resume.
With `google-colab-cli` installed and authenticated, `--runtime colab --submit`
runs the current working code on a dedicated Colab GPU, retrieves results to the
PC, and releases the session. Omitting `--submit` exports a notebook bundle.
Light SuperMimic tracking uses the official learner with the converted robot and
screened local motions; see the [Light/Colab guide](docs/light_supermimic.md).
Existing direct training and playback entry points below remain compatible.

JAX/TPU experiments have moved to [tpu-gym](https://github.com/JJUNE0/tpu-gym), with an independent environment, task registry, tests, and training outputs.

Native mjlab task environments trained with the checked-out FlashSAC PyTorch
agent, networks, update functions, GPU replay buffer and checkpoint implementation.
AMP-only motion imitation is available with `--reward-mode amp`. I-BFM is a
reference for scene geometry and task semantics; its recorded `motion.npz` files
are read as data, and its environment/controller is not imported at runtime.

## Code layout

```text
src/
├── humanoid_rl/
│   ├── algorithm/
│   │   ├── flashsac/       # Training loop, upstream config, checkpoints and logging
│   │   ├── amp/            # Discriminator, expert features, replay and style rewards
│   │   ├── add/            # Differential discriminator and tracking features
│   │   └── bfm_zero/       # Original FB-CPR-Aux integration; configs/train.json
│   ├── env/                # mjlab adapter, actions and controller wiring
│   │   └── imitation/      # AMP/ADD environment setup and reference resets
│   ├── tasks/
│   │   ├── carry_box/      # Scene assets, observations, rewards, termination, stage MLP
│   │   └── motion_prior/   # mjlab H1 mocap-guided pretraining environment
│   ├── train.py            # Training CLI
│   └── play.py             # Playback CLI
├── humanoid_retarget/
│   └── datasets/           # Local mocap files, readers and geometry metadata
│       ├── cmu/            # Official CMU ASF/AMC metadata, C3D timing headers and manifests
│       ├── lafan1/         # Human BVH and H1/G1 retargeted copies
│       └── loco_mujoco/    # Default H1 mocap copies
└── robots/
    └── humanoid/
        ├── g1/             # Native model access, joint conventions and control profiles
        ├── h1/             # Copied MJCF, STL meshes and local scene includes
        └── humanoid_light/ # 26-DoF Humanoid Light MJCF, meshes and coupled-PD control profile
```

`humanoid_retarget` reads source motions without importing RL trainers or
simulators. It stores original LAFAN1 BVH, CMU ASF/AMC sources, copied H1/G1
retargeted motions and LocoMuJoCo's default H1 mocap. Its carry reader loads
already-retargeted G1 motion; human-to-H1 conversion is not implemented yet.
Algorithm-specific feature encoding, caching and transition sampling remain in
`algorithm/amp`.

`robots` owns robot definitions shared by tasks and motion processing. G1's
native XML, meshes and actuator defaults are reused from mjlab, rather than
copied. H1 and 26-DoF Humanoid Light have self-contained MJCF and mesh sets under
`robots/humanoid/h1` and `robots/humanoid/humanoid_light`. The mjlab action wiring for a saved G1 control profile lives in `env`.
Third-party repositories remain under `third_party`; their source layouts are
preserved. BFM-Zero uses the registered `H1-Motion-Prior` and
`Humanoid-Light-Motion-Prior` tasks, with its original FB-CPR-Aux learning core
and the local mjlab environment.

Root `train.py`, `play.py`, `humanoid-train` and `humanoid-play` keep the same CLI
arguments. Existing checkpoint tensors and metadata retain their formats.

Inspect copied motion data with `.venv/bin/python -m humanoid_retarget.datasets`.
See [dataset details](src/humanoid_retarget/datasets/README.md) and
[the copied H1 model](src/robots/humanoid/h1/README.md).

## Motion datasets (LAFAN1 & CMU) and Humanoid-Light retargeting

Large raw mocap archives (`.zip`, `.bvh`, `.amc`, `.npz`, `.pkl`) and generated
`outputs/` directories are excluded from Git; provenance manifests and SHA-256
checksums are tracked under `src/humanoid_retarget/datasets/`.

### 1. Download original LAFAN1 & CMU datasets

- **LAFAN1 (Ubisoft La Forge Animation Dataset — 77 BVH sequences, ~30 Hz)**
  - Official repository: <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>
  - Direct archive (`lafan1.zip`, 144,051,503 bytes):
    <https://github.com/ubisoft/ubisoft-laforge-animation-dataset/raw/master/lafan1/lafan1.zip>
  - Automatic download & extraction to `src/humanoid_retarget/datasets/lafan1/raw/`:
    ```bash
    .venv/bin/python scripts/prepare_h1_data.py
    ```

- **CMU Graphics Lab Motion Capture Database (2,548 ASF/AMC sequences, 60/120 Hz)**
  - Official database & trial index: <http://mocap.cs.cmu.edu/> (<http://mocap.cs.cmu.edu/search.php>)
  - Official bulk ASF/AMC archive (`allasfamc.zip`, 1,083,549,385 bytes,
    SHA-256 `4b0fc3a7ba3cbb2aa23948b4cefe7efa45edf864b2b5e00713659a4f2144a591`):
    <http://mocap.cs.cmu.edu/allasfamc.zip>
  - Download into `src/humanoid_retarget/datasets/cmu/full/archives/allasfamc.zip` and unpack/verify frame rates:
    ```bash
    mkdir -p src/humanoid_retarget/datasets/cmu/full/archives
    curl -L http://mocap.cs.cmu.edu/allasfamc.zip \
      -o src/humanoid_retarget/datasets/cmu/full/archives/allasfamc.zip
    .venv/bin/python scripts/prepare_cmu_full.py \
      --archive src/humanoid_retarget/datasets/cmu/full/archives/allasfamc.zip \
      --index src/humanoid_retarget/datasets/cmu/full/metadata/official_index.json
    ```

### 2. Retarget LAFAN1 & CMU to 26-DoF Humanoid-Light (50 Hz)

1. Stage the tracked `Humanoid-Light` MJCF, meshes, and control profile (or re-export from USD with `scripts/convert_humanoid_light.py` when Isaac Sim is installed), then generate the calibrated Mink/DAQP IK profile (`outputs/humanoid_light_retarget/light_profile.json`):
   ```bash
   mkdir -p outputs/humanoid_light_conversion
   cp -r src/robots/humanoid/humanoid_light/assets/* outputs/humanoid_light_conversion/
   .venv/bin/python scripts/retarget_humanoid_light_lafan.py --no-render
   ```
2. Convert all 77 LAFAN1 BVH sequences to 50 Hz `qpos`/`qvel` archives (`outputs/humanoid_light_lafan_full/motions/*.npz`) and build `sequence_index.csv`:
   ```bash
   .venv/bin/python scripts/retarget_humanoid_light_lafan_full.py --workers 6
   .venv/bin/python scripts/summarize_humanoid_light_lafan_full.py
   ```
3. Convert all timing-verified CMU ASF/AMC sequences to 50 Hz archives (`outputs/humanoid_light_cmu_full/motions/*.npz`) and build `sequence_index.csv`:
   ```bash
   .venv/bin/python scripts/retarget_humanoid_light_cmu_full.py --workers 12
   .venv/bin/python scripts/summarize_humanoid_light_cmu_full.py
   ```
4. Screen joint ranges, joint velocity limits, floor clearance, and non-structural self-collisions across quarter-frame interpolants to produce the train/validation manifest (`outputs/humanoid_light_bfm/manifest.json`):
   ```bash
   OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/prepare_humanoid_light_bfm.py --workers 4
   ```

## Algorithm-local training configuration

Humanoid Light can pretrain on the prepared CMU + LAFAN mixture with
`./scripts/train_humanoid_light_bfm.sh`. See the
[Humanoid Light BFM-Zero guide](docs/humanoid_light_bfm_zero.md) for data screening,
resource settings, launch/resume commands, and reference tracking evaluation.

Each algorithm has a `configs/train.json`:

- `src/humanoid_rl/algorithm/flashsac/configs/train.json`: FlashSAC runtime,
  network and optimizer defaults.
- `src/humanoid_rl/algorithm/amp/configs/train.json`: AMP discriminator/data/reset
  settings, layered on the FlashSAC training defaults.
- `src/humanoid_rl/algorithm/add/configs/train.json`: ADD discriminator/data/reset
  settings, layered on the FlashSAC training defaults.
- `src/humanoid_rl/algorithm/bfm_zero/configs/train.json`: FB-CPR-Aux training,
  network, expert-motion and environment settings.

Defaults are loaded first, followed by an optional `--train-config experiment.json`,
then explicit CLI options. Experiment JSON files can override only the fields
they change. Existing `--imitation-config` YAML files still override AMP/ADD
settings. Resume restores the saved learning configuration; BFM-Zero rejects a
new `--train-config` during resume and allows explicitly requested runtime
changes such as device, number of environments and the larger training budget.

```bash
.venv/bin/python train.py --algorithm bfm_zero --task H1-Motion-Prior \
  --train-config src/humanoid_rl/algorithm/bfm_zero/configs/smoke.json
```

This smoke preset runs four H1 environments for 10 vector steps and 10 original
FB-CPR-Aux updates. It verifies execution, not learned motion quality.
BFM-Zero dependencies are installed with `uv sync --extra bfm-zero`. See the
[BFM-Zero code guide](src/humanoid_rl/algorithm/bfm_zero/README.md) for observation
dimensions, data processing, playback and the scope of this initial integration.

## Setup

```bash
git submodule update --init --recursive
python scripts/apply_flashsac_patch.py
uv sync
uv run python train.py --list-tasks
```

Python 3.13 and the locked PyPI mjlab 1.6.0 installation are used. The mjlab
submodule is a source reference, not an editable dependency: its current dependency
requirements differ from the published package despite sharing a version number.
Do not run `uv sync` inside FlashSAC: its complete simulator dependency set targets
Python 3.10/3.11. Only Gymnasium and OmegaConf are added for its PyTorch core here.
Training requires an NVIDIA GPU.

The tracked `patches/flashsac-mjlab.patch` preserves the minimal submodule changes:
optional JAX imports, opt-in tensor action output, mixed-precision checkpoint
compatibility, a replay-batch transform hook for AMP rewards, safe replay capacity
restoration, and per-transition n-step discounts at early timeouts (also used
by the Wolf integration), plus an optional actor Q-bonus hook for the control critic. The patch script
is idempotent and rejects mismatching changes instead of overwriting them.

## Recorded controller comparison

To repeat ADD with the demonstration controller's joint command mapping and
explicit PD, use `--imitation-config configs/add_single_episode_reference_control.yaml`
with `--reward-mode add` and the same training arguments below.

`controller_profile: reference` reads only the episode's saved `robot_config.yaml`,
`physics.npz`, `metadata.json` and joint ranges from `robot.xml`. It applies the
recorded default joint angles, individual action coefficients (including all
three waist overrides), action normalization/clipping, Kp/Kd, torque limits and
passive joint armature/friction/damping. The actuator is mjlab's
`IdealPdActuator`: it recomputes `clip(kp*(target-q)-kd*qd, -effort, effort)` at
each 200 Hz physics substep, with a 50 Hz policy. The native mjlab G1 geometry,
scene and learning networks are retained. This reproduces the joint control path;
it does not claim every inertial/contact property of the teacher robot is equal.

The action target is `clip(default + clip(raw_action*5, -5, 5) * coefficient *
effort/kp, joint_lower, joint_upper)`. The reference mode requires
`action_scale_multiplier: 1` because the recorded factor of five is already
included. The recorded explicit controller clips to its `effort_limit` array;
its separate `effort_limit_scale: 0.8` training field is not applied in this
sim2sim torque loop. The actual archived controller file is identified by the
SHA recorded in the episode metadata.

Every checkpoint contains `controller.pt` with a fingerprinted portable profile.
Resume and playback restore it alongside `fixed_scene.pt`; source configuration
files are not needed for playback. Existing native-controller checkpoints retain
their original controller semantics. The reference comparison starts with fresh
policy, critic, discriminator and replay, keeping the previous ADD rewards,
control costs, phase observation, fall rule and reset rules.

## ADD single-trajectory tracking

ADD is available with `--reward-mode add`. It tracks the same successful
`episode_00002`, with its exact initial robot/box/support scene, using the
existing FlashSAC actor, double critic, aligned replay and auxiliary control critic:

```bash
.venv/bin/python train.py --task G1-Carry-Box \
  --reward-mode add --imitation-config configs/add_single_episode_control.yaml \
  --observe-hand-feet --num-envs 1024 --total-env-steps 51200000 \
  --buffer-size 2097152 --min-buffer 102400 --batch-size 2048 \
  --updates-per-step 2 --n-step 1 --no-mixed-precision \
  --save-every 10240000 --save-replay \
  --eval-every 2500 --eval-num-envs 64 --eval-steps 600 \
  --wandb-project humanoid-g1-carry --wandb-entity bjy6614-postech
```

The actor receives the original 120D observation plus reference phase (1D).
At each 50 Hz step, ADD computes `reference[frame] - simulation[frame]` (109D):
world root XYZ/6D rotation, heading-frame root velocities, joint positions and
velocities, wrist/ankle XYZs, all existing box-relative position/6D rotation/
relative velocities/hand-to-box XYZs, plus world box XYZ. World coordinates
subtract each vector world's origin. Consecutive residuals form the 218D
discriminator input. Both robot and box trajectory errors affect the reward.

The discriminator's positive target is the all-zero residual. Following the
[official MimicKit implementation](https://github.com/xbpeng/MimicKit/blob/main/mimickit/learning/add_agent.py),
it uses BCE logits, positive/negative logit gradient penalties and output-weight
regularization. A running mean-absolute-error normalizer divides without
centering, preserving the zero target. Reward is `softplus(logit)`, capped at
`-log(1e-4)` as in the official probability floor, and replay rewards are
recomputed using the current discriminator before each SAC update.

This is an ADD + FlashSAC experiment with native mjlab G1 features and box
tracking, rather than the paper's complete PPO setup. The comparison config
keeps the previous native PD/action scaling, fall rule and control costs;
it adds the required reference clock/actor phase and ends the episode at the
last reference frame (11.34 s). Reference completion is a true terminal, so
the critic does not bootstrap beyond the clip. Carry task reward and frozen
MLP are off. Keeping the controller unchanged retains the control mismatches
found in the earlier diagnosis; this experiment tests this ADD configuration.

Evaluation records physical root/joint/hand-foot/box errors as well as carry
contact/lift metrics. `best` first minimizes phase-aligned reference tracking
RMS with a fixed expert-only scale, so moving discriminator statistics do not
change the ranking. `add.pt` stores the discriminator, normalizer and optimizer;
the existing replay exporter saves paired residuals. Resume restores the ADD
mode, and `play.py --task G1-Carry-Box --checkpoint <checkpoint> --viewer native`
restores the 121D actor and portable fixed scene without requiring rollout files.

## AMP-only training

For the single-demonstration overfit diagnostic, use:

```bash
.venv/bin/python train.py --task G1-Carry-Box \
  --amp-config configs/amp_single_episode_control.yaml \
  --observe-hand-feet --num-envs 1024 --total-env-steps 51200000 \
  --buffer-size 2097152 --min-buffer 102400 --batch-size 2048 \
  --updates-per-step 2 --n-step 1 --no-mixed-precision \
  --save-every 10240000 --save-replay \
  --eval-every 2500 --eval-num-envs 64 --eval-steps 600 \
  --wandb-project humanoid-g1-carry --wandb-entity bjy6614-postech
```

`single_motion: episode_00002/motion.npz` selects only that successful 11.34 s
demonstration. `fixed_initial_state: true` replaces every scene randomization
and ordinary reset with the recorded first frame: robot pose/velocities/joints,
box geometry pose/velocities/XYZ dimensions/mass, exact support dimensions/poses,
friction and goal. The box is 0.3 m per side and 0.7 kg, and the source supports
are 0.4 m per side. Original world poses are retained modulo vector-world origins;
the nominal initial robot-to-box distance is 1.5 m. No random-time RSI is used.
The native mjlab G1 dynamics, PD/action scaling, FlashSAC settings and control
critic stay the same; this does not claim identical robot dynamics to I-BFM.
The actor remains 120D, AMP pairs 208D, with no MLP or carry task reward.

This diagnostic has **no held-out expert clip**. Expert diagnostics are explicitly
named `amp/train_reference_*`; no validation accuracy/generalization is reported
for the training clip. The 12 s policy evaluation covers the full demonstration,
always starts from its exact first frame and leaves fall termination disabled.
For this experiment only, `best` prioritizes placement, held lift and hand contact
before goal distance and upright diagnostics. A low-root-height/deep-crouch fall
flag remains a log-only proxy and does not reset the policy.

Every checkpoint includes `fixed_scene.pt`. `play.py --task G1-Carry-Box
--checkpoint <checkpoint> --viewer native` restores the scene from that portable
bundle without needing source rollout files. Resume preserves the AMP config and
scene bundle; training still needs its original expert data/fingerprint.

The carry sequence adds components and includes a no-MLP comparison, with a fresh
policy/replay for each experiment and the same training budget:

| Experiment | Actor context | Motion reward | Carry task reward |
| --- | --- | --- | --- |
| 1 | Proprioception, scene, previous action, wrist/ankle XYZ (120D) | AMP | Off |
| 2 | Experiment 1 + frozen stage MLP context (148D) | Same AMP | Off |
| 2b | Same frozen MLP actor (148D) | Robot + box AMP (208D pair), coherent scene RSI | Off |
| 2c | Experiment 1 actor without MLP (120D) | Same robot + box AMP and coherent RSI as 2b | Off |
| 3 (planned) | Experiment 2b + commanded stage | Stage-conditioned object AMP | Off |
| 4 (planned) | Experiment 3 | Same conditional AMP | Stage progress and validated transitions |

The six control costs and auxiliary critic remain active in these comparisons.
Experiment 2 is available with `--stage-model`:

```bash
uv run python train.py --task G1-Carry-Box --amp-config configs/amp_control.yaml \
  --stage-model ../I-BFM-Code/ibfm/analysis/carry_mlp_failure_v1/seed42/best.pt \
  --num-envs 1024 --total-env-steps 51200000 --no-mixed-precision
```

For raw checkpoints, the first training episode supplies the original named-joint
defaults, palm offsets and box body/geometry conventions. `--stage-reference`
can specify that source episode explicitly. Source files are read only during
initial configuration; runtime imports no I-BFM training code. Each checkpoint
includes a portable `stage_model.pt` with weights, preprocessing and source hash.
Resume/play restore this bundle without the original data/code/checkpoint paths.
The exported bundle can also be passed to `--stage-model` for a fresh experiment.

The frozen predictor receives its original 101D schema, not the 120D actor input.
The adapter reorders joints by name, uses source default joint angles, scales local
base angular velocity by .25, uses full-root coordinates and rotation columns 0/2,
and converts geometry-centred box poses into the original body frame. Palm points,
hand-box contacts, virtual COM velocity and goal-minus-box vectors follow that
schema. Root world XYZ has the vector-environment origin removed. A six-pose FK
comparison found about 5 mm wrist-position differences between robot models;
collision and scene differences remain possible sources of transfer error.

The 28 appended inputs are stage probabilities (9), required-skill probabilities
(9), per-stage sigmoid progress (9), and calibrated current-failure probability
(1). Weights, normalization and calibration remain frozen; inference stays on the
environment device and consumes no random samples. This estimator changes only
policy/critic observations in experiment 2. AMP remains 166D, and no progress,
transition or failure reward is added. `stage/input_clip_fraction` logs input
normalization saturation; it is a diagnostic, not a calibrated OOD detector.

Labels describe current physical stages and current box-lost/fallen states;
failure probability does not forecast episode failure. Progress was trained as
normalized time inside normally completed stages and is not geometric task
completion. Experiments 3/4 will use the same v3 definitions for expert sidecars,
with failed/incomplete progress masks preserved. A commanded stage must be
distinct from the estimated current stage; matching expert conditions prevents
the discriminator from learning only label-distribution differences.

The planned fourth experiment uses bounded progress shaping and a one-time bonus
for each validated normal stage transition. Absolute progress is not paid every
step. The controller must reject stage flicker, repeated visits, failure labels
as ordinal progress, and invalid transitions; simulator physical evidence must
validate advancement. Terminal/reset accounting and stage memory must be carried
into replay. No such task shaping is enabled in experiments 1/2.

Experiment 2b adds object interactions **before** stage-conditioned AMP. Use
`configs/amp_object_control.yaml`; `observe_object: true` selects the versioned
`g1-body-object104-v1` schema. Each state preserves the original robot83 prefix
and appends box position relative to the root (3), box rotation in the root heading
frame (6), box-minus-root linear/angular velocities in that frame (3 + 3), and
vectors from the two wrist link origins to the box centre (6). A consecutive pair
is 208D; the discriminator is `208 -> 512 -> 256 -> 1`. Actor/critic inputs remain
148D with the frozen MLP. No goal, task stage, contact bit, dimensions or mass is
added to the discriminator, and no carry task reward is enabled.

Expert box free-body poses/velocities are converted to the **geometry centre**
and frame using each episode's `box_config.yaml`. This includes the angular
velocity cross offset term for linear velocity. Expert wrists use native mjlab
G1 FK, just like the robot83 features. Box geometry and support layout sidecars
are included in cache fingerprints; legacy robot83 caches/checkpoints still load
under their original schema. New object AMP starts fresh networks and replay.

With object AMP, 50% reference starts restore the same frame's robot, box, box
dimensions/mass, original source/target support top locations and goal under one
shared horizontal translation/yaw rotation. Support widths retain their sampled
values. Collision bounds and box inertia are updated with the dimensions. These
RSI starts use expert scene layouts, so their support positions/heights can exceed
the ordinary `carry_box.yaml` sampling ranges. The other 50% and all evaluation/
playback starts use the ordinary random scene. Nothing welds the box to the hands.
This experiment changes both the discriminator features and the inconsistent
robot-only RSI; it does not isolate those two effects as separate ablations.

```bash
uv run python train.py --task G1-Carry-Box \
  --amp-config configs/amp_object_control.yaml --scene-config configs/carry_box.yaml \
  --stage-model logs/flashsac/G1-Carry-Box/20261023_112500_002673/step_51200000/stage_model.pt \
  --num-envs 1024 --total-env-steps 51200000 \
  --buffer-size 2097152 --min-buffer 102400 --batch-size 2048 \
  --no-mixed-precision --eval-every 2500 --eval-num-envs 64 --eval-steps 500
```

Evaluation now also logs any/bilateral hand-box contact, a >5 cm rise while both
hands contact the box, the rough placement condition, maximum rise and minimum
goal error. These are diagnostics, not task rewards; `best` still ranks upright
survival. Full replay storage for 148D policy / 104D AMP / six control terms is
about 4.24 GiB at 2,097,152 transitions, before network/simulation overhead.
The no-MLP comparison uses the same capacity and approximately 3.80 GiB. It is
queued behind experiment 2b with the same 1,024 envs / 50,000 vector steps, seed,
object discriminator, coherent scene RSI and control critic; omit `--stage-model`.

The final 50,000-step experiments 1/2 are recorded in
[`outputs/rollouts/handfeet_vs_frozen_mlp_50000`](outputs/rollouts/handfeet_vs_frozen_mlp_50000).
Both MP4s show 30 seconds at 25 FPS, deterministic mean actors, seed 42, env 0 of
64 ordinary starts. Initial poses, dimensions, masses and inertias were verified
identical across the two policies. `comparison_final_50000.mp4` shows experiment 1
on the left and experiment 2 on the right. `record_rollouts.py` reproduces them;
`summary.json` records all 64 worlds' diagnostics. Both runs had zero hand-box
contacts and no lift/placement in this rollout, despite high upright survival.

```bash
# Robot motion prior only: no task reward and no placement-success termination.
uv run python train.py --task G1-Carry-Box --reward-mode amp \
  --amp-config configs/amp.yaml --num-envs 1024 --no-mixed-precision

# Exactly ten vector steps: 8 parallel envs, 80 collected transitions.
# Small warmup ensures actual updates happen during this short validation run.
uv run python train.py --task G1-Carry-Box --reward-mode amp \
  --amp-config configs/amp_smoke.yaml --num-envs 8 --total-env-steps 80 \
  --buffer-size 256 --min-buffer 8 --batch-size 16 \
  --log-every 1 --save-every 40 --save-replay --no-mixed-precision
```

`configs/amp_control.yaml` is the current **AMP + control-cost** experiment, with
FB and carry task rewards disabled:

```bash
uv run python train.py --task G1-Carry-Box --reward-mode amp \
  --amp-config configs/amp_control.yaml --scene-config configs/carry_box.yaml \
  --num-envs 1024 --total-env-steps 51200000 \
  --buffer-size 2097152 --min-buffer 102400 --batch-size 2048 \
  --updates-per-step 2 --n-step 1 --no-mixed-precision \
  --eval-every 2500 --eval-num-envs 64 --eval-steps 500 \
  --save-every 10240000 --save-replay
```

50,000 vector steps collect 51,200,000 transitions with 1024 environments.
Replay keeps the latest 2,097,152 transitions (2048 vector steps), including raw
AMP pairs and six control terms; tensor storage is about 3.48 GiB on the GPU
with the current 120D observation (3.29 GiB for the older 108D observation).
The run starts from fresh networks/replay. Old AMP checkpoints retain their
original action scales and termination rules when resumed or played.

The control configuration reuses the six active costs from
`I-BFM-Code/checkpoints/ibfm-166m/config.json`: action-rate -0.1, feet orientation
-0.4, ankle roll -4, soft joint-limit violation -10, foot slip -4, and undesired
pelvis/shoulder/hip contact -1. The joint-limit boundary is 95% of the hard range.
Contact gating uses force magnitude >1 N in mjlab; I-BFM's orientation term uses
vertical force >1 N. Torques and extra carrying terms are inactive. Costs are
computed after the physics step and before resetting, and action differences use
the original command units (normalized action multiplied by five). Weighted costs
are normalized by bias-corrected EMA standard deviation (decay .99), without mean
subtraction, keeping penalties negative.

A separate twin scalar critic reuses the FlashSAC critic trunk (two blocks,
width 256). Its target is `normalized_cost + gamma*(1-terminated)*min(Q_aux_next)`;
timeouts bootstrap, and no entropy bonus is added to this target. The scalar head
is unbounded because negative control returns can exceed the AMP critic's C51
support. The actor maximizes
`Q_AMP + .02*stopgrad(mean(abs(Q_AMP))/max(mean(abs(Q_aux)),1))*Q_aux + alpha*entropy`.
This Q balancing substitutes Q_AMP for I-BFM's Q_FB reference; it is an adaptation,
not an exact reproduction of the FB objective. Critic parameters stay frozen
during the actor gradient. Auxiliary weights, targets, optimizer, scheduler and
normalizer are saved with the checkpoint; six raw costs also survive replay saves.

The configuration removes fall/drop resets and gives no fixed fall penalty.
Success resets without a task bonus; the 30 s time limit truncates. Evaluation
starts from ordinary standing spawns, uses deterministic `tanh(mean)` for ten
seconds and records first-fall survival at 1/5/10 s. `evaluation.jsonl` and
TensorBoard record these independently of training episode length. `best` points
to the checkpoint ranked by 5 s survival, then censored upright duration. Periodic
full checkpoints preserve replay; evaluation checkpoints save models only.

For this run D uses Adam 1e-5, reducing the previous run's discriminator learning
rate tenfold. LSGAN, expert gradient penalty (10), frozen expert feature statistics,
and replay reward relabeling are unchanged. This does not adopt I-BFM's
latent-conditional BCE/interpolation-GP discriminator or its FB networks.

The previous bounded SAC action range could not command many expert arm/knee
poses. This configuration multiplies mjlab G1 position-offset scales by five and
clips targets to physical joint limits; zero still commands the default pose.
In a 35-clip training-data audit, command-range coverage rose from 79.1% to 99.86%
of joint-position samples before clipping. Exact tracking is not guaranteed.
Playback restores the saved action configuration automatically.
Network figures for this configuration are in
[`docs/figures/amp_control`](docs/figures/amp_control), including the separate
[`control_critic_io.png`](docs/figures/amp_control/control_critic_io.png).

`--amp-data /path/to/rollouts` overrides `data_root` from the YAML. Supplying AMP
data/config also selects AMP mode when `--reward-mode` is omitted. Existing task
reward training remains available without AMP options. With the original `configs/amp.yaml`, the environment
reward terms are empty and placement success does not terminate episodes. Falling
and timeout termination remain. New carry runs use the 120D actor observation
described below. The discriminator remains 83D per state / 166D per pair.

The initial discriminator deliberately evaluates **robot motion only**. It never
receives box geometry, object motion, goal, contact, action or teacher phase labels.
The experiment can therefore reward similar reaching/carrying poses even when
the robot misses the box. This does not enforce grasping, goal progress, the order
of phases, or a complete carrying task.

The implementation follows the least-squares discriminator and style reward of
[AMP, Equations 7–8](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf):

```text
L_D = mean((D(expert)-1)^2) + mean((D(policy)+1)^2)
      + (10/2) * mean(||gradient_x D(expert)||^2)
r_AMP = reward_scale * clamp(1 - 0.25*(D(policy)-1)^2, 0, 1)
```

The raw score is not a probability. The discriminator runs in FP32 with a
166 → 512 → 256 → 1 ELU MLP and Adam at 1e-4. Its feature normalizer is fitted once
on train expert states and frozen; the same statistics normalize both policy and
expert samples. Gradient penalty is measured on normalized expert feature pairs.
For initial bring-up, use `--no-mixed-precision`: the initial FP16 smoke test
skipped some actor optimizer steps because of gradient overflow. FP32 was verified
to execute all ten actor and twenty critic/discriminator optimizer steps in the
ten-vector-step run.

Each state feature is 83-dimensional, in a heading-aligned robot frame:

| Feature | Dimensions |
| --- | --- |
| Root height above environment origin | 1 |
| Root orientation with heading removed, rotation 6D | 6 |
| Root world linear/angular velocity, rotated into heading frame | 3 + 3 |
| Raw named joint positions and velocities | 29 + 29 |
| Both wrist and ankle link positions relative to root | 12 |
| Adjacent state pair, separated by 20 ms | **166** |

Source joint arrays are reordered by name. Source root angular velocity is in
the local free-joint frame and is converted to world before the common feature
map. Hand/foot positions are recomputed with the same mjlab G1 kinematic model as
the simulator; stored teacher body coordinates are not mixed with mjlab geometry.
Tests compare the expert encoder and GPU environment for identical physical states.

Only `prior_train.txt` and `prior_validation.txt` are consumed. At the tested
snapshot this selects 350 train clips / 193,489 transitions, and 88 validation
clips / 47,100 transitions. Pairs never cross clip boundaries or environment resets.
Phase labels balance sampling, not discriminator inputs. SHA-256 fingerprints
cover the exact motion contents, joint order and kinematic model; the frozen split
paths and fingerprints are saved in checkpoint metadata. Encoded feature caches
are stored under `outputs/amp_cache`.

By default, half of resets sample a **robot-only reference pose and velocity** from
the training clips, aligned to the current heading/spawn. This helps expose the
policy to reaching, lifting and carrying configurations during early exploration.
The randomized box, supports and goal are left unchanged; reference initialization
does not force hand contact or carry the box for the policy. Set
`reference_init_probability: 0.0` in the AMP YAML for ordinary standing starts.
Playback/evaluation always uses the ordinary task start, without these reference
resets.

The original FlashSAC replay stores the SAC data. `AmpReplay` adds the aligned raw
83-dimensional state features on the same ring indices. Before each SAC update,
`AmpTrainer` trains the discriminator, then recomputes that sampled batch's reward
using the current discriminator under `no_grad`. Reward normalization occurs after
relabeling. Running return statistics are updated with the current collection
rewards; they are scale estimates, not a recomputation of all past returns.
AMP mode is restricted to **n-step 1** until intermediate n-step features and
rewards are explicitly supported. Task-reward mode retains the Wolf n-step 3 default.

Checkpoint `amp.pt` contains discriminator, optimizer, normalizer, dataset
fingerprints and update count. With `--save-replay`, new checkpoints place
`amp_replay.pt` and `replay_buffer.pt` together in a `replay/` subdirectory. The
complete directory is published atomically after both files are written. Resume
also supports legacy checkpoints with the replay files at the checkpoint root.
Model metadata is saved before optional replay export. A replay I/O failure is
recorded in `run.json` and `checkpoint/replay_save_failed`, while model playback
and final evaluation remain available; a missing replay is refilled on resume.
Resume restores the saved AMP config and rejects a different feature/dataset
fingerprint. A relocated identical dataset can be selected with `--amp-data`.
`play.py --checkpoint ...` loads only the actor, without a dataset or discriminator.

TensorBoard logs discriminator/regression loss, gradient penalty, expert/policy
scores, held-out expert scores and normalized feature distance. **High discriminator
accuracy is not evidence of good imitation**; ten steps only validate execution
and gradients. Longer rollouts from ordinary starts and visual comparison are
needed to judge whether the motion style has been learned.

The requested ten-step FP32 run is recorded in `outputs/amp_10step_report.json`.
It verified actor 10 / critic 20 / temperature 10 / discriminator 20 optimizer
steps, finite metrics, parameter changes between vector steps 5 and 10, and
checkpoint/replay restoration. Discriminator loss decreased from 1.5264 to 1.0778.
The untrained/very early actor should not be interpreted as a carrying policy.

Architecture images are in `docs/figures/` as PNG and SVG:

- `amp_connections`: whole training graph and optimizer/reward paths.
- `flashsac_network_io`: actor, double critic, target critic and temperature.
- `amp_discriminator_io`: common features, discriminator, loss and reward.

Regenerate from a real AMP checkpoint with the system Graphviz `dot` executable:

```bash
uv run python scripts/draw_amp_architecture.py --checkpoint /path/to/step_N
```

## Train

```bash
uv run python train.py --task G1-Carry-Box --num-envs 1024 \
  --scene-config configs/carry_box.yaml

# Short GPU smoke run with actual gradient updates and frequent timeouts.
uv run python train.py --task G1-Carry-Box --num-envs 4 \
  --total-env-steps 64 --buffer-size 128 --min-buffer 16 --batch-size 16 \
  --episode-length 0.04 --no-mixed-precision --log-every 1 --save-replay

# Exercise the same learner on an existing mjlab task.
uv run python train.py --task Mjlab-Cartpole-Balance --num-envs 16 \
  --total-env-steps 512 --buffer-size 1024 --min-buffer 64 --batch-size 32

uv run python train.py --task G1-Carry-Box \
  --resume logs/flashsac/G1-Carry-Box/<run>/step_<N> \
  --total-env-steps 20000000

uv run tensorboard --logdir logs/flashsac
```

`--total-env-steps` counts transitions across all environments, rounded up to a
whole vector step. `--updates-per-step` counts learner updates per **vector step**.
`--log-every` also uses vector steps; `--save-every` uses total transitions.
Logs, resolved configuration and checkpoints go under `logs/flashsac/<task>/<run>`.
Resume restores algorithm settings, optimizer state, learning-rate schedule and
scene ranges. Simulator episodes restart. Replay is restored if it was saved with
`--save-replay`; otherwise it is refilled before updates restart. Storage size,
batch size, device and compilation settings can be changed on resume.

## Weights & Biases

[The carry project](https://forge.coreweave.com/wandb/bjy6614-postech/humanoid-g1-carry)
contains the four completed main AMP runs and the current object-AMP experiment.
The two 30-second hand/foot versus frozen-MLP rollout videos are attached to their
corresponding completed runs. Imported runs preserve their original scalar tags,
environment-transition counts and source wall times. Default chart axes use
`vector_steps`, so 51,200,000 transitions with 1,024 envs appear at step 50,000.
Historical uploads collect no new system statistics for past training.

Future invocations can log directly to W&B alongside TensorBoard:

```bash
uv run python train.py --task G1-Carry-Box \
  --amp-config configs/amp_object_control.yaml --scene-config configs/carry_box.yaml \
  --num-envs 1024 --total-env-steps 51200000 --no-mixed-precision \
  --wandb-project humanoid-g1-carry --wandb-entity bjy6614-postech
```

`--wandb-name` sets a display name; `--wandb-mode offline` supports local logging
checks. W&B identity and URL are saved in run/checkpoint metadata. Metrics include
reward, discriminator/actor/critic losses, control critic, available evaluations
and MLP diagnostics. Native logging can also collect current system statistics.

An already-running TensorBoard-only job can be followed without restarting it:

```bash
uv run python scripts/sync_wandb.py \
  --run-dir logs/flashsac/G1-Carry-Box/<run> \
  --project humanoid-g1-carry --entity bjy6614-postech \
  --follow --status-file outputs/training_runs/<job>/status.json
```

For completed runs omit `--follow` and `--status-file`. Stable run IDs and per-tag
cursors prevent repeat imports and retain evaluations flushed later at an already
logged step. `wandb_sync.json` records the URL, cursors and completion state.
Config, raw TensorBoard events, available evaluation JSON and explicitly selected
rollout video are uploaded; source code saving is disabled.

## Play / inspect the scene

```bash
# Open paused to inspect the sampled scene before physics advances.
uv run python play.py --task G1-Carry-Box --paused \
  --scene-config configs/carry_box.yaml

# Simulate several randomized scenes with normalized random joint actions.
uv run python play.py --task G1-Carry-Box --agent random --num-envs 4

# Replay a saved FlashSAC actor deterministically (checkpoint directory, not actor.pt).
uv run python play.py --task G1-Carry-Box \
  --checkpoint logs/flashsac/G1-Carry-Box/<run>/step_<N>

# Browser viewer; open the address printed in the terminal.
uv run python play.py --task G1-Carry-Box --viewer viser --paused

# Render the initial scene without a window, then run four control steps.
uv run python play.py --task G1-Carry-Box --headless --steps 4 \
  --num-envs 2 --screenshot outputs/carry_scene.png
```

The default `--agent zero` sets normalized action to zero, commanding the G1's
configured default PD joint targets. It is a scene/control inspection mode;
balance is not guaranteed. `--agent random` samples actions in [-1, 1]. Supplying
`--checkpoint` selects the trained actor automatically, loads its saved scene
ranges and episode length, and uses `tanh(mean)` without exploration noise.
Only actor weights are loaded, so playback allocates no replay buffer or critic.
`--scene-config` and `--episode-length` can override the saved scene settings.

`--viewer auto` uses the native MuJoCo window when a desktop display is available,
otherwise Viser. `--steps N` bounds playback to N vector control steps; omit it
for interactive use. `--headless` requires `--steps`. Screenshots capture the
initial scene and can be used with either viewer. `--seed` changes the initial
sampling sequence; resetting advances it to the next scene.

Native viewer controls:

| Key | Action |
| --- | --- |
| Enter | Reset and resample scenes |
| Space | Pause / resume |
| Right arrow | One control step while paused |
| `,` / `.` | Previous / next environment |
| A | Toggle neighboring environments |
| `-` / `=` | Slow down / speed up |

Viser provides equivalent controls through its browser UI. Simulation stays in
native mjlab and the existing off-policy adapter handles termination/reset logic.
The source support is cyan, the target support green, and the box orange.
Implementation: `play.py` → `src/humanoid_rl/play.py`.

## Randomized carry scene

`configs/carry_box.yaml` contains all sampling ranges, in metres, kg and radians.
The defaults are also declared in `CarrySceneCfg` in
`src/humanoid_rl/tasks/carry_box/env_cfg.py`. Use `--scene-config` to override them.
Every environment independently resamples on every reset:

| Property | Default range |
| --- | --- |
| Box full XYZ dimensions | 0.20–0.40 m on each axis |
| Box mass | 0.5–0.9 kg |
| Box yaw | −0.1–0.1 rad |
| Box XY offset on source | Up to ±0.025 m, clipped to fit |
| Each support full XY dimensions | 0.50–0.60 m on each axis |
| Each support top height | 0.10–0.55 m |
| Source support X / Y | 0.8–1.2 / −0.25–0.25 m |
| Target X / Y offset from source | 0.9–1.6 / −0.6–0.6 m |

Supports are immovable mocap bodies with 2 cm thickness. They are repositioned at
reset and collide with the box, while robot-support contact is disabled to match
the teacher scene. Box pose refers to a centered, axis-aligned body/geometry frame
in this environment; importing I-BFM motions later requires its body-to-geometry
transform. Geometry bounds and cuboid mass/inertia are updated for sampled sizes.
The sampler keeps the box above the source support, the goal above the target,
and supports separated. Range validation rejects layouts that cannot fit the box.

Control is 50 Hz (4 × 5 ms). The built-in mjlab G1 asset and actuator gains/scales
are reused without importing the teacher controller. Action is a 29-dimensional
normalized PD position offset. New carry observations are 120-dimensional:

| Group | Dimensions |
| --- | --- |
| Proprioception: joint offsets/velocities, base linear/angular velocity, gravity | 67 |
| Box pose in robot heading frame: relative position + rotation 6D | 9 |
| Goal position in robot heading frame | 3 |
| Previous action | 29 |
| Both wrist and ankle XYZs relative to root, in the heading frame | 12 |

The key-body order is left wrist, right wrist, left ankle, right ankle. These
are link origins, not palm/sole contact points. Their frame matches the AMP
key-body features: subtract root position and remove root yaw, retaining root
roll/pitch. The extra positions are appended after the original 108D prefix.
Both SAC critics and the auxiliary critic share this 120D observation with the
actor. They add the 29D action when evaluating Q. No stage-model outputs are
included unless `--stage-model` is selected, which appends the 28D frozen context.

`--no-observe-hand-feet` selects the earlier 108D schema for a new ablation run.
Checkpoints record `observe_hand_feet`; playback and resume restore it. Legacy
checkpoints without that field keep their original 108D input. A schema change
requires a fresh run because saved input layers and replay observations differ.

Box dimensions, mass and object velocities are not supplied to the actor. The
critic initially uses the same observation. Simulator state still contains these
quantities for physics and reset logic.

The provisional reward is +1 on placement: box within 20 cm of the goal, speed
below 0.2 m/s, target-support contact, and no hand-box contact. Success terminates
the episode so the reward occurs once. Falling terminates, and the 30 s time limit
truncates. Upright/hold-time refinements and shaping are deferred; this is an
environment scaffold, not a tuned task reward or a trained carrying policy.

The adapter explicitly resets done environments after copying their terminal
observations. Timeouts bootstrap from terminal states using the actual n-step
discount; resets never become replay next-states for the preceding episode.

## FlashSAC defaults

Reference: `wolf-quadruped/rl/wolf-stair-rl/scripts/flash_rl/train.py`,
`agent_config()` and its CLI defaults, plus `wolf-quadruped/rl/FlashSAC/configs/agent/flashSAC.yaml`.
The upstream YAML in this repository is loaded directly for network/optimizer
settings, rather than duplicating an algorithm configuration.

| Setting | Default |
| --- | --- |
| Replay capacity / warmup | 1,000,000 / 100,000 transitions |
| Batch / updates per vector step / n-step | 2048 / 2 / 3 |
| Discount | 0.99 |
| Actor blocks / width | 2 / 128 |
| Critic blocks / width / bins | 2 / 256 / 101 |
| Learning rate | 3e-4 → 1.5e-4, upstream cosine schedule |
| Target tau / actor update period | 0.01 / 2 |
| Temperature / target sigma | 0.01 / 0.15 |
| Reward normalization / normalized return support | Enabled / ±5 |
| Compilation / CUDA mixed precision | Off / on |

Compilation is opt-in with `--compile`, as in Wolf. `--no-mixed-precision` is useful
for bring-up. FlashSAC's `use_amp` flag means automatic mixed precision, not an
adversarial motion prior. Replay is on the training device by default; select
`--buffer-device cpu` or reduce capacity if GPU memory is tight.

## Checks

```bash
uv run python -m unittest discover -s tests -v
```

Tests cover randomized sizes/poses, physically consistent box/goal heights,
partial resets, terminal observations, early-timeout n-step bootstrapping,
playback observation refresh after viewer resets, deterministic actor loading,
expert/simulation AMP feature parity, reference initialization, discriminator
updates/checkpointing and current-discriminator reward relabeling, hand/foot
feature parity and invariance to translation/yaw, and legacy observation schemas.
The environment integration tests skip without CUDA. CLI smoke runs above also
exercise actual FlashSAC updates and checkpointing.
