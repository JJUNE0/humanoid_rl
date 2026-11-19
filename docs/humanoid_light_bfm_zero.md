# Humanoid Light BFM-Zero with CMU and LAFAN

New experiments can use the [common launcher](training.md):
`python3 humanoid.py launch --experiment light_bfm_zero --runtime local`.
That explicit preset uses experimental 2.5 ms physics with eight substeps and
no artificial lying resets. The legacy commands and defaults below retain
their original 5 ms physics. Neither setup certifies long-run stability.

The registered task is `Humanoid-Light-Motion-Prior`. It uses the original
FB-CPR-Aux learner from `third_party/BFM-Zero`, the converted 26-joint robot,
and both retargeted motion collections. Preparation runs only short validation
jobs; the long experiment starts with the command below.

```bash
./scripts/train_humanoid_light_bfm.sh
```

The default first stage is 50 million aggregate environment transitions, not
50 million vector steps. Override it with `--total-env-steps`. This is an
experiment budget, not a claim of convergence or successful acrobatics.

## Data preparation

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/prepare_humanoid_light_bfm.py --workers 2
.venv/bin/python scripts/cache_humanoid_light_bfm.py
```

Inputs are `outputs/humanoid_light_cmu_full/motions` and
`outputs/humanoid_light_lafan_full/motions`; no new motion download or retargeting
is performed. Source files are never rewritten. The prepared manifest and
statistics are in `outputs/humanoid_light_bfm/`.

On a shared machine, run preparation, caching and training sequentially. The
`scripts/run_humanoid_light_bfm_guarded.py` supervisor requires a separate systemd
service with `MemoryMax=24G` or less; it must not share the other experiment's
terminal scope. The current guarded launch uses `MemoryHigh=20G`,
`MemorySwapMax=1G`, `OOMPolicy=stop`, and two screening workers. It waits for
36 GiB available system RAM before each stage and 19 GiB free GPU memory before
training. While running, it stops its own stage if system available RAM stays
below 12 GiB or GPU free memory stays below 2 GiB. This does not reserve resources
against future allocations by other jobs.

The shared-GPU launch limits this training process's PyTorch allocator to 60%
of GPU memory and limits compiler workers to one. MuJoCo/Warp allocations are
outside that allocator limit and are covered by the GPU free-memory monitor.
The network, batch, environment count and replay settings remain the task
defaults. Expert archives are memory mapped while the dataset is assembled,
reducing overlap between cached sources and copied trajectory storage.

The preparation script screens stored poses and quarter-frame interpolants
against the original mesh hulls, native joint ranges, and stored joint speeds.
It uses the existing conservative distance checks and 31 recorded structural
assembly exclusions. Rejected intervals split a source into independent runs,
with a two-frame boundary guard. A run must retain at least 25 frames. It never
bridges a bad interval, joins source files, pads the end, or invents a motion.
The ground tolerance is 1 mm and the body clearance threshold is 50 micrometres;
these match screening criteria rather than an 8 mm clearance guarantee.

Timing-unverified CMU files are excluded. LAFAN subject 5 and CMU subject IDs
divisible by 10 are reserved for validation. This separates capture subject IDs;
CMU may contain repeat performers under different IDs. Terrain, props and
partners are not reconstructed. Passing geometry checks does not prove dynamic
tracking or correct support timing, and it does not remove all source-tracking
errors or optimizer stalls. The original audits remain available for review.

Both ordinary expert batches and reference-state resets use 50:50 source weights,
then uniform accepted chunks within each source. Ten-second chunks are kept
separate. Expert latent rollouts can end before five seconds: they use only the
actual remaining states and sample another reference at the real end. No padded
states are sent to the discriminator or backward network. This is a local
sampling extension; network architectures and the FB/CPR/Aux update equations
are unchanged.

## Robot and features

The assets are copied into `src/robots/humanoid/humanoid_light/assets/`, so
training does not depend on an external IsaacLab checkout. Capsule visuals do
not collide or contribute mass; original collision hulls and inertial properties
are retained. Nonstructural self contacts are enabled, and the recorded assembly
exclusions remain active.

Ordinary joints use source PD gains/effort limits. Ankle and torso pitch/roll
pairs use source motor-coordinate feedback and the transposed coupling Jacobian,
with effort clipping after mapping into joint torque. Policy commands are joint
targets transformed into motor coordinates. Zero holds the default pose; -1/+1
reaches each native lower/upper joint limit, including asymmetric ranges. This
action adapter differs from the source IsaacLab motor-target action slots.
Source delay defaults are zero. Source speed limits are used by reference QC
and diagnostics; MuJoCo does not reproduce IsaacLab's solver-side hard velocity
cap. There is no hidden projection or clipping of simulated joint velocities.

Expert and online features share the same FK/link-origin convention, native
joint/body order, metres/radians/seconds, and wxyz root quaternions:

| Observation | Size |
| --- | ---: |
| Joint positions/velocities, gravity, root angular velocity | 58 |
| Root height and heading-relative body positions/orientations/velocities | 463 |
| Previous normalized action | 26 |
| Four-step state/action history | 336 |

These are full robot-state BFM features. The retargeting objective's velocity,
pose, contact and collision terms are not automatically learning losses.
Auxiliary costs cover action rate, supported-foot orientation/slip, joint limits,
and head contacts. Hands, knees and torso are allowed to support floor motions.

## Resource settings and checkpoints

The task overlay is
`src/humanoid_rl/algorithm/bfm_zero/configs/humanoid_light.json`.

| Setting | Value |
| --- | --- |
| Forward/actor width and residual layers | 2048 / 6, original architecture |
| Latent size / backward width / discriminator width | 256 / 256 / 1024 |
| Environments / batch / updates per vector step | 1024 / 1024 / 16 |
| Policy control | 50 Hz; 5 ms physics with decimation 4 |
| Online replay | 640,000 transitions on CPU, about 4.90 GiB |
| Expert storage | CPU; sampled features transferred to GPU |
| Compilation / mixed precision | Enabled / disabled |
| Replay in checkpoints | Disabled |
| Save interval / retention | 2,048,000 transitions / latest 2 |

Full network/optimizer checkpoints are several GiB each. Allow space for the
two retained checkpoints plus one temporary checkpoint during publication, and
for the precomputed expert cache. Without saved replay, resume restores the
networks/optimizers and refills replay from fresh episodes:

```bash
./scripts/train_humanoid_light_bfm.sh --resume /absolute/path/to/checkpoints/step_N \
  --total-env-steps 100000000
```

Do not use the development smoke checkpoint as an experiment initialization:
it uses a small development manifest and does not represent a learned policy.
Other GPU jobs share the same 3090; measure available memory and throughput at
launch. Short preparation measurements are not a convergence-time estimate.

## Tracking evaluation and viewing

```bash
.venv/bin/python -m humanoid_rl.algorithm.bfm_zero.evaluate \
  --checkpoint /absolute/path/to/checkpoints/step_N

.venv/bin/python play.py --task Humanoid-Light-Motion-Prior --agent trained \
  --checkpoint /absolute/path/to/checkpoints/step_N
```

The evaluator infers reference latents using the original tracking inference,
initializes the robot at each selected reference's first state, and records joint
RMSE, root-height/tilt error and torque saturation for held-out CMU/LAFAN files.
It is a reference-initialized tracking diagnostic, not an autonomous takeoff or
recovery success metric. Playback currently samples random latent commands.
Automatic periodic tracking evaluation remains unavailable (`eval_every=0`);
use the standalone evaluator on a saved checkpoint.

## Verification

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -p 'test_*bfm*.py' -v
```

Light tests verify independent source coupling equations and power consistency,
full-range action targets, expert/online feature parity under arbitrary root
rotations, active self contacts/assembly filters, gap-safe sequence sampling,
source weights, and real-boundary handling for short expert rollouts.
