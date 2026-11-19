# BFM-Zero on mjlab G1 and H1

This package connects the original **FB-CPR-Aux** learning core to the original
G1 model or the copied H1 model and local mocap data. It imports the agent, networks, normalizers and
replay classes from `third_party/BFM-Zero/humanoidverse/agents`. It does not
import that repository's Isaac environment or install its simulator dependency
set. The original learning objectives and discriminator gradient penalty remain
in the upstream source.

## What motion_prior means

`H1-Motion-Prior` names the environment for humanoid pretraining guided by mocap.
The simulator supplies physical state transitions. Expert mocap sequences guide
the latent-conditioned discriminator, while online interaction trains F/B,
the actor, the style critic and the auxiliary control critic.

There is no learned one-step dynamics/world model and no task-specific tracking
reward in this environment. F/B factorizes long-term discounted state visitation;
it does not predict the next robot pose. The motion dataset constrains the
behavior distribution and provides reference states for initialization and
expert-derived latent commands.

## Original G1 baseline

```bash
.venv/bin/python train.py --algorithm bfm_zero --task G1-Motion-Prior
```

Task selection layers `configs/g1.json` over `configs/train.json`: 1024 envs,
384M transitions, batch 1024, 16 updates per vector step, 2048-wide six-layer
networks, z=256 and a 5.12M-transition CPU replay (about 40.96 GiB). It preserves
the original learning core and auxiliary weights, including ankle roll.
`configs/g1_smoke.json` overrides scale for a 4-env/10-step connection check.

G1 uses the original copied MJCF in `robots/humanoid/g1/bfm_zero_assets`, plus
original hard-waist PD gains, joint limits, effort limits, armature and joint
friction settings. The action mapping, virtual head, 64/463/29/372 feature
dimensions, reset mixture, no-fall termination and >500 timeout match the
original conventions. Observation noise and enabled domain randomizations
have native mjlab counterparts. Current differences are documented in
`tasks/motion_prior/agents/README.md`, including missing EMD-based prioritization.

Preparation, already completed locally:

```bash
.venv/bin/python scripts/prepare_g1_bfm_zero.py
/tmp/bfm-zero-original-audit-venv/bin/python scripts/export_g1_bfm_motions.py
```

The second command uses the original MotionLibRobot in the separate upstream
audit environment. It exports all 862 original clips (430,138 frames at 50 Hz),
including filtered velocities and original style features, into independent
numeric NPZs under `humanoid_retarget/datasets/lafan1/retargeted/g1/bfm_zero_50hz`.
The mjlab training environment needs neither Isaac nor Open3D. Checkpoints
include effective robot/control settings; the latest 3 are retained by default.

### RTX 3090 / 64 GB RAM profile

```bash
.venv/bin/python train.py --algorithm bfm_zero --task G1-Motion-Prior \
  --train-config src/humanoid_rl/algorithm/bfm_zero/configs/g1_3090.json
```

This profile reduces CPU replay capacity to 1,280,000 transitions (10.24 GiB),
enables the original agent's `torch.compile` path, saves a checkpoint every
1,024,000 transitions and retains the latest two checkpoints. It inherits the full G1
motion dataset, environment settings, 1024 environments, batch 1024, 16 updates
per vector step, 2048-wide six-layer networks and 384M-transition budget.
The smaller replay retains fewer past transitions and can affect learning;
matching the original training result still needs a full experiment.

Each online transition uses 8,590 bytes: current and next observations each
contain 928 float32 values (64 state + 463 privileged + 29 last action + 372
history), plus action29, latent256, six auxiliary costs and two boolean flags.
`DictBuffer` allocates its entire capacity on the first insertion, so reducing
`num_envs` or `batch_size` alone does not reduce this CPU allocation.

| Replay capacity | CPU tensor storage |
| --- | --- |
| 5,120,000 | 40.96 GiB |
| 2,560,000 | 20.48 GiB |
| 1,280,000 | 10.24 GiB |
| 640,000 | 5.12 GiB |

These sizes exclude the simulator, networks, optimizer states, expert motions
and other applications. Run one training process at a time and complete the
CUDA environment tests before starting it, since those tests also create
simulators and load motions. The profile keeps replay on CPU to leave GPU
memory for simulation and learning. Mixed precision does not change the replay
tensors' float32 storage in this implementation.

On the local RTX 3090 / i9-11900K / 64 GB machine, an initial validation run
with compilation disabled, using the
full replay allocation, full dataset and unchanged network/batch/update sizes
completed 11,264 transitions, 32 optimizer updates and checkpoint saving. Peak
process RSS was 16.68 GiB, minimum system available RAM was 42.77 GiB, and the
observed total GPU usage reached about 16.73 GiB (PyTorch peak reserved memory
was 13.72 GiB; simulator allocations also use GPU memory). The result and logs
are under `outputs/g1_3090_validation/`. This checks initial memory fit and
finite updates, rather than long-run convergence or eventual learning quality.

For resumable experiments, add `--save-replay`. Model and optimizer files from
the full-size validation checkpoint occupy 6.44 GiB; a full replay adds 10.24
GiB. Retaining two complete checkpoints plus one temporary checkpoint during
atomic publication therefore needs about 50.03 GiB at peak. Existing runs,
compiler caches and logs consume additional disk space.

The [paper's training table](https://arxiv.org/html/2511.04131v1#A2.SS1) uses
192M transitions / 3M agent updates. The
[released example configuration](https://github.com/LeCAR-Lab/BFM-Zero/blob/main/humanoidverse/train.py)
uses 384M transitions / 6M updates. The upstream README suggests checking motion
EMD after 50M--100M transitions. The current mjlab integration has no automatic
EMD/tracking evaluator, so those milestones require separate evaluation.

The first local steady intervals measured about 166 transitions/s without
compilation and 203 transitions/s with compilation, using the same full model,
batch size and 16 updates per vector step. The compiled run completed 30,720
transitions / 336 updates without nonfinite metrics during this measurement.
Extrapolating 203 transitions/s gives roughly 2.8 days for 50M transitions,
10.9 days for the paper's 192M and 21.8 days for the released example's 384M;
these short measurements exclude checkpoint/evaluation overhead and do not
establish convergence. Launch metadata, measured throughput and the current
service name are recorded in `logs/bfm_zero/launches/g1_3090_latest.json`.

## Follow the code

1. `humanoid_rl/train.py` selects `--algorithm bfm_zero --task H1-Motion-Prior`.
2. `configs/train.json` contains the training defaults. `config.py` constructs
   the original `FBcprAuxAgentConfig`, including the observation input filters.
3. `tasks/motion_prior/env_cfg.py` creates the mjlab environment using
   `robots/humanoid/h1/mjlab.py`. Robot selections and reward defaults live in
   `tasks/motion_prior/agents/{h1,g1}/`; `mdp.py` shares the cost/reset calculations.
4. `dataset.py` reads local H1 NPZ motions, resamples them from 40 to 50 Hz,
   computes matching forward kinematics, caches features and builds the upstream
   `TrajectoryDictBuffer`. Samples never cross clip/reset boundaries.
5. `env/bfm_features.py` defines the same proprioceptive/privileged features for
   expert data and online mjlab states. `env/bfm_adapter.py` preserves terminal
   observations before partial resets.
6. `runner.py` collects physics transitions into the original `DictBuffer`,
   calls `agent.update()`, logs metrics and saves the original agent format.
   The actual F/B, D, Q and actor updates are in the upstream agent files.

## Inputs and outputs

The copied H1 has 19 actuated joints and 20 robot bodies. These dimensions are
derived from that model, rather than reusing G1's 29-joint observation sizes.

| Observation | Dimension | Contents |
|---|---:|---|
| `state` | 44 | q minus neutral q (19), q velocity (19), projected gravity (3), body-frame root angular velocity times 0.25 (3) |
| `last_action` | 19 | Previous policy action multiplied by `normalize_action_to` (5), as in original BFM |
| `history_actor` | 252 | Four preceding observations and their previous actions, newest first, grouped as actions/angular velocity/q/q velocity/gravity |
| `privileged_state` | 298 | Root height; root-relative body XYZ; body rotation tangent/normal (X/Z axes); body linear/angular velocities in the heading frame |

| Network | Input | Output with default z_dim=256 |
|---|---|---|
| Actor | state + last action + history + z | 19 normalized joint commands |
| Backward B | state + privileged state | 256D embedding |
| Forward F | all observations + action + z | Two 256D successor representations |
| Discriminator D | state + privileged state + z | One logit/probability |
| Style critic Q_D | all observations + action + z | Two scalar style values |
| Auxiliary critic Q_R | all observations + action + z | Two scalar control-cost values |

Expert data need no recorded actions. The expert B/D filters use only `state`
and `privileged_state`; policy histories and actual actions come from online
interaction. The discriminator's WGAN-style gradient penalty defaults to 10,
as in the original FB-CPR-Aux configuration.

## Run

```bash
uv sync --extra bfm-zero

# Inspect the effective settings without creating a simulator.
.venv/bin/python train.py --algorithm bfm_zero --task H1-Motion-Prior \
  --train-config src/humanoid_rl/algorithm/bfm_zero/configs/smoke.json --print-config

# Ten vector control steps, four environments, ten updates of all learning modules.
.venv/bin/python train.py --algorithm bfm_zero --task H1-Motion-Prior \
  --train-config src/humanoid_rl/algorithm/bfm_zero/configs/smoke.json

# Show the H1 scene with its neutral PD command.
.venv/bin/python play.py --task H1-Motion-Prior --agent zero

# Replay a saved BFM actor. Supply an actual printed checkpoint directory.
.venv/bin/python play.py --task H1-Motion-Prior --checkpoint <checkpoint-directory>

# Continue a checkpoint, preserving its training/model/data configuration.
.venv/bin/python train.py --algorithm bfm_zero --task H1-Motion-Prior \
  --resume <checkpoint-directory> --total-env-steps <larger-total>
```

Playback uses deterministic actor actions with random latent commands held for
`update_z_every_step` steps. It is not a reference-tracking evaluation. The
checkpoint contains the model/config/optimizer files from upstream plus
`run.json`; each run also saves the effective `train.json`, TensorBoard events
and `metrics.jsonl`. Replay export is enabled with `--save-replay`; if replay is
absent, resume refills it from fresh simulator episodes.

## Configuration and current scope

`configs/train.json` follows the original residual widths (2048), six layers,
z_dim=256, batch=1024, discount=0.98 and 16 updates per vector step. The training
budget is the paper's 192M transitions. Replay defaults to CPU for this machine;
the 5.12M-transition capacity is approximately 29 GiB with the current schema.
It is a full-size configuration and has not been validated by a long H1 run.
`smoke.json` reduces network widths, batch/replay size and data selection to
verify the code path. It does not establish learned behavior quality.

The H1-specific choices are explicit:

- PD gains/neutral pose are mapped from Unitree's H1 example and recorded in
  `robots/humanoid/h1/control.json`. The original BFM action rule is
  `q_target = q_default + 0.25 * clip(action * 5, -5, 5) * effort_limit / Kp`.
  PD torques are capped; position targets are not clipped. Torque caps come
  from the copied MJCF and explicit PD runs every 5 ms.
- The control period is 20 ms. Mocap quaternions use shortest-path SLERP;
  root/joint positions and supplied qvel are interpolated. No physical motion
  correction is applied.
- Experts are split into 10-second clips. Subject 5 is held out by default;
  explicitly selecting files is a small-data/overfit mode. The smoke setup uses
  `walk1_subject1.npz` and caps source frames at 600.
- Episodes last 10 seconds. Physical fall termination is off, matching the
  original BFM configuration; fall fraction is logged separately. Resets use
  random expert states plus 30% reference lie-down states, preserving sampled
  joints/velocities and rotating the root +/-90 degrees around world X at 0.5 m.
- Auxiliary penalties cover action rate, feet orientation, joint position
  limits, slippage and undesired contacts. H1 has no ankle-roll joints, so that
  G1-specific penalty is omitted. There is no carry-box/task reward.
- Foot-orientation costs use upward force on the robot, as in the original
  `Fz > 1` gate. mjlab netforce points primary->secondary, so the configured
  conversion is `-sensor_force_z > 1`. Slippage retains a force-magnitude gate.
- New runs save the resolved robot/reward/observation settings in
  `train_config.motion_prior_agent`. Older checkpoints without that snapshot
  restore their magnitude contact gate, normalized action observations,
  original history timing, neutral 0.25 m lying initialization and saved
  controller (uniform position scale and target clipping).
- Observation noise, domain randomization, held-out EMD/long-horizon tracking
  evaluation and goal/reward-prompt playback interfaces are not connected in
  this initial H1 integration. `eval_every` must remain zero.

The upstream model retains its goal/reward/tracking inference methods; the
current runner and playback CLI expose training and random-latent playback.
