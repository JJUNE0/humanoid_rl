# Robot-specific motion-prior settings

`get_agent_config("H1-Motion-Prior")` loads `h1/config.py` and `h1/rewards.json`.
`G1-Motion-Prior` loads the same interface in `g1/`. Both tasks are registered.
The G1 motion-prior task uses `robots/humanoid/g1/bfm_zero.py`, the copied original
G1 MJCF and original `g1_29dof_hard_waist` controller. Carry-box continues using
its existing native mjlab G1/controller definitions.

Robot MJCF, canonical joint order, PD gains and neutral pose belong in
`src/robots/humanoid/{h1,g1}`. The task profiles import joint names from there,
select foot bodies/undesired contacts/ankle-roll joints, and configure resets
and action observation conventions. `env_cfg.py` builds sensors/actions from
the selected profile; `mdp.py` contains the shared calculations.

`rewards.json` contains the active penalty weights, contact-force threshold,
soft joint-limit fraction and foot-orientation contact rule. G1 includes the
ankle-roll penalty; the copied 19-joint H1 has no ankle-roll joints, so enabling
that term is rejected. Zero-weight torque terms in original BFM are omitted.

Weights come from `third_party/BFM-Zero/humanoidverse/train.py`'s
`aux_rewards_scaling`. The environment reward YAML has different weights and
a curriculum, but the original aux critic receives **unweighted raw costs**
and applies `aux_rewards_scaling` itself. We retain its reward normalizer and
actor regularization coefficients in the original FB-CPR-Aux agent.

## Experiment overrides and saved runs

Algorithm/network/replay defaults stay in `algorithm/bfm_zero/configs/train.json`.
`configs/g1.json` overlays original G1 data/noise/DR settings before experiment
overrides, selected automatically by `--task G1-Motion-Prior`.
Robot reward defaults are applied before an experiment's optional overrides:

```json
{
  "auxiliary_rewards": {"penalty_slippage": -3.0},
  "auxiliary_reward_parameters": {"contact_force_threshold": 2.0}
}
```

Pass this with `--train-config`. Other robot rewards keep their defaults.
The resolved `motion_prior_agent` snapshot and auxiliary weights are saved in
the effective `train.json` and checkpoint `run.json`. Resume/play restore that
snapshot instead of loading changed defaults. Old checkpoints without a
snapshot explicitly restore the previous H1 observation/reset/contact rules
and use their saved controller, preserving the old action mapping.

## Comparison with original BFM-Zero

The comparison is between our native mjlab implementation and the untouched
BFM-Zero checkout pinned at `05ad6119a25b06c341c191a72ccf936e987f59cf`.
It is not a difference between G1 and H1 reward theory, or an I-BFM comparison.

- Original foot orientation: `contact_forces[..., 2] > 1`. The earlier H1 code
  used `norm(force) > 1` instead. New profiles select vertical force. mjlab's
  world-frame netforce points primary->secondary; standing H1 measurements
  were approximately -264 N and -247 N on Z. Thus the equivalent gate is
  `-sensor_force_z > threshold`, not a literal positive raw-sensor comparison.
  Horizontal-only contact is excluded from orientation cost; slippage still
  uses force magnitude, matching its separate original formula.
- Original `last_action` and action history contain the normalized policy
  action multiplied by 5. Earlier H1 observations used the raw policy output.
  Original history stores prior states with the action already present in
  those observations, rather than pairing a prior state with the next action.
- Original `action_rescale=True` multiplies position offsets by each joint's
  `effort_limit / Kp`. Earlier H1 omitted this factor and clipped targets to
  joint limits. New H1 control settings reproduce the original equation and
  cap PD torque, using H1-specific gains/torque limits.
- Original lying resets retain the motion's joints/velocities, rotate the
  reference root +/-90 degrees about world X and use 0.5 m root height. Earlier
  H1 used a neutral joint pose at 0.25 m with a world-Y rotation. New profiles
  use the reference initialization; those choices are explicit profile fields.

The original G1 MuJoCo environment can be exercised without Isaac Sim:

```bash
/tmp/bfm-zero-original-audit-venv/bin/python scripts/audit_bfm_zero_original.py
```

The temporary audit interpreter contains the upstream MuJoCo dependencies;
the normal project interpreter need not install them. The script runs 10
physics/control steps with small deterministic actions, checks action units
and history alignment, and writes `outputs/bfm_zero_original_audit/report.json`.
This is an environment smoke check, not an original-policy training result.
`tests/test_motion_prior_agents.py` additionally executes the original cost
and PD function bodies against the same inputs, including side/downward
contacts, to verify our formulas and action processing.

G1 additionally extends physical state features with the original virtual head
(torso + 0.35 m), giving 64/463/29/372 actor/privileged/action/history dimensions.
It uses original >500 timeout semantics, all 862 original 10-second LAFAN clips,
and original MotionLibRobot velocity filtering/interpolation exported at 50 Hz.
Default G1 training enables original noise magnitudes and native mjlab
counterparts of mass/inertia scaling, torso COM offsets, geometry friction,
per-reset joint offsets and periodic velocity pushes.

Known remaining differences: G1 uses MuJoCo Warp contact dynamics rather than
Isaac/PhysX and the original MJCF rather than loading USD. Friction has a single
MuJoCo coefficient rather than separate static/dynamic PhysX coefficients;
joint friction/contact solver semantics differ. Reference resets choose the
exported 50 Hz frames rather than continuously interpolated times. The original
EMD tracking evaluator and its motion-priority feedback are not connected, so
expert clip sampling remains uniform. Compilation is off for bring-up, replay
is on CPU and replay snapshots are disabled; G1 keeps the latest 3 checkpoints
to bound disk usage. H1 still has noise/DR disabled and a 500-step timeout.
These checks do not establish learning quality or full Isaac-to-mjlab equivalence.
