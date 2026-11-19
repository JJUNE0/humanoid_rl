# Unitree H1 (19 joints)

`h1.xml` is an actual copied MJCF model, with 21 local STL meshes in `assets/`.
Its shared scene and mimic-site definitions are copied into `common/`. The only
MJCF change is rewriting `../common/...` includes to `common/...`; no masses,
joint axes, limits, collisions or actuator parameters were changed.

The source is `robfiras/loco-mujoco-models` at revision
`1aeea8b21cb160ff9f24d9ee8198661c9591f245`. `manifest.json` records source URLs,
Git blob IDs, original SHA-256 hashes and local hashes. Copyright/license notices
are retained in the XML and accompanying `LICENSE.*` files.

```python
import mujoco
from robots.humanoid.h1 import get_spec

model = get_spec().compile()
data = mujoco.MjData(model)
print(model.nq, model.nv, model.nu)  # 26, 25, 19
```

The floating root uses xyz + wxyz. `config.JOINT_NAMES` defines the 19 hinge
joints in MuJoCo state order; the copied actuator order matches this order.
`get_spec()` includes the source floor and lights. Its actuators are torque
motors. `mjlab.py` builds the H1 entity for `H1-Motion-Prior`: it removes the
source floor/lights/torque actuators from a fresh spec and creates explicit PD
motors with the source torque limits. The copied MJCF remains unchanged.
`control.json` records the mapped Unitree H1 PD gains, neutral pose and normalized
BFM action scaling.

Matching H1 motions are stored locally under
`src/humanoid_retarget/datasets/lafan1/retargeted/h1` and
`src/humanoid_retarget/datasets/loco_mujoco/mocap/h1`. No LocoMuJoCo Python
installation, third-party asset path or network connection is needed to load
this model.
