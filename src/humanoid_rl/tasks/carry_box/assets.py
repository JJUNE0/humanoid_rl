"""Native mjlab scene entities; no dependency on the I-BFM runtime."""

import mujoco


def box_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="box")
    body.add_freejoint(name="box_joint")
    body.add_geom(
        name="box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.15, 0.15, 0.15),
        mass=0.7,
        friction=(0.9, 0.01, 0.001),
        contype=3,
        conaffinity=3,
        condim=3,
        rgba=(0.8, 0.45, 0.15, 1),
    )
    return spec


def support_spec(rgba=(0.15, 0.6, 0.65, 1)) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    # Mocap bodies are immovable under contact but can be repositioned at reset.
    body = spec.worldbody.add_body(name="support", mocap=True)
    body.add_geom(
        name="support_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.25, 0.25, 0.01),
        friction=(0.9, 0.01, 0.001),
        # Matches I-BFM's box-support enabled / robot-support disabled scene.
        contype=2,
        conaffinity=2,
        condim=3,
        rgba=rgba,
    )
    return spec
