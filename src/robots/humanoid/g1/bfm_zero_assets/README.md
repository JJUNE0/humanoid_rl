# Original BFM-Zero G1 assets

`g1_29dof.xml` and its 36 referenced STL meshes are copied from
`third_party/BFM-Zero/humanoidverse/data/robots/g1`, pinned at
`05ad6119a25b06c341c191a72ccf936e987f59cf`. `manifest.json` records source-file
SHA-256 hashes; source files are copied unchanged. `LICENSE.BFM-Zero` retains
the upstream license notice.

`control.json` is generated from `g1_29dof_hard_waist.yaml`. The runtime
`g1/bfm_zero.py` constructs a fresh spec, removes the XML motors and installs
native mjlab explicit PD motors. It applies the original training joint limits,
armature/friction, zero passive damping and per-joint effort limits, and chooses
robot contact friction over the unit-friction floor. These runtime changes are
separate from the copied source asset. The original USD/PhysX scene is not loaded.

The original "hard waist" configuration has three waist joints with Kp=300;
it does not remove the waist roll/pitch DOFs. All 29 joints remain actuated.
Original style observations add a virtual head 0.35 m above the torso; it is a
feature point and adds no new body mass.

Regenerate local copies using `scripts/prepare_g1_bfm_zero.py`.
