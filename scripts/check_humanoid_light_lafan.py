"""Check that original LAFAN1 inputs exist for a future Light retargeter.

This is an input/model compatibility audit, not a completed retargeter. No G1
motion, FK features, or inverse-kinematics offsets are copied to Light.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def header(path):
    names, channels, dt, frames = [], 0, None, None
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if line.startswith(("ROOT ", "JOINT ")):
                names.append(line.split()[1])
            elif line.startswith("CHANNELS "):
                channels += int(line.split()[1])
            elif line.startswith("Frames:"):
                frames = int(line.split(":")[1])
            elif line.startswith("Frame Time:"):
                dt = float(line.split(":")[1])
                values = np.fromstring(next(stream), sep=" ")
                if len(values) != channels or not np.isfinite(values).all():
                    raise ValueError(f"Invalid first frame: {path}")
                break
    if not dt or not frames or len(names) != len(set(names)):
        raise ValueError(f"Invalid BVH header: {path}")
    return {"joint_names": names, "channels": channels, "frames": frames, "fps": 1 / dt}


def main():
    root = ROOT / "src/humanoid_retarget/datasets/lafan1"
    original = json.loads((root / "raw_manifest.json").read_text())
    g1 = json.loads((root / "retargeted/g1/bfm_zero_50hz/manifest.json").read_text())
    subset = sorted({re.sub(r"_clip\d+$", "", Path(c["file"]).stem) for c in g1["clips"]})
    files = []
    semantic_names = {"Hips", "Spine2", "Head", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
                      "RightUpLeg", "RightLeg", "RightFoot", "RightToe", "LeftArm", "LeftForeArm",
                      "LeftHand", "RightArm", "RightForeArm", "RightHand"}
    for entry in original["files"]:
        path = root / entry["path"]
        meta = header(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"] or meta["frames"] != entry["frames"]:
            raise ValueError(f"Manifest mismatch: {path}")
        missing = sorted(semantic_names - set(meta["joint_names"]))
        if missing:
            raise ValueError(f"Semantic keypoints missing in {path}: {missing}")
        files.append({"name": path.stem, "path": str(path.relative_to(ROOT)),
                      "sha256": digest, "bytes": path.stat().st_size, **meta,
                      "used_in_existing_bfm_zero_subset": path.stem in subset})
    missing = sorted(set(subset) - {f["name"] for f in files})
    if missing:
        raise ValueError(f"BFM subset raw inputs unavailable: {missing}")
    asset = ROOT / "outputs/humanoid_light_conversion/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(asset))
    correspondence = {"Hips": "pelvis_link", "Spine2": "torso_roll_link", "Head": "head_link"}
    for side in ["Left", "Right"]:
        robot_side = side.lower()
        for human, robot in [("UpLeg", "hip_yaw"), ("Leg", "knee"), ("Foot", "ankle_roll"),
                             ("Arm", "shoulder_yaw"), ("ForeArm", "elbow"), ("Hand", "wrist")]:
            correspondence[side + human] = f"{robot_side}_{robot}_link"
    for name in correspondence.values():
        model.body(name)  # Verify every semantic target is present.
    selected = [f for f in files if f["used_in_existing_bfm_zero_subset"]]
    target_frames = sum(int(np.floor((f["frames"]-1) / f["fps"] * 50 + 1e-8)) + 1 for f in selected)
    source_frames = sum(f["frames"] for f in selected)
    actor_dim, privileged_dim = 2 * model.nu + 6, 15 * (model.nbody - 1) - 2
    report = {
        "status": "original_inputs_and_target_model_compatible_custom_retargeter_required",
        "retargeted_trajectories_generated": False,
        "raw_source": {k: v for k, v in original.items() if k != "files"},
        "raw_sequence_count_verified": len(files), "raw_frame_count_verified": sum(f["frames"] for f in files),
        "raw_bytes": sum(f["bytes"] for f in files), "raw_sha256_verified": True,
        "bfm_zero_subset": {"sequences": len(subset), "names": subset, "missing_originals": missing,
                  "original_bvh_frames": source_frames, "original_bvh_duration_hours": sum(f["frames"] / f["fps"] for f in selected) / 3600,
                  "existing_g1_training_clips": len(g1["clips"]), "estimated_50hz_frames_before_cutting": target_frames,
                  "note": "Source names come from the existing official-G1 BFM-Zero export manifest. Retarget/cut durations may differ."},
        "target": {"mjcf": str(asset.relative_to(ROOT)), "nq": model.nq, "nv": model.nv,
                   "actuated_joint_count": model.nu, "bodies_excluding_world": model.nbody - 1,
                   "joint_names": [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE],
                   "bfm_actor_state_dim": actor_dim, "bfm_privileged_state_dim": privileged_dim},
        "candidate_orientation_correspondence_not_calibrated_position_targets": correspondence,
        "external_retargeting_interface_verified": {
            "tool": "GMR", "human_input": "bvh_lafan1", "built_in_humanoid_light_config": False,
            "sources": ["https://github.com/YanjieZe/GMR#retargeting-from-bvh-lafan1-nokov-to-robot",
                        "https://github.com/YanjieZe/GMR/blob/master/general_motion_retargeting/params.py",
                        "https://github.com/YanjieZe/GMR/blob/master/DOC.md"]},
        "required_work": [
            "Register this MJCF and calibrate neutral-pose link orientation/position offsets; G1 offsets cannot be reused.",
            "Decode original BVH channel rotations, convert centimeters to meters and Y-up to Z-up, and fit segment lengths to Light.",
            "Fit floating root and all 26 robot coordinates with joint limits and temporal regularization.",
            "Use source foot contact labels with ground, no-slip and self-collision screening; review unreachable poses per clip.",
            "Validate several short walk/dance/fight clips before processing all selected source sequences.",
            "Resample to 50 Hz with quaternion interpolation and regenerate consistent velocities, FK and Light-specific BFM features.",
            "Implement the Light control/environment profile and validate dynamic tracking before full BFM training."],
        "estimated_uncompressed_float32_bytes": {
            "qpos_qvel_only_for_selected_raw_durations": target_frames * (model.nq + model.nv) * 4,
            "bfm_actor_plus_privileged_features": target_frames * (actor_dim + privileged_dim) * 4,
            "note": "Estimates for current feature schema, before compression/cutting/filtering; no new dataset is generated."},
        "files": files,
    }
    output = ROOT / "outputs/humanoid_light_motion_audit/lafan_feasibility.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"verified_raw_sequences": len(files), "bfm_subset_originals_found": len(subset),
                      "missing": missing, "subset_frames": source_frames, "estimated_50hz_frames": target_frames,
                      "raw_bytes": report["raw_bytes"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
