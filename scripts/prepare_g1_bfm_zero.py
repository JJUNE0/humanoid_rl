"""Copy original BFM-Zero G1 assets/control parameters into the robot package."""

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    upstream = ROOT/"third_party/BFM-Zero"
    source = upstream/"humanoidverse/data/robots/g1"
    target = ROOT/"src/robots/humanoid/g1/bfm_zero_assets"
    target.mkdir(parents=True, exist_ok=True)
    names = ["g1_29dof.xml"]
    tree = ET.parse(source/names[0])
    names += ["meshes/"+node.attrib["file"] for node in tree.findall("asset/mesh")]
    manifest = {}
    for name in names:
        destination = target/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, destination)
        manifest[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    shutil.copy2(upstream/"LICENSE", target/"LICENSE.BFM-Zero")
    path = upstream/"humanoidverse/config/robot/g1/g1_29dof_hard_waist.yaml"
    robot = yaml.safe_load(path.read_text())["robot"]
    joints, control = robot["dof_names"], robot["control"]
    def gains(mapping):
        result = {}
        for name in joints:
            matches = [value for pattern, value in mapping.items() if pattern in name]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous original gain: {name}")
            result[name] = matches[0]
        return result
    profile = {
        "source": "BFM-Zero g1_29dof_hard_waist.yaml; original G1 MJCF and Isaac training joint parameters",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "joint_names": joints, "body_names": robot["body_names"],
        "action_scale": control["action_scale"], "normalize_action_to": control["normalize_action_to"],
        "action_rescale": True, "clip_position_targets": False,
        "stiffness": gains(control["stiffness"]), "damping": gains(control["damping"]),
        "default_joint_pos": robot["init_state"]["default_joint_angles"],
        "root_height": robot["init_state"]["pos"][2],
        "effort_limit": dict(zip(joints, robot["dof_effort_limit_list"])),
        "joint_limits": dict(zip(joints, zip(robot["dof_pos_lower_limit_list"], robot["dof_pos_upper_limit_list"]))),
        "armature": dict(zip(joints, robot["dof_armature_list"])),
        "frictionloss": dict(zip(joints, robot["dof_joint_friction_list"])),
        "randomize_link_body_names": robot["randomize_link_body_names"],
    }
    (target/"control.json").write_text(json.dumps(profile, indent=2)+"\n")
    (target/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Copied {len(names)} original G1 asset files into {target}")


if __name__ == "__main__":
    main()
