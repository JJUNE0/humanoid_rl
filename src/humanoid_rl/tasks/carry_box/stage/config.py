"""Portable preprocessing specification for the user's 101D carry predictor."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

SCHEMA_VERSION = "carry-current101-v1"
BUNDLE_VERSION = "frozen-carry-mlp-v1"
CONTEXT_DIM = 28
STAGES = ("approach", "acquire", "lift", "transport", "place", "release", "complete", "box_lost", "fallen")
SKILLS = ("approach", "acquire", "lift", "transport", "place", "release", "stop", "wait_for_box", "get_up")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schema_from_episode(directory):
    """Read source conventions once; runtime never imports the I-BFM code."""
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    robot = yaml.safe_load((directory / "robot_config.yaml").read_text())
    box = yaml.safe_load((directory / "box_config.yaml").read_text())
    names = metadata["joint_names"]
    defaults = robot["training"]["init_state"]["default_joint_angles"]
    with np.load(directory / "physics.npz", allow_pickle=False) as physics:
        com = physics["body_ipos"][metadata["body_names"].index("sim2sim_carry_box")].tolist()
    schema = {
        "version": SCHEMA_VERSION,
        "joint_names": names,
        "default_joint_positions": [defaults[name] for name in names],
        "hand_contact_offsets": box["hand_contact_offsets"],
        "box_geom_position": box["geom_pos"],
        "box_geom_quaternion_wxyz": box["geom_quat_wxyz"],
        "box_com_position": com,
        "source_episode": str(directory),
        "source_robot_sha256": file_hash(directory / "robot.xml"),
    }
    validate_schema(schema)
    return schema


def validate_schema(schema):
    if schema["version"] != SCHEMA_VERSION:
        raise ValueError("Unknown carry MLP input schema")
    if len(schema["joint_names"]) != 29 or len(set(schema["joint_names"])) != 29:
        raise ValueError("Carry MLP schema needs 29 unique named joints")
    for name, shape in (
        ("default_joint_positions", (29,)), ("hand_contact_offsets", (2, 3)),
        ("box_geom_position", (3,)), ("box_geom_quaternion_wxyz", (4,)), ("box_com_position", (3,)),
    ):
        values = np.asarray(schema[name])
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid MLP schema field {name}")
    if not np.isclose(np.linalg.norm(schema["box_geom_quaternion_wxyz"]), 1, atol=1e-4):
        raise ValueError("Teacher geometry quaternion must be unit length")


@dataclass
class StageModelConfig:
    checkpoint: str
    source_sha256: str
    schema: dict

    @classmethod
    def from_reference(cls, checkpoint, reference_episode):
        checkpoint = Path(checkpoint).resolve()
        return cls(str(checkpoint), file_hash(checkpoint), schema_from_episode(reference_episode))

    @classmethod
    def from_bundle(cls, checkpoint):
        import torch
        checkpoint = Path(checkpoint).resolve()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("format") != BUNDLE_VERSION:
            raise ValueError("This is a raw stage checkpoint; specify a reference episode")
        validate_schema(state["schema"])
        return cls(str(checkpoint), state["source_sha256"], state["schema"])

    @classmethod
    def from_metadata(cls, metadata, checkpoint_directory):
        validate_schema(metadata["schema"])
        path = Path(checkpoint_directory) / "stage_model.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Frozen stage predictor bundle missing: {path}")
        return cls(str(path.resolve()), metadata["source_sha256"], metadata["schema"])

    def metadata(self):
        return asdict(self)
