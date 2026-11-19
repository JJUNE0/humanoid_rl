"""Read immutable expert split lists; never join transitions across episode resets."""

import hashlib
import json
import tempfile
from pathlib import Path

import torch

from humanoid_retarget.datasets.carry_box import load_motion, read_split_lists, scene_description

from humanoid_rl.algorithm.amp.features import OBJECT_REFERENCE_DIM, ExpertFeatures, feature_schema


class MotionSplit:
    def __init__(
        self, states, references, starts, phase_ids, phase_names, fingerprint, paths
    ):
        self.states, self.references, self.starts = states, references, starts
        self.phase_names = phase_names
        self.buckets = [starts[phase_ids == i] for i in range(len(phase_names))]
        self.fingerprint, self.paths = fingerprint, paths
        self.phase_counts = {
            name: len(bucket) for name, bucket in zip(phase_names, self.buckets)
        }

    def indices(self, count, balanced=True):
        if not balanced:
            return self.starts[torch.randint(len(self.starts), (count,))]
        phases = torch.randint(len(self.buckets), (count,))
        result = torch.empty(count, dtype=torch.long)
        for i, bucket in enumerate(self.buckets):
            mask = phases == i
            result[mask] = bucket[torch.randint(len(bucket), (int(mask.sum()),))]
        return result

    def sample_pairs(self, count, device="cpu", balanced=True):
        ids = self.indices(count, balanced)
        return torch.cat((self.states[ids], self.states[ids + 1]), -1).to(device)

    def sample_references(self, count, device, balanced=True):
        return self.references[self.indices(count, balanced)].to(device)


class MotionDataset:
    def __init__(
        self, root, joint_names, cache_dir="outputs/amp_cache", max_motions=None,
        observe_object=False,
        single_motion=None,
    ):
        self.root = Path(root).resolve()
        self.joint_names = tuple(joint_names)
        self.observe_object = observe_object
        self.feature_version, self.feature_groups = feature_schema(observe_object)
        self.feature_dim = sum(self.feature_groups.values())
        self.encoder = ExpertFeatures(joint_names, observe_object)
        self.cache_dir = Path(cache_dir)
        self.single_motion = single_motion
        if single_motion is not None:
            relative = Path(single_motion)
            if relative.is_absolute() or ".." in relative.parts or relative.name != "motion.npz":
                raise ValueError("single_motion must be a relative episode/motion.npz path")
            if max_motions is not None:
                raise ValueError("single_motion selects the clip; omit max_motions")
            self.train = self._load_split("train", [relative.as_posix()])
            self.validation = None
            print("AMP single-motion overfit: held-out expert validation DISABLED", flush=True)
            return
        lists = read_split_lists(self.root, max_motions)
        self.train = self._load_split("train", lists["train"])
        self.validation = self._load_split("validation", lists["validation"])

    def _load_split(self, split, paths):
        digest = hashlib.sha256(self.feature_version.encode())
        digest.update(json.dumps(self.joint_names).encode())
        for field in ("body_pos", "body_quat", "jnt_pos", "jnt_axis"):
            digest.update(getattr(self.encoder.model, field).tobytes())
        blobs, scenes = [], []
        for relative in paths:
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError(f"Expert path leaves dataset root: {relative}")
            blob = path.read_bytes()
            digest.update(relative.encode())
            digest.update(blob)
            blobs.append(blob)
            if self.observe_object:
                # Changes to a body's geometry frame/support layout invalidate
                # both object features and scene references, even if qpos is unchanged.
                sidecars = []
                for name in ("box_config.yaml", "robot.xml"):
                    sidecar = (path.parent / name).resolve()
                    if not sidecar.is_relative_to(self.root):
                        raise ValueError(f"Expert geometry path leaves dataset root: {sidecar}")
                    data = sidecar.read_bytes()
                    digest.update(name.encode())
                    digest.update(data)
                    sidecars.append(data)
                scenes.append(scene_description(*sidecars))
            else:
                scenes.append(None)
        fingerprint = digest.hexdigest()
        cache_path = self.cache_dir / f"{self.feature_version}-{split}-{fingerprint[:16]}.pt"
        if cache_path.exists():
            saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            saved = self._encode(paths, blobs, fingerprint, scenes)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=cache_path.parent, suffix=".tmp", delete=False
            ) as file:
                temporary = Path(file.name)
            try:
                torch.save(saved, temporary)
                temporary.replace(cache_path)
            finally:
                temporary.unlink(missing_ok=True)
        if (
            saved["fingerprint"] != fingerprint
            or saved["states"].shape[-1] != self.feature_dim
            or saved["references"].shape[-1] != (OBJECT_REFERENCE_DIM if self.observe_object else 71)
        ):
            raise ValueError(f"Incompatible AMP feature cache: {cache_path}")
        result = MotionSplit(**saved)
        print(
            f"AMP {split}: {len(paths)} clips, {len(result.starts)} transitions, phases={result.phase_counts}",
            flush=True,
        )
        return result

    def _encode(self, paths, blobs, fingerprint, scenes):
        states, references, starts, phases = [], [], [], []
        offset = 0
        for relative, blob, scene in zip(paths, blobs, scenes):
            motion = load_motion(blob, label=relative)
            features, raw = self.encoder.encode(motion, scene)
            if (
                len(features) != len(motion["time"])
                or not torch.isfinite(features).all()
                or not torch.isfinite(raw).all()
            ):
                raise ValueError(
                    f"Nonfinite/misaligned expert states in {relative}"
                )
            states.append(features)
            references.append(raw)
            starts.append(torch.arange(len(features) - 1) + offset)
            phases.extend(motion["phase"][:-1].tolist())
            offset += len(features)
        phase_names = sorted(set(phases))
        phase_ids = torch.tensor([phase_names.index(name) for name in phases])
        return dict(
            states=torch.cat(states),
            references=torch.cat(references),
            starts=torch.cat(starts),
            phase_ids=phase_ids,
            phase_names=phase_names,
            fingerprint=fingerprint,
            paths=paths,
        )
