"""H1 expert states, shared observation encoding and episode-safe sequence sampling."""

import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np
import torch

from humanoid_retarget.datasets.robot_motion import load_robot_motion
from humanoid_retarget.datasets.paths import manifest_root
from humanoid_retarget.resample import resample_robot_motion
from humanoid_rl.env.bfm_features import FEATURE_VERSION, privileged_state, proprioception
from humanoid_rl.algorithm.bfm_zero.upstream import ROOT, enable_bfm_zero


class DeviceTrajectorySampler:
    """Keep expert storage on CPU and transfer only sampled minibatches."""

    def __init__(self, buffer, device):
        self.buffer, self.device = buffer, device

    def sample(self, *args, **kwargs):
        from torch.utils._pytree import tree_map
        return tree_map(lambda value: value.to(self.device), self.buffer.sample(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self.buffer, name)


def encode_expert(model, qpos, qvel, default_joint_pos):
    """The same link origins, body order and cvel conversion as mjlab EntityData."""
    data = mujoco.MjData(model)
    body_ids = np.arange(1, model.nbody)
    count, bodies = len(qpos), len(body_ids)
    positions = np.empty((count, bodies, 3), np.float32)
    rotations = np.empty((count, bodies, 4), np.float32)
    linear, angular = np.empty_like(positions), np.empty_like(positions)
    for index in range(count):
        data.qpos[:], data.qvel[:] = qpos[index], qvel[index]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        positions[index], rotations[index] = data.xpos[body_ids], data.xquat[body_ids]
        angular[index] = data.cvel[body_ids, :3]
        offset = data.subtree_com[1]-data.xpos[body_ids]
        linear[index] = data.cvel[body_ids, 3:]-np.cross(data.cvel[body_ids, :3], offset)
    q, qd = torch.from_numpy(qpos), torch.from_numpy(qvel)
    pos, rot, vel, omega = map(torch.from_numpy, (positions, rotations, linear, angular))
    return {
        "state": proprioception(q[:, 7:], qd[:, 6:], torch.as_tensor(default_joint_pos), rot[:, 0], omega[:, 0]),
        "privileged_state": privileged_state(pos, rot, vel, omega),
    }


class MotionDataset:
    def __init__(self, settings, env_settings, control, sequence_length, expert_device="cpu", agent_cfg=None):
        enable_bfm_zero()
        from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer
        self.robot = agent_cfg.robot if agent_cfg else "h1"
        if self.robot == "g1":
            from robots.humanoid.g1.bfm_zero import get_spec
            from robots.humanoid.g1.config import JOINT_NAMES
            self.feature_version = "g1-bfm-state64-priv463-original-v1"
        elif self.robot == "humanoid_light":
            from robots.humanoid.humanoid_light.config import JOINT_NAMES, get_spec
            self.feature_version = "humanoid-light-bfm-state58-priv463-v1"
        else:
            from robots.humanoid.h1.config import JOINT_NAMES, get_spec
            self.feature_version = FEATURE_VERSION

        root = Path(settings["root"])
        self.root = (root if root.is_absolute() else ROOT/root).resolve()
        files = settings.get("files")
        self.validation_paths = []
        self.manifest = None
        selected_ranges = {}
        if settings.get("manifest"):
            manifest_path = Path(settings["manifest"])
            manifest_path = manifest_path if manifest_path.is_absolute() else ROOT/manifest_path
            self.manifest = json.loads(manifest_path.read_text())
            if self.manifest.get("robot") != self.robot or self.manifest.get("version") != 1:
                raise ValueError("Expert manifest robot/version differs from the selected task")
            split = settings.get("split", "train")
            if split not in self.manifest["splits"]:
                raise ValueError(f"Missing expert split {split!r}")
            self.root = manifest_root(settings, ROOT)
            for record in self.manifest["splits"][split]:
                path = (self.root/record["path"]).resolve()
                selected_ranges.setdefault(path, []).append(record)
            paths = sorted(selected_ranges)
            self.validation_paths = sorted({record["path"] for record in self.manifest["splits"].get("validation", [])})
            if files:
                paths = [path for path in paths if path.name in files or str(path.relative_to(self.root)) in files]
        elif files:
            paths = [(self.root/name).resolve() for name in files]
        else:
            subject = settings.get("validation_subject", 5)
            all_paths = sorted(self.root.glob("*.npz"))
            self.validation_paths = [path.name for path in all_paths
                                     if subject is not None and re.search(fr"_subject{subject}(?:_|$)",path.stem)]
            paths = [path for path in all_paths if path.name not in self.validation_paths]
        if settings.get("limit") is not None:
            paths = paths[:settings["limit"]]
        if not paths or any(not path.is_file() or not path.is_relative_to(self.root) for path in paths):
            raise ValueError(f"No valid local expert motions under {self.root}")
        model = get_spec().compile()
        defaults = np.asarray([control["default_joint_pos"][name] for name in JOINT_NAMES], np.float32)
        self.fps = 1/(env_settings.physics_dt*env_settings.decimation)
        if not np.isclose(self.fps, 50):
            raise ValueError("BFM motion-prior integration currently uses 50 Hz control")
        self.clips, episodes, all_qpos, all_qvel = [], [], [], []
        fingerprint = hashlib.sha256(self.feature_version.encode())
        if self.manifest is not None:
            fingerprint.update(json.dumps(self.manifest, sort_keys=True).encode())
            fingerprint.update(json.dumps({key: settings.get(key) for key in
                                          ("clip_seconds", "source_weights", "split")}, sort_keys=True).encode())
        cache = Path(settings["cache_dir"])
        cache = cache if cache.is_absolute() else ROOT/cache
        cache.mkdir(parents=True, exist_ok=True)
        for path in paths:
            ranges = selected_ranges.get(path)
            contents = path.read_bytes()
            if ranges and any(record["sha256"] != hashlib.sha256(contents).hexdigest() for record in ranges):
                raise ValueError(f"Expert motion changed since screening: {path}")
            parameters = {"feature_version": self.feature_version, "fps": self.fps,
                          "max_frames": settings.get("max_frames"), "defaults": defaults.tolist(),
                          **({"ranges": ranges} if ranges else {})}
            digest = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode())
            digest.update(contents)
            for key in ("body_pos", "body_quat", "jnt_pos", "jnt_axis"):
                digest.update(getattr(model, key).tobytes())
            fingerprint.update(digest.digest())
            cached = cache/f"{path.stem}-{digest.hexdigest()[:16]}.pt"
            if cached.is_file():
                segments = torch.load(cached, map_location="cpu", weights_only=True, mmap=True)
            else:
                motion = load_robot_motion(path)
                if motion.joint_names != ("root",)+JOINT_NAMES:
                    raise ValueError(f"Expert joint order differs from the {self.robot} MJCF: {path}")
                segments = []
                intervals = ([(record["start_frame"], record["stop_frame"]) for record in ranges]
                             if ranges else zip(motion.split_points[:-1], motion.split_points[1:]))
                for start, stop in intervals:
                    if not 0 <= start < stop <= len(motion.qpos):
                        raise ValueError(f"Expert manifest range outside {path}")
                    if not any(left <= start < stop <= right for left, right in zip(motion.split_points[:-1], motion.split_points[1:])):
                        raise ValueError(f"Expert manifest crosses a source trajectory boundary: {path}")
                    if settings.get("max_frames") is not None:
                        stop = min(stop, start+settings["max_frames"])
                    if stop-start < 2:
                        continue
                    qpos, qvel = resample_robot_motion(motion.qpos[start:stop], motion.qvel[start:stop], motion.fps, self.fps)
                    if self.robot == "g1":
                        if not np.isclose(motion.fps,self.fps):
                            raise ValueError("Original G1 exports must match the 50 Hz control rate")
                        with np.load(path,allow_pickle=False) as source:
                            obs = {"state":torch.from_numpy(source["expert_state"][start:stop].copy()),
                                   "privileged_state":torch.from_numpy(source["expert_privileged_state"][start:stop].copy())}
                        if obs["state"].shape != (len(qpos),64) or obs["privileged_state"].shape != (len(qpos),463):
                            raise ValueError(f"Original G1 feature export has incorrect dimensions: {path}")
                    else:
                        obs = encode_expert(model, qpos, qvel, defaults)
                    segments.append({"qpos": torch.from_numpy(qpos), "qvel": torch.from_numpy(qvel), "observation": obs,
                                     "source_start_frame": int(start)})
                temporary = cached.with_suffix(".tmp")
                torch.save(segments, temporary)
                temporary.replace(cached)
                # Retain file-backed views while assembling the full dataset.
                # Keeping every freshly encoded source in anonymous RAM makes
                # the later trajectory-buffer copies overlap with all sources.
                segments = torch.load(cached, map_location="cpu", weights_only=True, mmap=True)
            stride = int(round(settings["clip_seconds"]*self.fps))
            if stride < sequence_length:
                raise ValueError("Expert clips must cover at least one complete training sequence")
            for segment in segments:
                for start in range(0, len(segment["qpos"])-1, stride):
                    stop = min(start+stride+1, len(segment["qpos"]))
                    if stop-start <= sequence_length:
                        continue
                    obs = {key: value[start:stop] for key, value in segment["observation"].items()}
                    qpos, qvel = segment["qpos"][start:stop], segment["qvel"][start:stop]
                    length = len(qpos)
                    truncated = torch.zeros(length, 1, dtype=torch.bool)
                    truncated[-1] = True
                    episodes.append({"observation": obs, "truncated": truncated,
                                     "motion_id": torch.full((length, 1), len(episodes), dtype=torch.long)})
                    self.clips.append({"source": path.name, "start_frame": start+segment.get("source_start_frame", 0),
                                       "frames": length,
                                       **({"dataset": ranges[0]["source"], "path": str(path.relative_to(self.root))} if ranges else {})})
                    all_qpos.append(qpos)
                    all_qvel.append(qvel)
            print(f"BFM expert {path.name}: encoded at {self.fps:g} Hz", flush=True)
        if not episodes:
            raise ValueError("Expert motions are shorter than sequence_length")
        storage_device = settings.get("expert_buffer_device", expert_device)
        buffer = TrajectoryDictBuffer(episodes, device=storage_device, seq_length=sequence_length, end_key="truncated")
        weights = settings.get("source_weights")
        if weights:
            groups = [clip.get("dataset", "legacy") for clip in self.clips]
            counts = {source: groups.count(source) for source in set(groups)}
            if set(groups)-set(weights) or any(not np.isfinite(w) or w <= 0 for w in weights.values()):
                raise ValueError("Source weights must cover every expert dataset with positive finite weights")
            buffer.priorities = torch.tensor([weights[source]/counts[source] for source in groups], device=storage_device)
            buffer.priorities /= buffer.priorities.sum()
        self.expert_slicer = (DeviceTrajectorySampler(buffer, expert_device)
                              if str(storage_device) != str(expert_device) else buffer)
        self.qpos, self.qvel = torch.cat(all_qpos), torch.cat(all_qvel)
        self.fingerprint = fingerprint.hexdigest()
        self.paths = [path.name for path in paths]
        self.clip_starts = torch.tensor(np.cumsum([0]+[clip["frames"] for clip in self.clips[:-1]]))
        self.clip_lengths = torch.tensor([clip["frames"] for clip in self.clips])
        self.clip_probabilities = buffer.priorities.detach().cpu()
        if not torch.isfinite(self.qpos).all() or not torch.isfinite(self.qvel).all():
            raise ValueError("Nonfinite expert states")

    def sample_states(self, count, device):
        if self.robot == "g1" or self.manifest is not None:
            clips = (torch.multinomial(self.clip_probabilities, count, replacement=True)
                     if self.manifest is not None else torch.randint(len(self.clips), (count,)))
            ids = self.clip_starts[clips]+(torch.rand(count)*(self.clip_lengths[clips]-1)).long()
        else:
            ids = torch.randint(len(self.qpos), (count,))
        return self.qpos[ids].to(device).clone(), self.qvel[ids].to(device).clone()

    def metadata(self):
        return {"feature_version": self.feature_version, "fingerprint": self.fingerprint,
                "fps": self.fps, "training_motions": self.paths, "validation_motions": self.validation_paths,
                "clip_count": len(self.clips), "frame_count": len(self.qpos),
                "source_clip_counts": {name: sum(clip.get("dataset", "legacy") == name for clip in self.clips)
                                       for name in sorted({clip.get("dataset", "legacy") for clip in self.clips})}}
