"""Pure Torch inference for the saved carry MLP; no source training dependencies."""

import json

import torch
from torch import nn

from humanoid_rl.tasks.carry_box.stage.config import BUNDLE_VERSION, CONTEXT_DIM, SKILLS, STAGES, file_hash


class FrozenCarryPredictor(nn.Module):
    """Raw state101 -> [stage9, required_skill9, progress9, current_failure1]."""

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        # Loading an estimator must not consume the policy's initialization RNG.
        with torch.random.fork_rng(devices=[]):
            self.body = nn.Sequential(nn.Linear(101, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU())
            self.stage = nn.Linear(128, 9)
            self.skill = nn.Linear(128, 9)
            self.progress = nn.Linear(128, 9)
            self.failure = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        for name, shape in (("input_mean", (101,)), ("input_std", (101,)),
                            ("hidden_mean", (128,)), ("hidden_std", (128,)), ("calibration", (2,))):
            self.register_buffer(name, torch.zeros(shape))
        # Raw user checkpoint contains numpy statistics. Exported bundles contain
        # tensors/builtins only, and can subsequently use weights_only=True.
        try:
            state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
        except Exception as error:
            import pickle
            if not isinstance(error, pickle.UnpicklingError):
                raise
            state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        if state.get("format") == BUNDLE_VERSION:
            if state["source_sha256"] != cfg.source_sha256:
                raise ValueError("Frozen stage bundle source fingerprint mismatch")
            if json.dumps(state["schema"], sort_keys=True) != json.dumps(cfg.schema, sort_keys=True):
                raise ValueError("Frozen stage bundle input schema mismatch")
            self.load_state_dict(state["model"])
        else:
            if file_hash(cfg.checkpoint) != cfg.source_sha256:
                raise ValueError("Frozen stage source checkpoint changed")
            if state["dim"] != 101 or tuple(state["stages"]) != STAGES or tuple(state["skills"]) != SKILLS:
                raise ValueError("Unsupported carry predictor architecture/label order")
            base = state["base_state_dict"]
            for name in ("body", "stage", "skill", "progress"):
                getattr(self, name).load_state_dict({k.removeprefix(name + "."): v for k, v in base.items() if k.startswith(name + ".")})
            failure = state["failure_state_dict"]
            self.failure.load_state_dict({k.removeprefix("layers."): v for k, v in failure.items() if k.startswith("layers.")})
            for dest, value in (
                (self.input_mean, state["mean"]), (self.input_std, state["std"]),
                (self.hidden_mean, failure["mean"]), (self.hidden_std, failure["std"]),
                (self.calibration, state["calibration"]),
            ):
                dest.copy_(torch.as_tensor(value, dtype=torch.float32))
        if not all(torch.isfinite(value).all() for value in self.state_dict().values()):
            raise ValueError("Nonfinite frozen MLP parameters/statistics")
        if not (self.input_std > 0).all() or not (self.hidden_std > 0).all():
            raise ValueError("Frozen MLP standard deviations must be positive")
        self.to(device)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, raw_state):
        if raw_state.ndim != 2 or raw_state.shape[1] != 101:
            raise ValueError("Frozen carry MLP expects [N,101] raw states")
        normalized = (raw_state - self.input_mean) / self.input_std
        h = self.body(normalized.clamp(-20, 20))
        failure = self.failure(((h - self.hidden_mean) / self.hidden_std).clamp(-20, 20)).squeeze(-1)
        failure = (self.calibration[0] * failure + self.calibration[1]).sigmoid()
        return torch.cat((self.stage(h).softmax(-1), self.skill(h).softmax(-1), self.progress(h).sigmoid(), failure[:, None]), -1)

    def save_bundle(self, path):
        torch.save({"format": BUNDLE_VERSION, "source_sha256": self.cfg.source_sha256,
                    "schema": self.cfg.schema, "context_dim": CONTEXT_DIM,
                    "model": self.state_dict()}, path)
