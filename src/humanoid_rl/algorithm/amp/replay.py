"""Keep AMP state pairs aligned with the existing FlashSAC replay ring."""

from pathlib import Path

import torch


class AmpReplay:
    def __init__(self, base, feature_dim, control_dim=0):
        if base._n_step != 1:
            raise ValueError("AMP reward relabeling currently requires n_step=1")
        self.base = base
        self.feature_dim = feature_dim
        self.states = torch.empty(base._max_length, feature_dim, device=base._device)
        self.next_states = torch.empty_like(self.states)
        self.control_dim = control_dim
        self.control_terms = torch.empty(base._max_length, control_dim, device=base._device)

    def __len__(self):
        return len(self.base)

    def can_sample(self):
        return self.base.can_sample()

    def add(self, transition):
        count = len(transition["observation"])
        start = self.base._current_idx
        ids = (
            torch.arange(count, device=self.base._device) + start
        ) % self.base._max_length
        # Let the original buffer own all SAC data and episode semantics.
        keys = (
            "observation",
            "action",
            "reward",
            "terminated",
            "truncated",
            "next_observation",
        )
        self.base.add({key: transition[key] for key in keys})
        self.states[ids] = transition["amp_observation"].detach().to(self.base._device)
        self.next_states[ids] = (
            transition["amp_next_observation"].detach().to(self.base._device)
        )
        if self.control_dim:
            self.control_terms[ids] = transition["control_terms"].detach().to(self.base._device)

    def sample(self, sample_idxs=None):
        if sample_idxs is None:
            ids = torch.randint(
                len(self), (self.base._sample_batch_size,), device=self.base._device
            )
        else:
            ids = torch.as_tensor(sample_idxs, device=self.base._device)
        batch = self.base.sample(ids)
        batch["amp_observation"] = self.states[ids]
        batch["amp_next_observation"] = self.next_states[ids]
        if self.control_dim:
            batch["control_terms"] = self.control_terms[ids]
        return batch

    def sample_pairs(self, count, device):
        ids = torch.randint(len(self), (count,), device=self.base._device)
        return torch.cat((self.states[ids], self.next_states[ids]), -1).to(device)

    def get_observations(self):
        return self.base.get_observations()

    def save(self, path):
        self.base.save(path)
        torch.save(
            {
                "states": self.states[: len(self)],
                "next_states": self.next_states[: len(self)],
                "feature_dim": self.feature_dim,
                "control_dim": self.control_dim,
                "control_terms": self.control_terms[: len(self)],
            },
            Path(path).with_name("amp_replay.pt"),
        )

    def load(self, path):
        state = torch.load(
            Path(path).with_name("amp_replay.pt"),
            map_location=self.base._device,
            weights_only=True,
        )
        if (
            state["feature_dim"] != self.feature_dim
            or state.get("control_dim", 0) != self.control_dim
            or len(state["states"]) > self.base._max_length
        ):
            raise ValueError("AMP replay dimensions/capacity do not match")
        self.base.load(path)
        if len(state["states"]) != len(self):
            raise ValueError("SAC replay and AMP replay lengths do not match")
        self.states[: len(self)] = state["states"]
        self.next_states[: len(self)] = state["next_states"]
        if self.control_dim:
            self.control_terms[: len(self)] = state["control_terms"]
