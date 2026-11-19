"""Original latent rollouts extended to variable-length, boundary-safe experts."""

import torch


class VariableExpertRollout:
    """End a short reference at its real boundary and sample the next command.

    Latent storage has unused slots, but no repeated/padded expert states are
    supplied to the learner or to backward_map. The FB/CPR/Aux updates are intact.
    """

    def __init__(self, dataset, agent, num_envs, device):
        self.dataset, self.agent, self.device = dataset, agent, device
        cfg = agent.cfg.train
        self.horizon = cfg.rollout_expert_trajectories_length
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.active[torch.randperm(num_envs, device=device)[:int(num_envs*cfg.rollout_expert_trajectories_percentage)]] = True
        self.position = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.length = torch.zeros_like(self.position)
        self.commands = torch.zeros(num_envs, self.horizon, agent.cfg.model.archi.z_dim, device=device)

    @torch.no_grad()
    def __call__(self, z, step_count, buffers):
        cfg = self.agent.cfg.train
        enabled = cfg.rollout_expert_trajectories
        # Same random/mixed latent branch as upstream FB. Its config is frozen;
        # the variable expert schedule replaces only the fixed-length branch.
        if z is None:
            z = self.agent._model.sample_z(len(step_count), device=self.device)
        else:
            mask = step_count % cfg.update_z_every_step == 0
            if cfg.use_mix_rollout and not self.agent.z_buffer.empty():
                new_z = self.agent.z_buffer.sample(len(z), device=self.device)
            else:
                new_z = self.agent._model.sample_z(len(z), device=self.device)
            z = torch.where(mask, new_z, z.to(self.device))
        if not enabled:
            return z
        ids = (self.active & ((step_count[:, 0] == 0) | (self.position >= self.length))).nonzero().flatten()
        if len(ids):
            self._refill(ids)
        ids = self.active.nonzero().flatten()
        z[ids] = self.commands[ids, self.position[ids]]
        self.position[ids] += 1
        return z

    def _refill(self, ids):
        dataset, model = self.dataset, self.agent._model
        clips = torch.multinomial(dataset.clip_probabilities, len(ids), replacement=True)
        lengths = (dataset.clip_lengths[clips]-1).clamp_max(self.horizon)
        starts = dataset.clip_starts[clips]+(torch.rand(len(ids))*(dataset.clip_lengths[clips]-lengths)).long()
        indices = torch.cat([torch.arange(int(start)+1, int(start+length)+1) for start, length in zip(starts, lengths)])
        storage = dataset.expert_slicer.storage["observation"]
        obs = {key: value[indices.to(value.device)].to(self.device) for key, value in storage.items()}
        encoded = model.backward_map(obs)
        offset = 0
        for env_id, length in zip(ids.tolist(), lengths.tolist()):
            segment = encoded[offset:offset+length]
            cumulative = torch.cat((torch.zeros_like(segment[:1]), segment.cumsum(0)))
            first = torch.arange(length, device=self.device)
            last = (first+model.cfg.seq_length).clamp_max(length)
            smoothed = (cumulative[last]-cumulative[first])/(last-first)[:, None]
            self.commands[env_id, :length] = model.project_z(smoothed)
            offset += length
        self.length[ids] = lengths.to(self.device)
        self.position[ids] = 0
