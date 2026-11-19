"""Finite-state validation with one device synchronization for each group."""


def first_nonfinite(fields):
    import torch

    if not fields:
        return None
    names = tuple(fields)
    device = fields[names[0]].device
    flags = torch.stack([torch.isfinite(fields[name]).all().to(device) for name in names])
    if bool(flags.all()):
        return None
    return names[flags.logical_not().nonzero()[0].item()]
