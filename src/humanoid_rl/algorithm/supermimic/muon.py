"""Opt-in FP16 Muon orthogonalization for GPUs without native BF16 GEMM."""

_original = None


def zeropower_fp16(grad, ns_coefficients, ns_steps, eps):
    import torch

    if ns_steps >= 100 or grad.ndim != 2 or len(ns_coefficients) != 3:
        raise ValueError("Muon requires a matrix, three coefficients and fewer than 100 NS steps")
    a, b, c = ns_coefficients
    # Normalize in FP32 before conversion so small, valid gradients do not
    # underflow to zero in FP16. The normalized matrix has spectral norm <= 1.
    value = grad.float()
    value = (value / value.norm().clamp(min=eps)).to(torch.float16)
    transpose = grad.size(0) > grad.size(1)
    if transpose:
        value = value.T
    for _ in range(ns_steps):
        gram = value @ value.T
        update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        value = torch.addmm(value, update, value, beta=a)
    return value.T if transpose else value


def configure(dtype):
    """Affects only this isolated trainer process; the installed package stays intact."""
    global _original
    if dtype not in ("bfloat16", "float16"):
        raise ValueError("training.muon_dtype must be bfloat16 or float16")
    from torch.optim import _muon

    if _original is None:
        _original = _muon._zeropower_via_newtonschulz
    _muon._zeropower_via_newtonschulz = zeropower_fp16 if dtype == "float16" else _original
