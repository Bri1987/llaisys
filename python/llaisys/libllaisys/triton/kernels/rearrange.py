import torch


def kernel(inp, out):
    """Simple rearrange kernel: currently performs an identity copy.

    This is a placeholder implementation that copies `inp` into `out`.
    Triton-based optimized rearrangement can replace this later.
    """
    # Ensure contiguity and matching dtype/device
    if not isinstance(inp, torch.Tensor):
        raise TypeError("rearrange.kernel expects torch.Tensor inputs")

    out.copy_(inp)

