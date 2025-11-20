"""Triton-backed `add` kernel for CUDA.

This module provides a `kernel(a, b, c, BLOCK_SIZE=1024)` function that
performs elementwise addition `c = a + b` on CUDA device using Triton. It
accepts PyTorch tensors (`torch.Tensor`) on CUDA with dtype `float32` or
`float16`.

Usage example:
    import torch
    from llaisys.libllaisys.ninetoothed.kernels.add import kernel

    a = torch.randn(10240, device='cuda', dtype=torch.float32)
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    kernel(a, b, c, BLOCK_SIZE=1024)

Notes:
- Triton must be installed (`pip install triton`) and PyTorch CUDA build
  available.
- The caller must ensure inputs/outputs are on the same CUDA device and
  have identical shapes/dtypes.
"""

import torch
try:
    import triton
    import triton.language as tl
except Exception:
    triton = None


def _make_add_kernel():
    if triton is None:
        raise RuntimeError("Triton not available")

    @triton.jit
    def _kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offset = pid * BLOCK
        idxs = offset + tl.cast(tl.arange(0, BLOCK), tl.int32)
        mask = idxs < n
        a = tl.load(a_ptr + idxs, mask=mask)
        b = tl.load(b_ptr + idxs, mask=mask)
        tl.store(c_ptr + idxs, a + b, mask=mask)

    return _kernel


_CACHED_KERNEL = None


def kernel(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, BLOCK_SIZE: int = 1024):
    """Run elementwise add using Triton on CUDA tensors.

    Args:
        a, b, c: `torch.Tensor` on CUDA. Same shape and dtype (float32/float16).
        BLOCK_SIZE: Triton block size (elements per program instance).
    """
    # This kernel expects `a`, `b`, `c` to be CUDA torch.Tensors.
    # Conversions from LLAISYS handles are performed by the launcher
    # (`triton.setup_kernels.llaisysAdd`).

    # now a,b,c are torch tensors on CUDA
    if triton is None:
        raise RuntimeError("Triton not found. Install triton (pip install triton).")

    if not (a.is_cuda and b.is_cuda and c.is_cuda):
        raise ValueError("All tensors must be CUDA tensors")
    if not (a.shape == b.shape == c.shape):
        raise ValueError("All tensors must have the same shape")
    if not (a.dtype == b.dtype == c.dtype):
        raise ValueError("All tensors must have the same dtype")
    if a.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("Unsupported dtype: only float32/float16/bfloat16 are supported")

    global _CACHED_KERNEL
    if _CACHED_KERNEL is None:
        _CACHED_KERNEL = _make_add_kernel()

    n = a.numel()
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch Triton kernel with raw pointers (let exceptions propagate on failure)
    # Pass torch tensors directly to Triton JIT (it will handle device pointers)
    _CACHED_KERNEL[grid](a, b, c, n, BLOCK=BLOCK_SIZE)
    # Debug: print the device result just after kernel/fallback to inspect
    try:
        if c.numel() <= 64:
            print("[triton.add] device result (post-kernel):", c)
    except Exception:
        pass

    # Launcher is responsible for any LLAISYS->Torch conversion and write-back.

