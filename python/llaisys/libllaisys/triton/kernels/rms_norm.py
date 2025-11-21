"""Triton RMSNorm kernel.

Computes per-row RMS normalization:

    Y_i = W * X_i / sqrt( (1/d) * sum_j X_j^2 + eps )

This module is Triton-only (no PyTorch fallback) so importing it
requires `triton` to be installed. The kernel is written for clarity
and correctness first (not heavily optimized)."""

import triton
import triton.language as tl


@triton.jit
def _kernel(inp_ptr, weight_ptr, out_ptr, M, D, EPS, BLOCK: tl.constexpr):
    """Each program instance handles one row and iterates over columns in blocks.

    - `inp_ptr`, `out_ptr` are flattened row-major arrays with shape (M, D).
    - `weight_ptr` is 1D of length D.
    - `BLOCK` is a compile-time block size for the inner loop.
    """
    row = tl.program_id(0)

    # vector of column offsets within a block
    cols = tl.arange(0, BLOCK)

    # accumulator as a length-1 vector of high precision
    acc = tl.zeros((1,), dtype=tl.float64)

    # compute base pointer for this row
    base = row * D

    # first pass: accumulate sum of squares
    for off in range(0, D, BLOCK):
        idx = base + off + cols
        mask = (off + cols) < D
        vals = tl.load(inp_ptr + idx, mask=mask, other=0.0).to(tl.float64)
        # tl.sum returns a scalar; add to length-1 vector to avoid scalar indexing
        acc += tl.sum(vals * vals)

    # compute RMS denominator (scalar)
    denom = tl.sqrt(tl.sum(acc) / D + EPS)

    # second pass: write normalized output
    for off in range(0, D, BLOCK):
        idx = base + off + cols
        mask = (off + cols) < D
        vals = tl.load(inp_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + off + cols, mask=mask, other=0.0).to(tl.float32)
        out_vals = (vals * w) / denom.to(tl.float32)
        tl.store(out_ptr + idx, out_vals, mask=mask)


def kernel(inp, weight, out, M, D, eps: float = 1e-5, BLOCK_SIZE: int = 1024):
    """Wrapper to launch the RMSNorm kernel.

    - `inp`/`out`/`weight` are `torch.Tensor` on CUDA and flattened/contiguous as needed.
    - `M` is number of rows, `D` is row length.
    """
    grid = (M,)
    _kernel[grid](inp, weight, out, M, D, eps, BLOCK=BLOCK_SIZE)
