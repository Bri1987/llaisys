"""Triton embedding kernel.

This module directly imports Triton and exposes a concise Triton-only
kernel. If Triton is not installed, Python import will raise ImportError.
"""

import triton
import triton.language as tl


@triton.jit
def _kernel(index_ptr, weight_ptr, out_ptr, N, D, TOT, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOT

    # compute row and col for each flattened offset
    row = offsets // D
    col = offsets - row * D

    # load indices for each row
    ids = tl.load(index_ptr + row, mask=mask, other=0)

    # compute source positions in weight: ids * D + col
    src_pos = ids * D + col

    vals = tl.load(weight_ptr + src_pos, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, vals, mask=mask)


def kernel(index, weight, out, N, D, BLOCK_SIZE: int = 1024):
    TOT = N * D
    grid = ((TOT + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _kernel[grid](index, weight, out, N, D, TOT, BLOCK=BLOCK_SIZE)
