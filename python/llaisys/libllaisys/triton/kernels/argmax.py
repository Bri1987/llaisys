
"""Triton argmax kernels (two-stage reduction).

This module directly imports and uses Triton. If Triton is not installed,
Python will raise ImportError at import time.
"""

import triton
import triton.language as tl


@triton.jit
def _stage1(vals_ptr, partial_vals_ptr, partial_idx_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK
    local_max = -1e20
    local_idx = -1
    # scalar-loop over block to avoid unsupported vector-indexing
    for i in range(BLOCK):
        off = base + i
        valid = off < n_elements
        v = tl.load(vals_ptr + off, mask=valid, other=-1e20)
        if valid and v > local_max:
            local_max = v
            local_idx = off

    tl.store(partial_vals_ptr + pid, local_max)
    tl.store(partial_idx_ptr + pid, local_idx)


@triton.jit
def _stage2(partial_vals_ptr, partial_idx_ptr, max_val_ptr, max_idx_ptr, m_blocks, BLOCK: tl.constexpr):
    best_v = -1e20
    best_i = -1
    # single-program reduction over partials
    for i in range(m_blocks):
        v = tl.load(partial_vals_ptr + i)
        idx = tl.load(partial_idx_ptr + i)
        if v > best_v:
            best_v = v
            best_i = idx
    tl.store(max_val_ptr, best_v)
    tl.store(max_idx_ptr, best_i)


def kernel_stage1(vals, partial_vals, partial_idx, n, BLOCK_SIZE=1024):
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _stage1[grid](vals, partial_vals, partial_idx, n, BLOCK=BLOCK_SIZE)


def kernel_stage2(partial_vals, partial_idx, max_val, max_idx, m_blocks, BLOCK_SIZE=1024):
    _stage2[(1,)](partial_vals, partial_idx, max_val, max_idx, m_blocks, BLOCK=BLOCK_SIZE)
 

