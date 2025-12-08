"""Triton RoPE kernel.

Computes RoPE per row: input shape (seq_len, n_heads, head_dim).
Kernel processes one (seq, head) row per program id and a block of positions
along the head_dim (only the first half); it writes both halves (a' and b').
"""
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(x_ptr, out_ptr, sin_ptr, cos_ptr, seq_len, n_heads, head_dim, HALF: tl.constexpr, BLOCK: tl.constexpr):
    # program ids: row_index enumerates seq_len * n_heads
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    # compute seq and head from row
    seq = row // n_heads
    head = row % n_heads

    # base offset for this (seq, head)
    base = (seq * n_heads + head) * head_dim

    col_start = col_block * BLOCK
    cols = col_start + tl.arange(0, BLOCK)
    mask = cols < HALF

    # load a and b for cols
    a_idx = base + cols
    b_idx = base + cols + HALF

    a = tl.load(x_ptr + a_idx, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + b_idx, mask=mask, other=0.0).to(tl.float32)

    # load precomputed sin/cos values for this (seq, half) block
    base = seq * HALF
    sin_v = tl.load(sin_ptr + (base + cols), mask=mask, other=0.0).to(tl.float32)
    cos_v = tl.load(cos_ptr + (base + cols), mask=mask, other=1.0).to(tl.float32)

    y_a = a * cos_v - b * sin_v
    y_b = b * cos_v + a * sin_v

    # store results
    tl.store(out_ptr + a_idx, y_a, mask=mask)
    tl.store(out_ptr + b_idx, y_b, mask=mask)


def kernel(x, out, sin, cos, BLOCK=128):
    """Launch the RoPE kernel using precomputed sin/cos arrays.

    x/out: 3D tensors flattened as 1D contiguous tensors
    sin/cos: 1D flattened arrays of length seq_len * (head_dim//2)
    """
    seq_len = x.shape[0]
    total_elems = x.numel()
    # infer HALF from sin length divided by seq_len
    HALF = sin.shape[0] // seq_len
    head_dim = HALF * 2
    n_heads = total_elems // (seq_len * head_dim)

    grid = (seq_len * n_heads, (HALF + BLOCK - 1) // BLOCK)
    _rope_kernel[grid](x, out, sin, cos, seq_len, n_heads, head_dim, HALF=HALF, BLOCK=BLOCK)
