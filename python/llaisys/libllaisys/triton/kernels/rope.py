"""Triton RoPE kernel.

Computes RoPE per row: input shape (seq_len, n_heads, head_dim).
Kernel processes one (seq, head) row per program id and a block of positions
along the head_dim (only the first half); it writes both halves (a' and b').
"""
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(x_ptr, out_ptr, pos_ptr, freqs_ptr, seq_len, n_heads, head_dim, HALF: tl.constexpr, BLOCK: tl.constexpr):
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

    # load position and freqs
    pos = tl.load(pos_ptr + seq)
    pos = pos.to(tl.float32)

    freq = tl.load(freqs_ptr + cols, mask=mask, other=0.0)
    # compute angle = pos * freq (broadcasting scalar pos over freq vector)
    angle = pos * freq
    cos_v = tl.cos(angle)
    sin_v = tl.sin(angle)

    y_a = a * cos_v - b * sin_v
    y_b = b * cos_v + a * sin_v

    # store results
    tl.store(out_ptr + a_idx, y_a, mask=mask)
    tl.store(out_ptr + b_idx, y_b, mask=mask)


def kernel(x, out, pos_ids, freqs, BLOCK=128):
    """Launch the RoPE kernel.

    x/out: 3D tensors flattened as 1D torch tensors (contiguous view)
    pos_ids: 1D tensor length seq_len
    freqs: 1D tensor length head_dim//2
    """
    seq_len = x.shape[0]
    # x provided as flattened vector but we need shapes; expect user to pass shape via freqs len
    # here we infer head_dim from freqs
    HALF = freqs.shape[0]
    # derive head_dim and n_heads from x and pos_ids: x.numel() = seq_len * n_heads * head_dim
    total_elems = x.numel()
    # head_dim = HALF * 2
    head_dim = HALF * 2
    n_heads = total_elems // (seq_len * head_dim)

    grid = (seq_len * n_heads, (HALF + BLOCK - 1) // BLOCK)
    _rope_kernel[grid](x, out, pos_ids, freqs, seq_len, n_heads, head_dim, HALF=HALF, BLOCK=BLOCK)
