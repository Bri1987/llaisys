"""Triton linear kernel: computes Y = X @ W^T + b.

This module uses Triton only (no PyTorch fallback). Import will fail
with ImportError if Triton is not installed.
"""

import triton
import triton.language as tl


@triton.jit
def _kernel(
    A, B, C, D, M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUT_FP32: tl.constexpr = True,
    OUT_FP16: tl.constexpr = False,
    OUT_BF16: tl.constexpr = False,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Blocked, vectorized implementation:
    # - load blocks of A: shape (BLOCK_M, BLOCK_K)
    # - load blocks of B: shape (BLOCK_N, BLOCK_K)
    # - accumulate C_block (BLOCK_M, BLOCK_N) via vectorized outer-products
    m_off = m_start + tl.arange(0, BLOCK_M)
    n_off = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_offs = tl.arange(0, BLOCK_K) + k

        # pointers for block loads
        a_ptr = A + (m_off[:, None] * K + k_offs[None, :])
        b_ptr = B + (n_off[:, None] * K + k_offs[None, :])

        # masks for partial tiles
        a_mask = (m_off[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (n_off[:, None] < N) & (k_offs[None, :] < K)

        a = tl.load(a_ptr, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr, mask=b_mask, other=0.0).to(tl.float32)

        # vectorized accumulation: sum over k of a[:, :, None] * b[None, :, :]
        acc += tl.sum(a[:, None, :] * b[None, :, :], axis=2)

        k += BLOCK_K

    # bias: load for n indices and add (broadcast over m dimension)
    bias = tl.load(D + n_off, mask=(n_off < N), other=0.0).to(tl.float32)
    out_block = acc + bias[None, :]

    # cast results to the output buffer dtype (fixes buffer-size mismatch bug)
    if OUT_FP32:
        out_block = out_block.to(tl.float32)
    elif OUT_FP16:
        out_block = out_block.to(tl.float16)
    elif OUT_BF16:
        out_block = out_block.to(tl.bfloat16)
    else:
        # default to float32 if nothing specified
        out_block = out_block.to(tl.float32)

    # store block
    c_ptr = C + (m_off[:, None] * N + n_off[None, :])
    store_mask = (m_off[:, None] < M) & (n_off[None, :] < N)
    tl.store(c_ptr, out_block, mask=store_mask)


def kernel(x, w, b, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    # wrapper: compute grid and launch
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    _kernel[grid](
        x,
        w,
        b if b is not None else tl.constexpr(0),
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
