"""Triton linear kernel: computes Y = X @ W^T + b.

This module uses Triton only (no PyTorch fallback). Import will fail
with ImportError if Triton is not installed.
"""

import triton
import triton.language as tl


@triton.jit
def _kernel(A, B, C, D, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Strict per-element accumulation to match Torch (correctness prioritized)
    for mm in range(0, BLOCK_M):
        for nn in range(0, BLOCK_N):
            m_idx_scalar = m_start + mm
            n_idx_scalar = n_start + nn
            # mask for valid output element
            valid_out = (m_idx_scalar < M) & (n_idx_scalar < N)
            # accumulate in high precision using a length-1 vector (tl supports vector ops)
            acc_vec = tl.zeros((1,), dtype=tl.float64)
            for kk in range(0, K):
                a_offset = m_idx_scalar * K + kk
                b_offset = n_idx_scalar * K + kk
                a_val = tl.load(A + a_offset, mask=(m_idx_scalar < M) & (kk < K), other=0.0).to(tl.float64)
                b_val = tl.load(B + b_offset, mask=(n_idx_scalar < N) & (kk < K), other=0.0).to(tl.float64)
                acc_vec += a_val * b_val

            # load bias and add
            bias_val = tl.load(D + n_idx_scalar, mask=(n_idx_scalar < N), other=0.0).to(tl.float64)
            out_val = (tl.sum(acc_vec) + bias_val).to(tl.float32)

            # store single element
            c_offset = m_idx_scalar * N + n_idx_scalar
            tl.store(C + c_offset, out_val, mask=valid_out)


def kernel(x, w, b, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    # wrapper: compute grid and launch
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    _kernel[grid](x, w, b if b is not None else tl.constexpr(0), M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
