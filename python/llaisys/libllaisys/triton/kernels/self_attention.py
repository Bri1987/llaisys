"""Triton self-attention (scaled dot-product, prefill) kernel.

This kernel is adapted from the reference implementation in the HW2
assignment. It's Triton-only and contains a numerically-stable
softmax accumulation across KV blocks. The kernel expects tensors in
shape (batch, heads, seq_len, emb_dim) with appropriate strides.
"""

import itertools
import triton
import triton.language as tl


@triton.autotune(
    configs=tuple(
        triton.Config(
            {"BLOCK_SIZE_M": block_size_m, "BLOCK_SIZE_N": block_size_n},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_size_m, block_size_n, num_stages, num_warps in itertools.product(
            (32, 64, 128, 256),
            (32, 64, 128),
            (2, 3, 4, 5),
            (4, 8),
        )
    ),
    key=["EMB_DIM", "seq_len_k_v"],
)
@triton.jit
def kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_stride_z, q_stride_h, q_stride_m, q_stride_k,
    k_stride_z, k_stride_h, k_stride_n, k_stride_k,
    v_stride_z, v_stride_h, v_stride_k, v_stride_n,
    o_stride_z, o_stride_h, o_stride_m, o_stride_n,
    scale, seq_len_q, seq_len_k_v,
    EMB_DIM: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    head_index = tl.program_id(1)
    batch_index = tl.program_id(2)

    log2e: tl.constexpr = 1.4426950408889634

    batch_head_offset_q = batch_index * q_stride_z + head_index * q_stride_h
    batch_head_offset_k = batch_index * k_stride_z + head_index * k_stride_h
    batch_head_offset_v = batch_index * v_stride_z + head_index * v_stride_h
    batch_head_offset_o = batch_index * o_stride_z + head_index * o_stride_h

    q_start = query_tile_index * BLOCK_SIZE_M

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + batch_head_offset_q,
        shape=(seq_len_q, EMB_DIM),
        strides=(q_stride_m, q_stride_k),
        offsets=(q_start, 0),
        block_shape=(BLOCK_SIZE_M, EMB_DIM),
        order=(1, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + batch_head_offset_k,
        shape=(EMB_DIM, seq_len_k_v),
        strides=(k_stride_k, k_stride_n),
        offsets=(0, 0),
        block_shape=(EMB_DIM, BLOCK_SIZE_N),
        order=(0, 1),
    )

    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + batch_head_offset_v,
        shape=(seq_len_k_v, EMB_DIM),
        strides=(v_stride_k, v_stride_n),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_N, EMB_DIM),
        order=(1, 0),
    )

    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + batch_head_offset_o,
        shape=(seq_len_q, EMB_DIM),
        strides=(o_stride_m, o_stride_n),
        offsets=(q_start, 0),
        block_shape=(BLOCK_SIZE_M, EMB_DIM),
        order=(1, 0),
    )

    q = tl.load(q_block_ptr, boundary_check=(0, 1))

    # initialize accumulators: output acc, normalizer l_i and max m_i
    acc = tl.zeros((BLOCK_SIZE_M, EMB_DIM), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=tl.float32)

    for i in range(0, tl.cdiv(seq_len_k_v, BLOCK_SIZE_N)):
        k = tl.load(k_block_ptr, boundary_check=(0, 1))
        v = tl.load(v_block_ptr, boundary_check=(0, 1))

        # compute QK^T for this block and apply scale
        mask = i * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) < seq_len_k_v
        qk = tl.where(mask, tl.dot(q, k), float("-inf"))
        s_ij = qk * scale

        # compute new max per row and adjust for numerical stability
        m_ij = tl.maximum(m_i, tl.max(s_ij, 1))

        # p = exp(s_ij - m_ij)
        p = tl.math.exp2((s_ij - m_ij[:, None]) * log2e)

        # alpha = exp(m_i - m_ij)
        alpha = tl.math.exp2((m_i - m_ij) * log2e)

        # update accumulators
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_SIZE_N))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_SIZE_N, 0))

    acc = acc / l_i[:, None]
    # store result (let Triton handle dtype casting)
    tl.store(o_block_ptr, acc, boundary_check=(0, 1))
