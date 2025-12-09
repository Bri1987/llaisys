import itertools
import math
import triton
import triton.language as tl

# [FIXED] A specialized kernel for small hd, now with GQA support
@triton.jit
def kernel_small_hd(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_stride_z, q_stride_h, q_stride_m, q_stride_k,
    k_stride_z, k_stride_h, k_stride_n, k_stride_k,
    v_stride_z, v_stride_h, v_stride_k, v_stride_n,
    o_stride_z, o_stride_h, o_stride_m, o_stride_n,
    scale, seq_len_q, seq_len_k_v,
    EMB_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    head_index = tl.program_id(1)
    batch_index = tl.program_id(2)

    head_index_kv = head_index // NUM_GROUPS
    
    log2e: tl.constexpr = 1.4426950408889634
    
    batch_head_offset_q = batch_index * q_stride_z + head_index * q_stride_h
    batch_head_offset_k = batch_index * k_stride_z + head_index_kv * k_stride_h
    batch_head_offset_v = batch_index * v_stride_z + head_index_kv * v_stride_h
    batch_head_offset_o = batch_index * o_stride_z + head_index * o_stride_h
    
    q_start = query_tile_index * 16
    m_offsets = q_start + tl.arange(0, 16)

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + batch_head_offset_q, shape=(seq_len_q, EMB_DIM),
        strides=(q_stride_m, q_stride_k), offsets=(q_start, 0),
        block_shape=(16, EMB_DIM), order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + batch_head_offset_k, shape=(EMB_DIM, seq_len_k_v),
        strides=(k_stride_k, k_stride_n), offsets=(0, 0),
        block_shape=(EMB_DIM, 16), order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + batch_head_offset_v, shape=(seq_len_k_v, EMB_DIM),
        strides=(v_stride_k, v_stride_n), offsets=(0, 0),
        block_shape=(16, EMB_DIM), order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + batch_head_offset_o, shape=(seq_len_q, EMB_DIM),
        strides=(o_stride_m, o_stride_n), offsets=(q_start, 0),
        block_shape=(16, EMB_DIM), order=(1, 0),
    )
    
    q = tl.load(q_block_ptr, boundary_check=(0, 1))
    
    acc = tl.zeros((16, EMB_DIM), dtype=tl.float32)
    l_i = tl.zeros((16,), dtype=tl.float32)
    m_i = tl.full((16,), float("-inf"), dtype=tl.float32)
    
    offset = seq_len_k_v - seq_len_q
    for i in range(0, tl.cdiv(seq_len_k_v, 16)):
        k = tl.load(k_block_ptr, boundary_check=(0, 1))
        v = tl.load(v_block_ptr, boundary_check=(0, 1))
        
        n_offsets = i * 16 + tl.arange(0, 16)
        mask = n_offsets < seq_len_k_v

        qk = tl.sum(q[:, None, :] * tl.trans(k)[None, :, :], 2)

        causal_mask = (m_offsets[:, None] + offset) >= n_offsets[None, :]
        combined_mask = mask[None, :] & causal_mask
        qk = tl.where(combined_mask, qk, float("-inf"))
        
        s_ij = qk * scale
        m_ij = tl.maximum(m_i, tl.max(s_ij, 1))
        p = tl.math.exp2((s_ij - m_ij[:, None]) * log2e)
        alpha = tl.math.exp2((m_i - m_ij) * log2e)
        acc = acc * alpha[:, None]
        
        # [ACCURACY FIX]
        acc += tl.sum(p[:, :, None] * v[None, :, :], 1)

        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
        k_block_ptr = tl.advance(k_block_ptr, (0, 16))
        v_block_ptr = tl.advance(v_block_ptr, (16, 0))

    acc = acc / l_i[:, None]
    tl.store(o_block_ptr, acc.to(o_ptr.type.element_ty), boundary_check=(0,))


# [FIXED] Autotuned kernel, now with GQA support
@triton.jit
def kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_stride_z, q_stride_h, q_stride_m, q_stride_k,
    k_stride_z, k_stride_h, k_stride_n, k_stride_k,
    v_stride_z, v_stride_h, v_stride_k, v_stride_n,
    o_stride_z, o_stride_h, o_stride_m, o_stride_n,
    scale, seq_len_q, seq_len_k_v,
    EMB_DIM: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    head_index = tl.program_id(1)
    batch_index = tl.program_id(2)

    head_index_kv = head_index // NUM_GROUPS
    
    log2e: tl.constexpr = 1.4426950408889634
    
    batch_head_offset_q = batch_index * q_stride_z + head_index * q_stride_h
    batch_head_offset_k = batch_index * k_stride_z + head_index_kv * k_stride_h  
    batch_head_offset_v = batch_index * v_stride_z + head_index_kv * v_stride_h
    batch_head_offset_o = batch_index * o_stride_z + head_index * o_stride_h
    
    q_start = query_tile_index * BLOCK_SIZE_M
    m_offsets = q_start + tl.arange(0, BLOCK_SIZE_M)
    
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
    
    acc = tl.zeros((BLOCK_SIZE_M, EMB_DIM), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=tl.float32)
    
    offset = seq_len_k_v - seq_len_q
    for i in range(0, tl.cdiv(seq_len_k_v, BLOCK_SIZE_N)):
        k = tl.load(k_block_ptr, boundary_check=(0, 1))
        v = tl.load(v_block_ptr, boundary_check=(0, 1))
        
        n_offsets = i * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = n_offsets < seq_len_k_v
        qk = tl.dot(q, k)

        if IS_CAUSAL:
            causal_mask = (m_offsets[:, None] + offset) >= n_offsets[None, :]
            combined_mask = mask[None, :] & causal_mask
            qk = tl.where(combined_mask, qk, float("-inf"))
        else:
            qk = tl.where(mask[None, :], qk, float("-inf"))
        
        s_ij = qk * scale
        m_ij = tl.maximum(m_i, tl.max(s_ij, 1))
        p = tl.math.exp2((s_ij - m_ij[:, None]) * log2e)
        alpha = tl.math.exp2((m_i - m_ij) * log2e)
        acc = acc * alpha[:, None]

        # [ACCURACY FIX]
        # Ensure both operands to tl.dot have the same dtype (promote to fp32)
        p_fp32 = tl.cast(p, tl.float32) if getattr(p, 'dtype', None) != tl.float32 else p
        v_fp32 = tl.cast(v, tl.float32) if getattr(v, 'dtype', None) != tl.float32 else v
        acc += tl.dot(p_fp32, v_fp32)
        
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_SIZE_N))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_SIZE_N, 0))

    acc = acc / l_i[:, None]
    tl.store(o_block_ptr, acc.to(o_ptr.type.element_ty), boundary_check=(0, 1))
    
    
@triton.jit
def split_kv_kernel_small_hd(
    q_ptr, k_ptr, v_ptr, past_k_ptr, past_v_ptr,
    scale,
    split_logsumexp_ptr, split_outputs_ptr,
    num_heads, num_groups,
    q_stride_z, q_stride_h, q_stride_m, q_stride_k,
    k_stride_z, k_stride_h, k_stride_n, k_stride_k,
    v_stride_z, v_stride_h, v_stride_n, v_stride_k,
    past_k_stride_z, past_k_stride_h, past_k_stride_n, past_k_stride_k,
    past_v_stride_z, past_v_stride_h, past_v_stride_n, past_v_stride_k,
    o_stride_z, o_stride_h, o_stride_s, o_stride_m, o_stride_k,
    seq_len_q, total_seq_len, current_seq_len, S,
    EMB_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    BLOCK_SIZE_M: tl.constexpr = 16
    BLOCK_SIZE_N: tl.constexpr = 16

    start_m = tl.program_id(0)
    n_split_id = tl.program_id(1)
    batch_head_id = tl.program_id(2)
    off_h = batch_head_id % num_heads
    off_z = batch_head_id // num_heads
    off_hk = off_h // num_groups

    m_offsets = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    k_offsets = tl.arange(0, EMB_DIM)

    final_l_ptr = split_logsumexp_ptr + ((off_z * num_heads + off_h) * S + n_split_id) * seq_len_q + m_offsets
    final_o_ptr = split_outputs_ptr + off_z * o_stride_z + off_h * o_stride_h + n_split_id * o_stride_s + \
                  (m_offsets[:, None] * o_stride_m + k_offsets[None, :] * o_stride_k)

    total_blocks_n = tl.cdiv(total_seq_len, BLOCK_SIZE_N)
    n_split_size = tl.cdiv(total_blocks_n, S) * BLOCK_SIZE_N

    left = n_split_id * n_split_size
    right = tl.minimum(left + n_split_size, total_seq_len)

    m_i = tl.full([BLOCK_SIZE_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_M, EMB_DIM], dtype=tl.float32)

    q_ptr_base = q_ptr + off_z * q_stride_z + off_h * q_stride_h
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr_base, shape=(seq_len_q, EMB_DIM), strides=(q_stride_m, q_stride_k),
        offsets=(start_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, EMB_DIM), order=(1, 0)
    )
    q = tl.load(q_block_ptr, boundary_check=(0,))

    past_seq_len = total_seq_len - current_seq_len
    past_loop_start = left
    past_loop_end = tl.minimum(right, past_seq_len)

    if past_loop_start < past_loop_end:
        past_k_ptr_base = past_k_ptr + off_z * past_k_stride_z + off_hk * past_k_stride_h
        past_v_ptr_base = past_v_ptr + off_z * past_v_stride_z + off_hk * past_v_stride_h

        for n_block_start in range(past_loop_start, past_loop_end, BLOCK_SIZE_N):
            past_k_block_ptr = tl.make_block_ptr(
                base=past_k_ptr_base, shape=(EMB_DIM, past_seq_len), strides=(past_k_stride_k, past_k_stride_n),
                offsets=(0, n_block_start), block_shape=(EMB_DIM, BLOCK_SIZE_N), order=(0, 1)
            )
            past_v_block_ptr = tl.make_block_ptr(
                base=past_v_ptr_base, shape=(past_seq_len, EMB_DIM), strides=(past_v_stride_n, past_v_stride_k),
                offsets=(n_block_start, 0), block_shape=(BLOCK_SIZE_N, EMB_DIM), order=(1, 0)
            )

            k = tl.load(past_k_block_ptr, boundary_check=(1,))
            v = tl.load(past_v_block_ptr, boundary_check=(0,))

            qk_scores = tl.sum(q[:, None, :] * tl.trans(k)[None, :, :], 2) * scale
            
            n_offsets = n_block_start + tl.arange(0, BLOCK_SIZE_N)
            padding_mask = n_offsets[None, :] < past_seq_len
            qk_scores = tl.where(padding_mask, qk_scores, float("-inf"))
            
            m_i_prev = m_i
            m_ij = tl.maximum(m_i_prev, tl.max(qk_scores, 1))
            
            is_m_ij_invalid = m_ij == -float('inf')
            alpha = tl.where(is_m_ij_invalid, 1.0, tl.exp(m_i_prev - m_ij))
            p_ij = tl.where(is_m_ij_invalid, 0.0, tl.exp(qk_scores - m_ij[:, None]))

            acc = acc * alpha[:, None]
            acc += tl.sum(p_ij[:, :, None] * v[None, :, :], 1)
            l_i = l_i * alpha + tl.sum(p_ij, 1)
            m_i = m_ij

    current_loop_start = tl.maximum(left, past_seq_len)
    current_loop_end = right

    if current_loop_start < current_loop_end:
        k_ptr_base = k_ptr + off_z * k_stride_z + off_hk * k_stride_h
        v_ptr_base = v_ptr + off_z * v_stride_z + off_hk * v_stride_h

        for n_block_start in range(current_loop_start, current_loop_end, BLOCK_SIZE_N):
            k_offset = n_block_start - past_seq_len

            k_block_ptr = tl.make_block_ptr(
                base=k_ptr_base, shape=(EMB_DIM, current_seq_len), strides=(k_stride_k, k_stride_n),
                offsets=(0, k_offset), block_shape=(EMB_DIM, BLOCK_SIZE_N), order=(0, 1)
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_ptr_base, shape=(current_seq_len, EMB_DIM), strides=(v_stride_n, v_stride_k),
                offsets=(k_offset, 0), block_shape=(BLOCK_SIZE_N, EMB_DIM), order=(1, 0)
            )

            k = tl.load(k_block_ptr, boundary_check=(1,))
            v = tl.load(v_block_ptr, boundary_check=(0,))

            qk_scores = tl.sum(q[:, None, :] * tl.trans(k)[None, :, :], 2) * scale
            
            n_offsets = n_block_start + tl.arange(0, BLOCK_SIZE_N)
            combined_mask = n_offsets[None, :] < total_seq_len
            if IS_CAUSAL:
                causal_mask = (past_seq_len + m_offsets[:, None]) >= n_offsets[None, :]
                combined_mask = combined_mask & causal_mask
            qk_scores = tl.where(combined_mask, qk_scores, float("-inf"))

            m_i_prev = m_i
            m_ij = tl.maximum(m_i_prev, tl.max(qk_scores, 1))

            is_m_ij_invalid = m_ij == -float('inf')
            alpha = tl.where(is_m_ij_invalid, 1.0, tl.exp(m_i_prev - m_ij))
            p_ij = tl.where(is_m_ij_invalid, 0.0, tl.exp(qk_scores - m_ij[:, None]))
            
            acc = acc * alpha[:, None]
            acc += tl.sum(p_ij[:, :, None] * v[None, :, :], 1)
            l_i = l_i * alpha + tl.sum(p_ij, 1)
            m_i = m_ij
    
    is_l_i_zero = l_i == 0.0
    inverse_l_i = tl.where(is_l_i_zero, 0.0, 1.0 / l_i)
    final_output = acc * inverse_l_i[:, None]
    final_logsumexp = tl.where(is_l_i_zero, -float("inf"), m_i + tl.log(l_i))

    row_mask = m_offsets < seq_len_q
    tl.store(final_l_ptr, final_logsumexp, mask=row_mask)
    tl.store(final_o_ptr, final_output.to(split_outputs_ptr.type.element_ty), mask=row_mask[:, None])
    

@triton.autotune(
    configs=tuple(
        triton.Config(
            {"BLOCK_SIZE_M": block_size_m, "BLOCK_SIZE_N": block_size_n},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_size_m, block_size_n, num_stages, num_warps in itertools.product(
            (16, 32, 64, 128),
            (32, 64, 128),
            (1, 2, 4, 5),
            (4, 8)
        )
    ),
    key=["EMB_DIM", "M_BINNED"],
)
@triton.jit
def split_kv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    past_k_ptr,
    past_v_ptr,
    scale,
    split_logsumexp_ptr, split_outputs_ptr,
    num_heads, num_groups,
    q_stride_z, q_stride_h, q_stride_m, q_stride_k,
    k_stride_z, k_stride_h, k_stride_n, k_stride_k,
    v_stride_z, v_stride_h, v_stride_n, v_stride_k,
    past_k_stride_z, past_k_stride_h, past_k_stride_n, past_k_stride_k,
    past_v_stride_z, past_v_stride_h, past_v_stride_n, past_v_stride_k,
    o_stride_z, o_stride_h, o_stride_s, o_stride_m, o_stride_k,
    seq_len_q,
    total_seq_len,
    current_seq_len,
    S, 
    EMB_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    M_BINNED: tl.constexpr, N_BINNED: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    n_split_id = tl.program_id(1)
    batch_head_id = tl.program_id(2)
    off_h = batch_head_id % num_heads
    off_z = batch_head_id // num_heads
    off_hk = off_h // num_groups

    m_offsets = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    k_offsets = tl.arange(0, EMB_DIM)

    final_l_ptr = split_logsumexp_ptr + ((off_z * num_heads + off_h) * S + n_split_id) * seq_len_q + m_offsets
    final_o_ptr = split_outputs_ptr + off_z * o_stride_z + off_h * o_stride_h + n_split_id * o_stride_s + \
                  (m_offsets[:, None] * o_stride_m + k_offsets[None, :] * o_stride_k)

    total_blocks_n = (total_seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    n_split_size = ((total_blocks_n + S - 1) // S) * BLOCK_SIZE_N
    
    left = n_split_id * n_split_size
    right = tl.minimum(left + n_split_size, total_seq_len)

    m_i = tl.full([BLOCK_SIZE_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_M, EMB_DIM], dtype=tl.float32)
    
    q_ptr_base = q_ptr + off_z * q_stride_z + off_h * q_stride_h
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr_base, shape=(seq_len_q, EMB_DIM), strides=(q_stride_m, q_stride_k),
        offsets=(start_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, EMB_DIM), order=(1, 0)
    )
    q = tl.load(q_block_ptr, boundary_check=(0,))

    past_seq_len = total_seq_len - current_seq_len
    past_loop_start = left
    past_loop_end = tl.minimum(right, past_seq_len)
    
    if past_loop_start < past_loop_end:
        past_k_ptr_base = past_k_ptr + off_z * past_k_stride_z + off_hk * past_k_stride_h
        past_v_ptr_base = past_v_ptr + off_z * past_v_stride_z + off_hk * past_v_stride_h
        
        for n_block_start in range(past_loop_start, past_loop_end, BLOCK_SIZE_N):
            past_k_block_ptr = tl.make_block_ptr(
                base=past_k_ptr_base, shape=(EMB_DIM, past_seq_len), strides=(past_k_stride_k, past_k_stride_n),
                offsets=(0, n_block_start), block_shape=(EMB_DIM, BLOCK_SIZE_N), order=(0, 1)
            )
            past_v_block_ptr = tl.make_block_ptr(
                base=past_v_ptr_base, shape=(past_seq_len, EMB_DIM), strides=(past_v_stride_n, past_v_stride_k),
                offsets=(n_block_start, 0), block_shape=(BLOCK_SIZE_N, EMB_DIM), order=(1, 0)
            )
            
            k = tl.load(past_k_block_ptr, boundary_check=(1,))
            v = tl.load(past_v_block_ptr, boundary_check=(0,))
            
            qk_scores = tl.dot(q, k) * scale
            
            n_offsets = n_block_start + tl.arange(0, BLOCK_SIZE_N)
            padding_mask = n_offsets[None, :] < past_seq_len
            qk_scores = tl.where(padding_mask, qk_scores, float("-inf"))
            
            m_i_prev = m_i
            m_ij = tl.maximum(m_i_prev, tl.max(qk_scores, 1))
            
            is_m_ij_invalid = m_ij == -float('inf')
            alpha = tl.where(is_m_ij_invalid, 1.0, tl.exp(m_i_prev - m_ij))
            p_ij = tl.where(is_m_ij_invalid, 0.0, tl.exp(qk_scores - m_ij[:, None]))

            acc = acc * alpha[:, None]
            # dot requires same dtypes; promote to fp32
            p_fp32 = tl.cast(p_ij, tl.float32) if getattr(p_ij, 'dtype', None) != tl.float32 else p_ij
            v_fp32 = tl.cast(v, tl.float32) if getattr(v, 'dtype', None) != tl.float32 else v
            acc += tl.dot(p_fp32, v_fp32)
            l_i = l_i * alpha + tl.sum(p_ij, 1)
            m_i = m_ij

    current_loop_start = tl.maximum(left, past_seq_len)
    current_loop_end = right

    if current_loop_start < current_loop_end:
        k_ptr_base = k_ptr + off_z * k_stride_z + off_hk * k_stride_h
        v_ptr_base = v_ptr + off_z * v_stride_z + off_hk * v_stride_h
        
        for n_block_start in range(current_loop_start, current_loop_end, BLOCK_SIZE_N):
            k_offset = n_block_start - past_seq_len
            
            k_block_ptr = tl.make_block_ptr(
                base=k_ptr_base, shape=(EMB_DIM, current_seq_len), strides=(k_stride_k, k_stride_n),
                offsets=(0, k_offset), block_shape=(EMB_DIM, BLOCK_SIZE_N), order=(0, 1)
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_ptr_base, shape=(current_seq_len, EMB_DIM), strides=(v_stride_n, v_stride_k),
                offsets=(k_offset, 0), block_shape=(BLOCK_SIZE_N, EMB_DIM), order=(1, 0)
            )

            k = tl.load(k_block_ptr, boundary_check=(1,))
            v = tl.load(v_block_ptr, boundary_check=(0,))

            qk_scores = tl.dot(q, k) * scale
            
            n_offsets = n_block_start + tl.arange(0, BLOCK_SIZE_N)
            combined_mask = n_offsets[None, :] < total_seq_len
            if IS_CAUSAL:
                causal_mask = (m_offsets[:, None] + past_seq_len) >= n_offsets[None, :]
                combined_mask = combined_mask & causal_mask
            qk_scores = tl.where(combined_mask, qk_scores, float("-inf"))

            m_i_prev = m_i
            m_ij = tl.maximum(m_i_prev, tl.max(qk_scores, 1))

            is_m_ij_invalid = m_ij == -float('inf')
            alpha = tl.where(is_m_ij_invalid, 1.0, tl.exp(m_i_prev - m_ij))
            p_ij = tl.where(is_m_ij_invalid, 0.0, tl.exp(qk_scores - m_ij[:, None]))
            
            acc = acc * alpha[:, None]
            # dot requires same dtypes; promote to fp32
            p_fp32 = tl.cast(p_ij, tl.float32) if getattr(p_ij, 'dtype', None) != tl.float32 else p_ij
            v_fp32 = tl.cast(v, tl.float32) if getattr(v, 'dtype', None) != tl.float32 else v
            acc += tl.dot(p_fp32, v_fp32)
            l_i = l_i * alpha + tl.sum(p_ij, 1)
            m_i = m_ij

    is_l_i_zero = l_i == 0.0
    inverse_l_i = tl.where(is_l_i_zero, 0.0, 1.0 / l_i)
    final_output = acc * inverse_l_i[:, None]
    final_logsumexp = tl.where(is_l_i_zero, -float("inf"), m_i + tl.log(l_i))
    
    if IS_CAUSAL and seq_len_q > total_seq_len:
        empty_row_mask = (m_offsets + past_seq_len) < 0
        final_output = tl.where(empty_row_mask[:, None], 0.0, final_output)
        final_logsumexp = tl.where(empty_row_mask, float("-inf"), final_logsumexp)

    row_mask = m_offsets < seq_len_q
    tl.store(final_l_ptr, final_logsumexp, mask=row_mask)
    tl.store(final_o_ptr, final_output.to(split_outputs.type.element_ty), mask=row_mask[:, None])        
    
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": block_size_m},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_size_m, num_stages, num_warps in itertools.product(
            (16, 32, 64, 128),
            (1, 2, 3, 4, 5),
            (4, 8)
        )
    ],
    key=["EMB_DIM", "S", "M_BINNED"],
)
@triton.jit
def combine_kv_splits_kernel(
    split_outputs, split_logsumexp,
    final_o, final_l,
    num_heads,
    stride_split_oz, stride_split_oh, stride_split_os, stride_split_om, stride_split_ok,
    stride_fin_oz, stride_fin_oh, stride_fin_om, stride_fin_ok,
    seq_len_q, S,
    EMB_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    M_BINNED: tl.constexpr,
):
    m_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    z_idx = tl.program_id(2)

    m_offsets = m_block_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    k_offsets = tl.arange(0, EMB_DIM)
    row_mask = m_offsets < seq_len_q

    base_split_l_ptr = split_logsumexp + (z_idx * num_heads + h_idx) * S * seq_len_q
    base_split_o_ptr = split_outputs + z_idx * stride_split_oz + h_idx * stride_split_oh
    
    base_final_l_ptr = final_l + (z_idx * num_heads + h_idx) * seq_len_q
    base_final_o_ptr = final_o + z_idx * stride_fin_oz + h_idx * stride_fin_oh
    final_o_ptr = base_final_o_ptr + m_offsets[:, None] * stride_fin_om + k_offsets[None, :] * stride_fin_ok
    final_l_ptr = base_final_l_ptr + m_offsets

    acc_o = tl.zeros([BLOCK_SIZE_M, EMB_DIM], dtype=tl.float32)
    acc_l = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_SIZE_M], value=-float("-inf"), dtype=tl.float32)

    for s_idx in range(S):
        current_l_ptr = base_split_l_ptr + s_idx * seq_len_q + m_offsets
        current_o_ptr = base_split_o_ptr + s_idx * stride_split_os + \
                        m_offsets[:, None] * stride_split_om + k_offsets[None, :]
        
        l_j = tl.load(current_l_ptr, mask=row_mask, other=-float('inf'))
        o_j = tl.load(current_o_ptr, mask=row_mask[:, None])

        m_new = tl.maximum(m_i, l_j)
        
        # Guard against exp(inf) = inf
        P_j_scaled = tl.exp(l_j - m_new)
        rescale_factor = tl.exp(m_i - m_new)
        
        acc_o = acc_o * rescale_factor[:, None] + o_j * P_j_scaled[:, None]
        acc_l = acc_l * rescale_factor + P_j_scaled
        m_i = m_new

    # [NAN-FIX] Robust finalization
    is_zero_mask = acc_l == 0.0
    final_l_val = tl.where(is_zero_mask, -float("inf"), m_i + tl.log(acc_l))
    inverse_acc_l = tl.where(is_zero_mask, 0.0, 1.0 / acc_l)
    final_o_val = acc_o * inverse_acc_l[:, None]
    
    tl.store(final_o_ptr, final_o_val.to(split_outputs.type.element_ty), mask=row_mask[:, None])
    tl.store(final_l_ptr, final_l_val, mask=row_mask)