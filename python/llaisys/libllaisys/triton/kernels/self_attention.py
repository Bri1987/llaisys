import itertools
import math
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
			(4, 8)       
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
	IS_CAUSAL: tl.constexpr,
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
    
	# 初始化累加器
	acc = tl.zeros((BLOCK_SIZE_M, EMB_DIM), dtype=tl.float32)
	l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
	m_i = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=tl.float32)
    
	offset = seq_len_k_v - seq_len_q
	for i in range(0, tl.cdiv(seq_len_k_v, BLOCK_SIZE_N)):
		k = tl.load(k_block_ptr, boundary_check=(0, 1))
		v = tl.load(v_block_ptr, boundary_check=(0, 1))
        
		# 计算 QK^T 并应用 scale 因子
		n_offsets = i * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
		mask = n_offsets < seq_len_k_v
		qk = tl.dot(q, k)
		if IS_CAUSAL:
			causal_mask = (m_offsets[:, None] + offset) >= n_offsets[None, :]
			qk = tl.where(mask[None, :].to(tl.bool) & causal_mask, qk, float("-inf"))
		else:
			qk = tl.where(mask[None, :].to(tl.bool), qk, float("-inf"))
		s_ij = qk * scale
        
		# 结合历史最大值 m_i，寻找当前块的新最大值 m_ij
		m_ij = tl.maximum(m_i, tl.max(s_ij, 1))
        
		# 计算缩放后的 softmax 概率 p
		p = tl.math.exp2((s_ij - m_ij[:, None]) * log2e)
        
		# 计算用于更新旧累加器的缩放因子 alpha
		alpha = tl.math.exp2((m_i - m_ij) * log2e)
        
		# 更新累加器 acc 和归一化因子 l_i
		acc = acc * alpha[:, None]
		acc += tl.dot(p.to(v.dtype), v)
        
		l_ij = tl.sum(p, 1)
		l_i = l_i * alpha + l_ij
		m_i = m_ij
        
		k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_SIZE_N))
		v_block_ptr = tl.advance(v_block_ptr, (BLOCK_SIZE_N, 0))

	acc = acc / l_i[:, None]
	tl.store(o_block_ptr, acc.to(o_ptr.type.element_ty), boundary_check=(0, 1))


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
	seq_len_k_v,
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

	total_blocks_n = (seq_len_k_v + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
	n_split_size = ((total_blocks_n + S - 1) // S) * BLOCK_SIZE_N
    
	left = n_split_id * n_split_size
	right = tl.minimum(left + n_split_size, seq_len_k_v)

	# --- 初始化 Online Softmax 状态 ---
	max_log_score = tl.full([BLOCK_SIZE_M], value=-float("-inf"), dtype=tl.float32)
	sum_exp_score = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
	output_accumulator = tl.zeros([BLOCK_SIZE_M, EMB_DIM], dtype=tl.float32)

	log2e: tl.constexpr = 1.4426950408889634
	qk_scale = scale * log2e
    
	# --- 加载 q ---
	q_ptr_base = q_ptr + off_z * q_stride_z + off_h * q_stride_h
	q_block_ptr = tl.make_block_ptr(
		base=q_ptr_base, shape=(seq_len_q, EMB_DIM), strides=(q_stride_m, q_stride_k),
		offsets=(start_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, EMB_DIM), order=(1, 0)
	)
	q = tl.load(q_block_ptr, boundary_check=(0,))

	# --- 循环 1: 处理 Past K/V ---
	past_seq_len = seq_len_k_v - current_seq_len
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
            
			qk_scores = tl.dot(q, k)
            
			new_max_score = tl.maximum(max_log_score, tl.max(qk_scores, 1))
			rescale_factor = tl.math.exp2((max_log_score - new_max_score) * qk_scale)
			softmax_probs = tl.math.exp2(qk_scores * qk_scale - new_max_score[:, None] * qk_scale)
			output_accumulator *= rescale_factor[:, None]
			output_accumulator += tl.dot(softmax_probs.to(tl.float16), v)
			sum_exp_score = sum_exp_score * rescale_factor + tl.sum(softmax_probs, 1)
			max_log_score = new_max_score

	# --- 循环 2: 处理 Current K/V ---
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

			qk_scores = tl.dot(q, k)
            
			if IS_CAUSAL:
				n_offsets = n_block_start + tl.arange(0, BLOCK_SIZE_N)
				causal_mask = (m_offsets[:, None] + past_seq_len) >= n_offsets[None, :]
				qk_scores = tl.where(causal_mask, qk_scores, float("-inf"))

			new_max_score = tl.maximum(max_log_score, tl.max(qk_scores, 1))
			rescale_factor = tl.math.exp2((max_log_score - new_max_score) * qk_scale)
			softmax_probs = tl.math.exp2(qk_scores * qk_scale - new_max_score[:, None] * qk_scale)
			output_accumulator *= rescale_factor[:, None]
			output_accumulator += tl.dot(softmax_probs.to(tl.float16), v)
			sum_exp_score = sum_exp_score * rescale_factor + tl.sum(softmax_probs, 1)
			max_log_score = new_max_score

	inverse_l_i = 1.0 / sum_exp_score
	final_output = output_accumulator * inverse_l_i[:, None]
	final_logsumexp = max_log_score * scale + tl.log(sum_exp_score)
    
	if IS_CAUSAL and seq_len_q > seq_len_k_v:
		empty_row_mask = (m_offsets + past_seq_len) < 0
		final_output = tl.where(empty_row_mask[:, None], 0.0, final_output)
		final_logsumexp = tl.where(empty_row_mask, float("-inf"), final_logsumexp)

	row_mask = m_offsets < seq_len_q
                  
	tl.store(final_l_ptr, final_logsumexp, mask=row_mask)
	tl.store(final_o_ptr, final_output.to(tl.float16), mask=row_mask[:, None])

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
	# --- 1. 获取程序ID ---
	m_block_idx = tl.program_id(0)
	h_idx = tl.program_id(1)
	z_idx = tl.program_id(2)

	# --- 2. 准备基础指针和偏移量 ---
	m_offsets = m_block_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
	k_offsets = tl.arange(0, EMB_DIM)
	row_mask = m_offsets < seq_len_q

	#  预先计算当前 (Batch, Head) slice 的基地址
	base_split_l_ptr = split_logsumexp + (z_idx * num_heads + h_idx) * S * seq_len_q
	base_split_o_ptr = split_outputs + z_idx * stride_split_oz + h_idx * stride_split_oh
    
	base_final_l_ptr = final_l + (z_idx * num_heads + h_idx) * seq_len_q
	base_final_o_ptr = final_o + z_idx * stride_fin_oz + h_idx * stride_fin_oh
	final_o_ptr = base_final_o_ptr + m_offsets[:, None] * stride_fin_om + k_offsets[None, :] * stride_fin_ok
	final_l_ptr = base_final_l_ptr + m_offsets

	# --- 3. 初始化累加器 ---
	acc_o = tl.zeros([BLOCK_SIZE_M, EMB_DIM], dtype=tl.float32)
	acc_l = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
	m_i = tl.full([BLOCK_SIZE_M], value=-float("-inf"), dtype=tl.float32)

	# --- 4. 遍历所有 split，在线更新 acc_o 和 acc_l ---
	for s_idx in range(S):
		current_l_ptr = base_split_l_ptr + s_idx * seq_len_q + m_offsets
		current_o_ptr = base_split_o_ptr + s_idx * stride_split_os + \
						m_offsets[:, None] * stride_split_om + k_offsets[None, :]
        
		l_j = tl.load(current_l_ptr, mask=row_mask, other=-float('inf'))
		o_j = tl.load(current_o_ptr, mask=row_mask[:, None])

		# --- 在线 softmax ---
		m_new = tl.maximum(m_i, l_j)
		P_j_scaled = tl.exp(l_j - m_new)
		rescale_factor = tl.exp(m_i - m_new)
		acc_o = acc_o * rescale_factor[:, None] + o_j * P_j_scaled[:, None]
		acc_l = acc_l * rescale_factor + P_j_scaled
		m_i = m_new

	# --- 5. 计算最终的 l 和 o ---
	final_l_val = m_i + tl.log(acc_l)
	final_o_val = acc_o / acc_l[:, None]
    
	tl.store(final_o_ptr, final_o_val.to(final_o.type.element_ty), mask=row_mask[:, None])
	tl.store(final_l_ptr, final_l_val, mask=row_mask)

