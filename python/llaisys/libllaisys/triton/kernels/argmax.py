import torch
try:
	import triton
	import triton.language as tl
except Exception:
	triton = None


if triton is None:
	# Fallback using torch on host (shouldn't be used when Triton is available)
	def kernel_stage1(vals, partial_vals, partial_idx, n, BLOCK_SIZE=1024):
		vals_flat = vals.view(-1)
		num_blocks = partial_vals.numel()
		for i in range(num_blocks):
			start = i * BLOCK_SIZE
			end = min(start + BLOCK_SIZE, n)
			block = vals_flat[start:end]
			if block.numel() == 0:
				partial_vals[i] = float('-inf')
				partial_idx[i] = -1
			else:
				# manual loop to avoid using torch.max if required
				mval = float('-inf')
				midx = -1
				for j in range(block.numel()):
					v = float(block[j].item())
					if v > mval:
						mval = v
						midx = start + j
				partial_vals[i] = mval
				partial_idx[i] = midx


	def kernel_stage2(partial_vals, partial_idx, max_val, max_idx, m_blocks, BLOCK_SIZE=1024):
		if partial_vals.numel() == 0:
			max_val[0] = float('-inf')
			max_idx[0] = -1
			return
		best_v = float('-inf')
		best_i = -1
		for i in range(partial_vals.numel()):
			v = float(partial_vals[i].item())
			idx = int(partial_idx[i].item())
			if v > best_v:
				best_v = v
				best_i = idx
		max_val[0] = best_v
		max_idx[0] = best_i

else:
	@triton.jit
	def _stage1(vals_ptr, partial_vals_ptr, partial_idx_ptr, n_elements, BLOCK: tl.constexpr):
		pid = tl.program_id(0)
		base = pid * BLOCK
		local_max = -1e20
		local_idx = -1
		# iterate over elements in this block using scalar loads to avoid
		# indexing into Triton vector expressions which can confuse the AST
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
		# single-program reduction over partials
		best_v = -1e20
		best_i = -1
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

