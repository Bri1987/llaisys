import triton
import triton.language as tl

@triton.jit
def _argmax_combine(val1, idx1, val2, idx2):
    """
    自定义归约逻辑：
    返回较大的值；如果值相等，返回索引较小的那个（保持稳定性）。
    """
    gt = val1 > val2
    # 如果 val1 > val2，选 val1，否则选 val2
    res_val = tl.where(gt, val1, val2)
    # 如果 val1 > val2，选 idx1，否则选 idx2
    res_idx = tl.where(gt, idx1, idx2)
    return res_val, res_idx

@triton.jit
def _kernel_stage1(
    vals_ptr, 
    partial_vals_ptr, 
    partial_idx_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    """
    Stage 1: 每个 Block 处理 BLOCK_SIZE 个元素，输出 1 个局部最大值和索引。
    """
    pid = tl.program_id(0)
    # 计算当前 Block 负责的全局偏移量
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 掩码：防止越界
    mask = offsets < n_elements

    # 1. 并行加载数据
    # 使用 -inf 填充越界部分，确保它们不会被选为最大值
    vals = tl.load(vals_ptr + offsets, mask=mask, other=float("-inf"))
    
    # 生成对应的全局索引
    idxs = offsets

    # 2. Block 内并行归约
    # 对 (value, index) 元组进行 reduce
    max_val, max_idx = tl.reduce((vals, idxs), axis=0, combine_fn=_argmax_combine)

    # 3. 写入 Partial 结果
    # 每个 Block 只写回一个标量
    tl.store(partial_vals_ptr + pid, max_val)
    tl.store(partial_idx_ptr + pid, max_idx)


@triton.jit
def _kernel_stage2(
    partial_vals_ptr, 
    partial_idx_ptr, 
    max_val_ptr, 
    max_idx_ptr, 
    n_partials, 
    BLOCK_SIZE: tl.constexpr
):
    """
    Stage 2: 读取 Stage 1 的所有输出，归约为最终结果。
    注意：这里假设 partial 数量 <= BLOCK_SIZE (通常 < 1024)。
    """
    # Stage 2 通常只有一个 Block (grid=[1])
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partials

    # 1. 加载所有 Partial 结果
    # 注意：Stage 2 不仅加载值，还要加载 Stage 1 算出的全局索引
    vals = tl.load(partial_vals_ptr + offsets, mask=mask, other=float("-inf"))
    idxs = tl.load(partial_idx_ptr + offsets, mask=mask, other=0) # other无所谓，因为val是-inf

    # 2. 最终归约
    final_val, final_idx = tl.reduce((vals, idxs), axis=0, combine_fn=_argmax_combine)

    # 3. 写入最终标量结果
    tl.store(max_val_ptr, final_val)
    tl.store(max_idx_ptr, final_idx)


def kernel_stage1(vals, partial_vals, partial_idx, n, BLOCK_SIZE=1024):
    """
    Python Launcher for Stage 1
    """
    # 计算需要的 Block 数量
    grid = (triton.cdiv(n, BLOCK_SIZE), )
    
    _kernel_stage1[grid](
        vals, 
        partial_vals, 
        partial_idx, 
        n, 
        BLOCK_SIZE=BLOCK_SIZE
    )


def kernel_stage2(partial_vals, partial_idx, max_val, max_idx, m_blocks, BLOCK_SIZE=1024):
    """
    Python Launcher for Stage 2
    """
    # 这里的 BLOCK_SIZE 必须大于等于 m_blocks (Stage 1 的输出数量)
    # 找到最近的 2 的幂次以适应 Triton
    input_size = m_blocks
    # 简单的 heuristic: 确保 BLOCK_SIZE 足够大
    # 如果 input_size 很小，用小一点的 block 也可以，但最大不超过显卡限制 (通常 1024)
    # 为了简单，我们这里复用传入的 BLOCK_SIZE，但在调用处应确保 BLOCK_SIZE >= m_blocks
    
    # 如果传入的 BLOCK_SIZE 小于 m_blocks，强制增大 (虽然通常外部调用逻辑会控制)
    curr_block = 128
    while curr_block < input_size:
        curr_block *= 2
    
    # Grid 为 1，因为我们要做全局归约
    _kernel_stage2[(1,)](
        partial_vals, 
        partial_idx, 
        max_val, 
        max_idx, 
        m_blocks, 
        BLOCK_SIZE=curr_block
    )