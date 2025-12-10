try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

@triton.jit
def _kernel(
    Index_ptr,    # 输入索引张量指针
    Weight_ptr,   # 权重表指针 [VocabSize, D]
    Out_ptr,      # 输出张量指针 [N, D]
    D,            # Embedding 维度
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr
):
    """
    [优化版] 2D Tiling Embedding Kernel
    
    Grid 布局:
      program_id(0): 处理 Embedding 维度方向的分块 (cols)
      program_id(1): 处理 Batch/Seq 维度的索引 (rows)
    """
    # 1. 确定当前负责的 Token 行号 (row index)
    pid_row = tl.program_id(1)
    
    # 2. 确定当前负责的 Embedding 维度分块 (col offset)
    pid_col = tl.program_id(0)
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 3. 加载 Token ID
    # 关键优化：每个 Block 只需要加载一次 Scalar 索引，而不是每个元素加载一次
    # Index_ptr 指向 [N]，直接加上 row 偏移
    idx_ptr = Index_ptr + pid_row
    token_id = tl.load(idx_ptr)
    
    # 4. 计算权重和输出的内存地址
    # 假设 Weight 是 Row-Major [Vocab, D]
    # 地址 = (token_id * D) + col_offsets
    w_ptr = Weight_ptr + (token_id * D) + col_offsets
    
    # 输出地址 = (row_index * D) + col_offsets
    o_ptr = Out_ptr + (pid_row * D) + col_offsets
    
    # 5. 边界检查掩码 (处理 D 不是 BLOCK_SIZE 倍数的情况)
    mask = col_offsets < D
    
    # 6. 向量化加载和存储
    # 关键优化：Triton 会自动将其编译为向量指令 (如 ld.global.v4.f32)
    val = tl.load(w_ptr, mask=mask, other=0.0)
    tl.store(o_ptr, val, mask=mask)


def kernel(index, weight, out, N, D, BLOCK_SIZE: int = None):
    """
    [优化版] Embedding 启动器
    """
    if triton is None:
        raise RuntimeError("Triton not found.")

    # 自动选择合适的 Block Size
    # Embedding 维度通常是 2 的幂 (e.g. 1024, 4096)，或者很大
    # 我们希望 Block Size 尽可能大以利用带宽，但不能超过 D 太多导致浪费
    if BLOCK_SIZE is None:
        if D >= 1024:
            BLOCK_SIZE = 1024
        elif D >= 512:
            BLOCK_SIZE = 512
        elif D >= 256:
            BLOCK_SIZE = 256
        else:
            # 向上取整到最近的 2 的幂
            BLOCK_SIZE = triton.next_power_of_2(D)

    # Grid 维度计算
    # X 轴: 覆盖 Embedding 维度 D (所需 block 数量)
    # Y 轴: 覆盖 Batch * SeqLen 维度 N
    grid = (triton.cdiv(D, BLOCK_SIZE), N)

    # 启动 Kernel
    _kernel[grid](
        index,       # LLAITensorAdapter (Triton 会调用 .data_ptr())
        weight,      # LLAITensorAdapter
        out,         # LLAITensorAdapter
        D,
        BLOCK_SIZE=BLOCK_SIZE
    )