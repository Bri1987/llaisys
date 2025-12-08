# file: triton/kernels/rms_norm.py

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

@triton.jit
def _kernel(
    # 接收 Tensor-like 对象
    Inp_tensor, Weight_tensor, Out_tensor,
    M, D, 
    EPS, 
    BLOCK_SIZE: tl.constexpr
):
    """
    每个 program instance (线程块) 处理一行。
    """
    # 当前处理的行号
    row_idx = tl.program_id(0)

    # --- 第一阶段: 计算均方根 (Sum of Squares) ---
    
    # 初始化累加器为 float32 或 float64 以保证精度
    # 对于 RMSNorm，float32 通常足够了
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # 遍历一行中的所有列 (分块进行)
    for col_start in range(0, D, BLOCK_SIZE):
        # 计算当前块的偏移量
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D
        
        # 计算加载地址 (行偏移 + 列偏移)
        row_offset = row_idx * D
        inp_ptrs = Inp_tensor + row_offset + offsets
        
        # 加载数据
        inp_vals = tl.load(inp_ptrs, mask=mask, other=0.0)
        
        # 计算平方并累加
        acc += inp_vals * inp_vals

    # 在块内进行规约，然后得到整个行的总和
    row_sum_sq = tl.sum(acc, axis=0)
    
    # 计算 RMS 分母
    # var = row_sum_sq / D
    # rstd = 1.0 / tl.sqrt(var + EPS)
    # 为了数值稳定性，使用 tl.rsqrt (reciprocal square root)
    rstd = tl.rsqrt(row_sum_sq / D + EPS)

    # --- 第二阶段: 归一化并写入结果 ---

    # 再次遍历所有列
    for col_start in range(0, D, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D
        
        # 计算加载/存储地址
        row_offset = row_idx * D
        inp_ptrs = Inp_tensor + row_offset + offsets
        weight_ptrs = Weight_tensor + offsets # weight 是一维的
        out_ptrs = Out_tensor + row_offset + offsets

        # 加载输入和权重
        inp_vals = tl.load(inp_ptrs, mask=mask, other=0.0)
        weight_vals = tl.load(weight_ptrs, mask=mask, other=0.0)
        
        # 计算最终输出
        # Y = X * (1 / sqrt(mean(X^2) + eps)) * W
        out_vals = inp_vals * rstd * weight_vals
        
        # 存储结果
        tl.store(out_ptrs, out_vals, mask=mask)


def kernel(inp, weight, out, M, D, eps: float = 1e-5, BLOCK_SIZE: int = 1024):
    """
    [原创无 Torch 版] 接收 Wrapper 对象的 RMSNorm 启动器。
    """
    if triton is None:
        raise RuntimeError("Triton not found.")

    # Grid 的大小等于行数 M，因为每个 program instance 处理一行
    grid = (M,)

    # 直接将 wrapper 对象传递给 JIT 内核
    _kernel[grid](
        inp,        # LLAITensorAdapter
        weight,     # LLAITensorAdapter
        out,        # LLAITensorAdapter
        M, D, 
        eps, 
        BLOCK_SIZE=BLOCK_SIZE
    )