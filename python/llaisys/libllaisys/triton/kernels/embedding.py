# file: triton/kernels/embedding.py

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

@triton.jit
def _kernel(
    # 接收 Tensor-like 对象
    Index_tensor, Weight_tensor, Out_tensor,
    N, D, 
    TOTAL_ELEMENTS, 
    BLOCK_SIZE: tl.constexpr
):
    """
    [原创无 Torch 版]
    每个 program instance 处理输出张量中的一个数据块。
    """
    # 当前 program instance 处理的全局偏移量 (在扁平化的输出张量中)
    pid = tl.program_id(0)
    global_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = global_offsets < TOTAL_ELEMENTS

    # --- 从全局偏移量计算出 2D 的 (row, col) 坐标 ---
    # row 对应要查询的第几个词 (0 to N-1)
    # col 对应 embedding 的维度 (0 to D-1)
    row_indices = global_offsets // D
    col_indices = global_offsets % D
    
    # --- 加载需要查询的 token ID ---
    # 使用 row_indices (去重后) 从 Index_tensor 加载
    # 注意: 这里的加载有冗余，因为同一个 row_idx 会被加载多次
    # 但这是最简单的实现方式
    token_ids = tl.load(Index_tensor + row_indices, mask=mask)

    # --- 计算源地址 (在 weight tensor 中) ---
    # 地址 = token_id * D + col_idx
    src_offsets = token_ids * D + col_indices

    # 从 weight tensor 中加载 embedding 数据
    embedding_vals = tl.load(Weight_tensor + src_offsets, mask=mask, other=0.0)

    # --- 将结果写入输出张量 ---
    # 输出张量的地址就是我们一开始的全局偏移量
    tl.store(Out_tensor + global_offsets, embedding_vals, mask=mask)


def kernel(index, weight, out, N, D, BLOCK_SIZE: int = 1024):
    """
    [原创无 Torch 版] 接收 Wrapper 对象的 Embedding 启动器。
    """
    if triton is None:
        raise RuntimeError("Triton not found.")

    # 总共要计算的元素数量
    total_elements = N * D
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    # 直接将 wrapper 对象传递给 JIT 内核
    _kernel[grid](
        index,       # LLAITensorAdapter
        weight,      # LLAITensorAdapter
        out,         # LLAITensorAdapter
        N, D,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )