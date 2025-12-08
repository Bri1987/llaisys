# file: triton/kernels/swiglu.py

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

@triton.jit
def _kernel(
    # 接收 Tensor-like 对象
    Gate_tensor, Up_tensor, Out_tensor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    [原创无 Torch 版]
    每个 program instance 处理数据的一个块。
    """
    pid = tl.program_id(0)
    
    # --- 计算偏移量和掩码 ---
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # --- 加载数据 ---
    # Triton 会自动从 Wrapper 对象中提取指针和类型
    gate_vals = tl.load(Gate_tensor + offsets, mask=mask, other=0.0)
    up_vals = tl.load(Up_tensor + offsets, mask=mask, other=0.0)
    
    # --- 计算 SwiGLU ---
    # 为了数值稳定性，将计算过程提升到 float32
    gate_f32 = gate_vals.to(tl.float32)
    up_f32 = up_vals.to(tl.float32)
    
    # Sigmoid an Tanh (SiLU / Swish) activation
    silu_gate = gate_f32 * tl.sigmoid(gate_f32)
    
    # Element-wise product
    output_f32 = silu_gate * up_f32
    
    # --- 存储结果 ---
    # 将结果转换回输出张量的原始数据类型
    output_final = output_f32.to(Out_tensor.dtype.element_ty)
    tl.store(Out_tensor + offsets, output_final, mask=mask)


def kernel(gate, up, out, BLOCK_SIZE: int = 1024):
    """
    [原创无 Torch 版] 接收 Wrapper 对象的 SwiGLU 启动器。
    """
    if triton is None:
        raise RuntimeError("Triton not found.")

    # 从 wrapper 对象中获取信息
    n_elements = gate.numel()

    # 计算 grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # 直接将 wrapper 对象传递给 JIT 内核
    _kernel[grid](
        gate,        # LLAITensorAdapter
        up,          # LLAITensorAdapter
        out,         # LLAITensorAdapter
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )