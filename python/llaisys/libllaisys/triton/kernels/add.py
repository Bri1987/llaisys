# file: triton/kernels/add.py

import triton
import triton.language as tl

# JIT 内核 (_kernel)
# 它现在接收的是 Tensor-like 对象，而不是整数指针
@triton.jit
def _kernel(
    A_tensor, 
    B_tensor, 
    C_tensor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements

    # Triton 会自动从 A_tensor 中获取指针
    # 我们仍然需要手动加上偏移量
    a = tl.load(A_tensor + offsets, mask=mask)
    b = tl.load(B_tensor + offsets, mask=mask)
    
    output = a + b
    
    tl.store(C_tensor + offsets, output, mask=mask)


def kernel(a, b, c, BLOCK_SIZE: int = 1024):
    """
    Run elementwise add using Triton on wrapper objects.

    Args:
        a, b, c: Objects that mimic a tensor interface for Triton,
                 like LLAITensorAdapter or torch.Tensor.
    """
    if triton is None:
        raise RuntimeError("Triton not found.")

    n_elements = a.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # 直接将 wrapper 对象传递给 JIT 内核
    _kernel[grid](
        a,  # a 是 LLAITensorAdapter 对象
        b,  # b 是 LLAITensorAdapter 对象
        c,  # c 是 LLAITensorAdapter 对象
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE,
    )