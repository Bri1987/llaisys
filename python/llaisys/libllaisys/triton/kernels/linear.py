import triton
import triton.language as tl

@triton.jit
def _kernel(
    # 指针
    X_ptr, W_ptr, Bias_ptr, Out_ptr,
    # 形状
    M, N, K,
    # Strides (步长)
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    # Block 参数 (这里作为参数传入，不再是 Autotune)
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    # 精度控制
    OUT_FP32: tl.constexpr,
):
    # --- 1. Grid 坐标计算 ---
    # 最简单的 2D Grid 映射，不使用 swizzle
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算当前 Block 在 M 和 N 方向的起始偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # --- 2. 指针计算 ---
    # 计算 X 和 W 的基础指针位置
    # X 形状 [M, K], 取 [BLOCK_M, BLOCK_K] 的块
    # W 形状 [N, K], 取 [BLOCK_N, BLOCK_K] 的块
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    # --- 3. 矩阵乘法主循环 ---
    # 初始化累加器为 fp32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 处理 K 维度边界 (当 K 不是 BLOCK_K 倍数时)
        current_k_len = K - k * BLOCK_K
        k_mask = offs_k < current_k_len

        # 加载 X Block: 检查 M 边界和 K 边界
        # other=0.0 非常重要，保证 padding 部分不影响加法
        mask_m = offs_m[:, None] < M
        x = tl.load(x_ptrs, mask=mask_m & k_mask[None, :], other=0.0)

        # 加载 W Block: 检查 N 边界和 K 边界
        mask_n = offs_n[:, None] < N
        w = tl.load(w_ptrs, mask=mask_n & k_mask[None, :], other=0.0)

        # 计算点积: accumulator += x @ w.T
        # W 加载出来是 [BLOCK_N, BLOCK_K], 转置后是 [BLOCK_K, BLOCK_N]
        # X [BLOCK_M, BLOCK_K] @ W^T [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        accumulator += tl.dot(x, tl.trans(w), allow_tf32=True)

        # 指针步进到下一个 K 块
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # --- 4. Bias & Store ---
    
    # 加载 Bias (如果存在)
    if Bias_ptr is not None:
        bias_ptrs = Bias_ptr + offs_n
        bias_mask = offs_n < N
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
        accumulator += bias[None, :]

    # 结果类型转换
    # 始终让 Triton 的 store 来处理最终的 dtype cast
    c = accumulator
    if OUT_FP32:
        c = c.to(tl.float32)
    # else: 保持 float32，store 到 f16/bf16 指针时 Triton 自动转换

    # 计算输出指针并写入
    # Out 形状 [M, N]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    out_ptrs = Out_ptr + (stride_om * offs_m[:, None] + stride_on * offs_n[None, :])
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    tl.store(out_ptrs, c, mask=out_mask)

# ----------- 精度正确的版本
# import triton
# import triton.language as tl

# @triton.jit
# def _kernel(
#     # 指针
#     X_ptr, W_ptr, Bias_ptr, Out_ptr,
#     # 形状
#     M, N, K,
#     # Strides
#     stride_xm, stride_xk,
#     stride_wn, stride_wk,
#     stride_om, stride_on,
#     # Block 参数
#     BLOCK_M: tl.constexpr, 
#     BLOCK_N: tl.constexpr, 
#     BLOCK_K: tl.constexpr,
#     # 精度控制
#     OUT_FP32: tl.constexpr,
# ):
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)
#     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#     offs_k = tl.arange(0, BLOCK_K)

#     x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
#     w_ptrs = W_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

#     accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)

#     for k in range(0, tl.cdiv(K, BLOCK_K)):
#         current_k_len = K - k * BLOCK_K
#         k_mask = offs_k < current_k_len

#         # 加载 X 和 W，此时它们是原始类型 (例如 float32)
#         mask_m = offs_m[:, None] < M
#         x = tl.load(x_ptrs, mask=mask_m & k_mask[None, :], other=0.0)

#         mask_n = offs_n[:, None] < N
#         w = tl.load(w_ptrs, mask=mask_n & k_mask[None, :], other=0.0)

#         # 转换为 float64
#         result_dot = tl.dot(x, tl.trans(w), allow_tf32=False)
#         accumulator += result_dot.to(tl.float64)

#         x_ptrs += BLOCK_K * stride_xk
#         w_ptrs += BLOCK_K * stride_wk

#     # Bias 的处理逻辑也需要类似地转换为 float64
#     if Bias_ptr is not None:
#         bias_ptrs = Bias_ptr + offs_n
#         bias_mask = offs_n < N
#         # 加载原始精度的 bias
#         bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
#         # 转换为 float64 再相加
#         accumulator += bias[None, :].to(tl.float64)

#     # tl.store 会自动处理从 float64 到目标类型的转换
#     c = accumulator
#     out_ptrs = Out_ptr + (stride_om * offs_m[:, None] + stride_on * offs_n[None, :])
#     out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
#     tl.store(out_ptrs, c, mask=out_mask)