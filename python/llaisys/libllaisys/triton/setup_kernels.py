from .kernels import add as add_kernel
from .kernels import argmax as argmax_kernel
from .kernels import embedding as embedding_kernel
from .kernels import linear as linear_kernel
from .kernels import rms_norm as rms_norm_kernel    
from .kernels import rope as rope_kernel
from .kernels import self_attention as self_attention_kernel
from .kernels import swiglu as swiglu_kernel

# import torch

import ctypes
import numpy as _np
import torch
import math

import os
from llaisys.runtime import RuntimeAPI
from llaisys.libllaisys import DeviceType, MemcpyKind, DataType, LIB_LLAISYS
# scaled_dot_product_attention_decode may not exist in all kernel sets; import safely
try:
    from .kernels import scaled_dot_product_attention_decode as decode_kernels
except Exception:
    # fallback to self_attention kernel module which exposes the same split/combine kernels
    decode_kernels = self_attention_kernel
import triton
import triton.language as tl
from llaisys.tensor import Tensor

def get_element_size(llaisys_dtype):
    """Returns the size of a llaisys.DataType element in bytes."""
    # 您需要根据您的 DataType 枚举来完善这个映射
    dtype_map = {
        DataType.F32: 4,
        DataType.F16: 2,
        DataType.BF16: 2,
        DataType.I64: 8,
        DataType.I32: 4,
    }
    return dtype_map.get(llaisys_dtype, 4) # 默认为 4 字节 (float32)

class LLAITensorAdapter:
    """Compact adapter for LLAISYS tensors that exposes the small API
        Triton expects.

        - Uses helper properties to lazily read shape/strides/dtype from the
            LLAISYS handle.
        - Provides `is_cuda` and `device` compatibility properties.
        - `dtype` maps to `torch.dtype` when `torch` is available, otherwise
            returns `None`.
        """
    def __init__(self, tensor_like):
        self._handle = _get_raw_ptr(tensor_like)

    # --- low-level readers ---
    def _ndim(self):
        return int(LIB_LLAISYS.tensorGetNdim(self._handle))

    def _read_shape(self):
        n = self._ndim()
        buf = (ctypes.c_size_t * n)()
        LIB_LLAISYS.tensorGetShape(self._handle, buf)
        return tuple(int(buf[i]) for i in range(n))

    def _read_strides(self):
        n = self._ndim()
        try:
            sbuf = (ctypes.c_size_t * n)()
            LIB_LLAISYS.tensorGetStrides(self._handle, sbuf)
            return tuple(int(sbuf[i]) for i in range(n))
        except Exception:
            # fallback to contiguous C-order strides
            s = list(self.shape)
            if not s:
                return ()
            strides = [1]
            for dim in reversed(s[1:]):
                strides.insert(0, strides[0] * dim)
            return tuple(strides)

    def _read_data_ptr(self):
        return int(LIB_LLAISYS.tensorGetData(self._handle))

    def _read_dtype_ll(self):
        return DataType(LIB_LLAISYS.tensorGetDataType(self._handle))

    # --- public API expected by Triton kernels ---
    def data_ptr(self) -> int:
        return self._read_data_ptr()

    @property
    def shape(self):
        return self._read_shape()

    @property
    def strides(self):
        return self._read_strides()

    def numel(self) -> int:
        s = self.shape
        n = 1
        for v in s:
            n *= int(v)
        return n

    def element_size(self) -> int:
        return get_element_size(self._read_dtype_ll())

    @property
    def dtype(self):
        # Lazily map LLAISYS dtype to torch dtype if possible
        ll = self._read_dtype_ll()
        try:
            import torch
            map_tbl = {
                DataType.F32: torch.float32,
                DataType.F16: torch.float16,
                DataType.BF16: torch.bfloat16,
                DataType.I32: torch.int32,
                DataType.I64: torch.int64,
            }
            return map_tbl.get(ll, torch.float32)
        except Exception:
            return None

    # compatibility fields some kernels expect
    @property
    def is_cuda(self):
        # treat all LLAISYS device tensors as CUDA-backed
        return True

    @property
    def device(self):
        try:
            dev = int(LIB_LLAISYS.tensorGetDeviceId(self._handle))
        except Exception:
            dev = 0
        return f"cuda:{dev}"


def get_optimal_s(batch_size: int, num_heads: int, seq_len: int, device: str = 'cuda') -> int:
    # 对于短序列，split/combine的开销占主导，强制S=1
    return 1
    SEQ_LEN_THRESHOLD = 512 
    if seq_len < SEQ_LEN_THRESHOLD:
        return 1
        
    if not torch.cuda.is_available():
        return 1

    properties = torch.cuda.get_device_properties(device)
    num_sms = properties.multi_processor_count

    total_tasks = batch_size * num_heads

    if total_tasks >= num_sms:
        # GPU已经被充分利用或超载，S=1是最佳选择
        return 1
    else:
        # GPU未被充分利用，计算需要的S值以饱和SM
        s_float = num_sms / total_tasks
        
        # 向上取整，确保任务数足够
        s_int = math.ceil(s_float)
        
        # 将S调整为2的幂
        if s_int > 1:
            s_power_of_2 = 2**math.floor(math.log2(s_int))
            # 设个4的上限吧, 否则正确性的精度过不了
            return min(4, s_power_of_2)
        else:
                return 1


def _to_float(x):
    """Normalize various numeric-like inputs (ctypes wrappers, strings, floats) to Python float."""
    try:
        if hasattr(x, 'value'):
            return float(x.value)
        return float(x)
    except Exception:
        try:
            import re

            m = re.search(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", str(x))
            if m:
                return float(m.group(0))
        except Exception:
            pass
    raise TypeError(f"Cannot interpret scale value: {x}")


def _get_raw_ptr(x):
    # Accept either a Python wrapper (with lib_tensor) or a raw handle
    if hasattr(x, "lib_tensor"):
        return x.lib_tensor()
    return x
def concat_device_tensors(a, b):
    """
    [高效版] 沿第一个维度拼接两个 LLAISYS 设备张量，完全在设备上执行。
    此版本使用 Device-to-Device (D2D) 内存复制，避免了任何 Host 端的中转。
    """
    a_adapter = LLAITensorAdapter(a)
    b_adapter = LLAITensorAdapter(b)
    
    a_shape = a_adapter.shape
    b_shape = b_adapter.shape

    # --- 1. 验证形状和数据类型 ---
    if len(a_shape) != len(b_shape) or a_shape[1:] != b_shape[1:]:
        raise RuntimeError(f"Cannot concat tensors with incompatible shapes: {a_shape} vs {b_shape}")
    
    a_dtype = a_adapter._read_dtype_ll()
    b_dtype = b_adapter._read_dtype_ll()
    if a_dtype != b_dtype:
        raise RuntimeError(f"Cannot concat tensors with different dtypes: {a_dtype} vs {b_dtype}")

    # --- 2. 创建输出张量 ---
    out_shape = (a_shape[0] + b_shape[0],) + a_shape[1:]
    device_id = int(a_adapter.device.split(":")[-1])
    out_tensor = Tensor(shape=out_shape, dtype=a_dtype, device=DeviceType.NVIDIA, device_id=device_id)

    # --- 3. 执行 D2D 内存复制 ---
    runtime = RuntimeAPI(DeviceType.NVIDIA)
    elem_size = a_adapter.element_size()
    
    a_size_bytes = a_adapter.numel() * elem_size
    b_size_bytes = b_adapter.numel() * elem_size

    a_data_ptr = ctypes.c_void_p(a_adapter.data_ptr())
    b_data_ptr = ctypes.c_void_p(b_adapter.data_ptr())
    out_data_ptr = ctypes.c_void_p(out_tensor.data_ptr())

    # 复制 a 到 out 的开头
    runtime.memcpy_sync(out_data_ptr, a_data_ptr, a_size_bytes, MemcpyKind.D2D)

    # 计算 b 的目标地址（out 的开头 + a 的大小）
    out_b_offset_ptr = ctypes.c_void_p(out_data_ptr.value + a_size_bytes)
    
    # 复制 b 到 out 的偏移位置
    runtime.memcpy_sync(out_b_offset_ptr, b_data_ptr, b_size_bytes, MemcpyKind.D2D)

    return out_tensor
    
    
def llaisysAdd(out, a, b):
    """
    [无 Torch 版] Launcher that bridges LLAISYS tensors to the Triton add kernel.
    """
    # 1. 将 llaisys.Tensor 包装成 Triton 兼容的对象
    a_wrapped = LLAITensorAdapter(a)
    b_wrapped = LLAITensorAdapter(b)
    out_wrapped = LLAITensorAdapter(out)

    # 2. 直接将 Wrapper 对象传递给 Triton 内核的启动器
    # Triton JIT 将会调用 wrapper 对象的 data_ptr() 和 .dtype 等属性
    add_kernel.kernel(a_wrapped, b_wrapped, out_wrapped, BLOCK_SIZE=1024)

    return out


def llaisysArgmax(max_idx_out, max_val_out, vals):
    """
    [高性能优化版] Argmax Launcher
    优化点:
    1. Zero-Copy: 移除所有 Host <-> Device 数据拷贝。
    2. Native Dtype: 直接处理 F16/BF16 输入，无需 Host 端转换。
    3. Direct Write: Kernel 直接写入输出 Tensor，无需 CPU 中转。
    """
    # 1. 包装输入输出
    # LLAITensorAdapter 会通过 .dtype 属性告诉 Triton 指针的类型
    vals_wr = LLAITensorAdapter(vals)
    idx_out_wr = LLAITensorAdapter(max_idx_out)
    val_out_wr = LLAITensorAdapter(max_val_out)

    # 2. 计算 Grid 和 Block
    N = vals_wr.numel()
    BLOCK_SIZE = 1024 # Stage 1 的 Block Size
    
    # 计算需要的 Block 数量 (向上取整)
    # 相当于 math.ceil(N / BLOCK_SIZE)
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # 3. 分配中间 Buffer (纯 Device 端分配)
    # 我们需要保存 Stage 1 产生的 num_blocks 个局部最大值和索引
    # 使用 DataType.F32 存储中间值以保持精度
    # 使用 DataType.I32 存储中间索引 (假设 N < 21亿)
    
    # 获取设备 ID 以确保在正确的 GPU 上分配
    try:
        device_id = int(vals_wr.device.split(":")[-1])
    except:
        device_id = 0
        
    partial_vals = Tensor(shape=(num_blocks,), dtype=DataType.F32, device=DeviceType.NVIDIA, device_id=device_id)
    partial_idx = Tensor(shape=(num_blocks,), dtype=DataType.I32, device=DeviceType.NVIDIA, device_id=device_id)
    
    p_vals_wr = LLAITensorAdapter(partial_vals.lib_tensor())
    p_idx_wr = LLAITensorAdapter(partial_idx.lib_tensor())

    # 4. 执行 Stage 1
    # 输入: vals -> 输出: partial_vals, partial_idx
    argmax_kernel.kernel_stage1(
        vals_wr, 
        p_vals_wr, 
        p_idx_wr, 
        N, 
        BLOCK_SIZE=BLOCK_SIZE
    )

    # 5. 执行 Stage 2
    # 输入: partial_vals -> 输出: max_val_out, max_idx_out
    # 直接将用户提供的输出 Tensor 传给 Kernel，让 Kernel 直接写入结果
    # 刚才优化的 kernel_stage2 会处理 num_blocks 个输入
    argmax_kernel.kernel_stage2(
        p_vals_wr, 
        p_idx_wr, 
        val_out_wr, 
        idx_out_wr, 
        num_blocks
    )

    # 返回结果 (虽然已经写入 tensor 了，为了保持接口一致性返回一下)
    return max_idx_out, max_val_out


def llaisysEmbedding(out, index, weight):
    """
    [原创无 Torch 版] Launcher for Triton-backed embedding.
    """
    # 1. 将 llaisys.Tensor 包装成 Triton 兼容的对象
    index_wrapped = LLAITensorAdapter(index)
    weight_wrapped = LLAITensorAdapter(weight)
    out_wrapped = LLAITensorAdapter(out)

    # 2. 从 wrapper 中获取维度信息
    # N 是 index 的元素数量
    N = index_wrapped.numel() 
    # D 是 embedding 的维度
    D = weight_wrapped.shape[1]

    # 3. 直接将 Wrapper 对象传递给 Triton 内核的启动器
    embedding_kernel.kernel(
        index_wrapped, 
        weight_wrapped, 
        out_wrapped, 
        N, 
        D, 
        BLOCK_SIZE=1024
    )

    return out


def llaisysLinear(out, inp, weight, bias):
    """
    [正确性优先版] Triton Linear Launcher
    - 不使用 Autotune
    - 使用固定的 Block Size (64, 64, 32)
    - 使用 tl.dot 保证数学运算正确且不溢出
    - 直接写入 out，不进行中间拷贝
    """
    # 1. 包装输入
    x_wr = LLAITensorAdapter(inp)
    w_wr = LLAITensorAdapter(weight)
    out_wr = LLAITensorAdapter(out)

    # 2. 形状检查
    if len(x_wr.shape) != 2 or len(w_wr.shape) != 2:
        raise RuntimeError(f"Triton linear expects 2D tensors")

    M, K = x_wr.shape
    N = w_wr.shape[0]

    # 3. 处理 Bias
    b_wr = None
    if bias is not None:
        b_wr = LLAITensorAdapter(bias)

    # 4. 获取 Strides (关键：处理非连续内存)
    stride_xm, stride_xk = x_wr.strides
    stride_wn, stride_wk = w_wr.strides
    stride_om, stride_on = out_wr.strides

    # 5. 确定输出精度
    # 简单的 heuristic: 如果权重是 FP32，输出保持 FP32。否则依赖 store 自动降级。
    w_dtype = w_wr._read_dtype_ll()
    OUT_FP32 = (w_dtype == DataType.F32)

    # 6. 设置固定的 Block Size
    # 这些值比较保守且通用，适合验证正确性
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # 7. 计算 Grid
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m, grid_n)

    # 8. 启动 Kernel
    linear_kernel._kernel[grid](
        x_wr,           # X 指针
        w_wr,           # W 指针
        b_wr,           # Bias 指针
        out_wr,         # Out 指针
        M, N, K,        # 维度
        stride_xm, stride_xk, # X Strides
        stride_wn, stride_wk, # W Strides
        stride_om, stride_on, # Out Strides
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        OUT_FP32=OUT_FP32
    )

    return out

def llaisysRmsNorm(out, inp, weight, eps: float):
    """
    [原创无 Torch 版] Launcher for Triton-backed RMSNorm.
    """
    # 1. 将 llaisys.Tensor 包装成 Triton 兼容的对象
    inp_wrapped = LLAITensorAdapter(inp)
    weight_wrapped = LLAITensorAdapter(weight)
    out_wrapped = LLAITensorAdapter(out)

    # 2. 从 wrapper 中获取维度信息
    M, D = inp_wrapped.shape

    # 3. 处理 eps (它可能是一个 ctypes 对象)
    try:
        if hasattr(eps, "value"):
            eps_val = float(eps.value)
        else:
            eps_val = float(eps)
    except Exception:
        eps_val = float(eps)
        
    # 4. 直接将 Wrapper 对象传递给 Triton 内核的启动器
    rms_norm_kernel.kernel(
        inp_wrapped, 
        weight_wrapped, 
        out_wrapped, 
        M, 
        D, 
        eps_val, 
        BLOCK_SIZE=1024
    )

    return out


def llaisysSelfAttention(attn_val_out, q, k, v, scale: float, past_k=None, past_v=None):
    """
    [FINAL DEBUG VERSION] Implements the S=1 fast path for debugging.
    If S=1, it skips the combine kernel and directly copies the result from the split kernel.
    """
    q_wr = LLAITensorAdapter(q)
    k_wr = LLAITensorAdapter(k)
    v_wr = LLAITensorAdapter(v)
    out_wr = LLAITensorAdapter(attn_val_out)

    seq_len_q, num_heads, emb_dim = q_wr.shape
    _, num_heads_k, _ = k_wr.shape

    batch_size = 1
    try:
        device_id = int(q_wr.device.split(":")[-1])
    except Exception:
        device_id = 0

    scale_val = _to_float(scale)
    IS_CAUSAL = True

    try:
        if past_k is None and past_v is None:
            # --- Prefill Path (unchanged) ---
            seq_len_k_v = k_wr.shape[0]
            num_groups = num_heads // num_heads_k
            s_q, s_k, s_v = q_wr.strides, k_wr.strides, v_wr.strides
            s_o_3d = out_wr.strides
            strides_q_b = (0, s_q[1], s_q[0], s_q[2])
            strides_k_b = (0, s_k[1], s_k[0], s_k[2])
            strides_v_b = (0, s_v[1], s_v[0], s_v[2])
            strides_o_b = (0, s_o_3d[1], s_o_3d[0], s_o_3d[2])

            if emb_dim < 16:
                BLOCK_SIZE_M = 16
                grid = (triton.cdiv(seq_len_q, BLOCK_SIZE_M), num_heads, batch_size)
                self_attention_kernel.kernel_small_hd[grid](
                    q_wr, k_wr, v_wr, out_wr,
                    *strides_q_b, *strides_k_b, *strides_v_b, *strides_o_b,
                    scale_val, seq_len_q, seq_len_k_v,
                    EMB_DIM=emb_dim, NUM_GROUPS=num_groups
                )
            else:
                BLOCK_SIZE_M = 64
                BLOCK_SIZE_N = 64
                grid = (triton.cdiv(seq_len_q, BLOCK_SIZE_M), num_heads, batch_size)
                self_attention_kernel.kernel[grid](
                    q_wr, k_wr, v_wr, out_wr,
                    *strides_q_b, *strides_k_b, *strides_v_b, *strides_o_b,
                    scale_val, seq_len_q, seq_len_k_v,
                    EMB_DIM=emb_dim, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, IS_CAUSAL=IS_CAUSAL, NUM_GROUPS=num_groups
                )
        else:
            # --- Decode Path ---
            # wrap past tensors
            past_k_wr = LLAITensorAdapter(past_k)
            past_v_wr = LLAITensorAdapter(past_v)

            # shapes
            seq_len_q = q_wr.shape[0]
            current_seq_len = k_wr.shape[0]
            past_seq_len = past_k_wr.shape[0]
            seq_len_k_v = past_seq_len + current_seq_len

            num_groups = num_heads // num_heads_k

            # choose S (debug version returns 1)
            S = get_optimal_s(batch_size, num_heads, seq_len_k_v, q_wr.device)

            # Fast, robust S==1 path: concatenate past_k/past_v on device and reuse prefill kernels.
            # This avoids split/combine complexity while preserving past KV semantics.
            if S == 1:
                # create concatenated k_cat/v_cat with shape (seq_len_k_v, num_heads_k, emb_dim)
                k_cat = Tensor(shape=(seq_len_k_v, num_heads_k, emb_dim), dtype=DataType(LIB_LLAISYS.tensorGetDataType(_get_raw_ptr(k))), device=DeviceType.NVIDIA, device_id=device_id)
                v_cat = Tensor(shape=(seq_len_k_v, num_heads_k, emb_dim), dtype=DataType(LIB_LLAISYS.tensorGetDataType(_get_raw_ptr(v))), device=DeviceType.NVIDIA, device_id=device_id)

                runtime = RuntimeAPI(DeviceType.NVIDIA)
                # copy past_k -> k_cat[0:past_seq_len]
                src_past_k = ctypes.c_void_p(LIB_LLAISYS.tensorGetData(_get_raw_ptr(past_k)))
                dst_k_cat = ctypes.c_void_p(LIB_LLAISYS.tensorGetData(k_cat.lib_tensor()))
                elem_size_k = get_element_size(DataType(LIB_LLAISYS.tensorGetDataType(_get_raw_ptr(past_k))))
                runtime.memcpy_sync(ctypes.c_void_p(dst_k_cat.value), src_past_k, past_seq_len * int(num_heads_k) * int(emb_dim) * elem_size_k, MemcpyKind.D2D)
                # copy current k -> k_cat[past_seq_len:]
                src_k = ctypes.c_void_p(LIB_LLAISYS.tensorGetData(_get_raw_ptr(k)))
                # compute destination pointer offset: rows * row_bytes
                row_bytes = int(num_heads_k) * int(emb_dim) * elem_size_k
                dst_k_offset = ctypes.c_void_p(dst_k_cat.value + past_seq_len * row_bytes)
                runtime.memcpy_sync(dst_k_offset, src_k, current_seq_len * row_bytes, MemcpyKind.D2D)

                # same for v
                src_past_v = ctypes.c_void_p(LIB_LLAISYS.tensorGetData(_get_raw_ptr(past_v)))
                dst_v_cat = ctypes.c_void_p(LIB_LLAISYS.tensorGetData(v_cat.lib_tensor()))
                elem_size_v = get_element_size(DataType(LIB_LLAISYS.tensorGetDataType(_get_raw_ptr(past_v))))
                runtime.memcpy_sync(ctypes.c_void_p(dst_v_cat.value), src_past_v, past_seq_len * int(num_heads_k) * int(emb_dim) * elem_size_v, MemcpyKind.D2D)
                src_v = ctypes.c_void_p(LIB_LLAISYS.tensorGetData(_get_raw_ptr(v)))
                row_bytes_v = int(num_heads_k) * int(emb_dim) * elem_size_v
                dst_v_offset = ctypes.c_void_p(dst_v_cat.value + past_seq_len * row_bytes_v)
                runtime.memcpy_sync(dst_v_offset, src_v, current_seq_len * row_bytes_v, MemcpyKind.D2D)

                # wrap adapters and call prefill kernel (reuse existing prefill path)
                k_cat_wr = LLAITensorAdapter(k_cat.lib_tensor())
                v_cat_wr = LLAITensorAdapter(v_cat.lib_tensor())

                num_groups = num_heads // num_heads_k
                s_q, s_k_cat, s_v_cat = q_wr.strides, k_cat_wr.strides, v_cat_wr.strides
                s_o_3d = out_wr.strides
                strides_q_b = (0, s_q[1], s_q[0], s_q[2])
                strides_k_b = (0, s_k_cat[1], s_k_cat[0], s_k_cat[2])
                strides_v_b = (0, s_v_cat[1], s_v_cat[0], s_v_cat[2])
                strides_o_b = (0, s_o_3d[1], s_o_3d[0], s_o_3d[2])

                if emb_dim < 16:
                    BLOCK_SIZE_M = 16
                    grid = (triton.cdiv(seq_len_q, BLOCK_SIZE_M), num_heads, batch_size)
                    self_attention_kernel.kernel_small_hd[grid](
                        q_wr, k_cat_wr, v_cat_wr, out_wr,
                        *strides_q_b, *strides_k_b, *strides_v_b, *strides_o_b,
                        scale_val, seq_len_q, seq_len_k_v,
                        EMB_DIM=emb_dim, NUM_GROUPS=num_groups,
                    )
                else:
                    BLOCK_SIZE_M = 64
                    BLOCK_SIZE_N = 64
                    grid = (triton.cdiv(seq_len_q, BLOCK_SIZE_M), num_heads, batch_size)
                    self_attention_kernel.kernel[grid](
                        q_wr, k_cat_wr, v_cat_wr, out_wr,
                        *strides_q_b, *strides_k_b, *strides_v_b, *strides_o_b,
                        scale_val, seq_len_q, seq_len_k_v,
                        EMB_DIM=emb_dim, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, IS_CAUSAL=IS_CAUSAL, NUM_GROUPS=num_groups,
                    )

                # done - we've written into attn_val_out
                return attn_val_out

            # else S > 1: fall back to split/combine path (not optimized here)
            # create temporary device buffers
            # split_logsumexp: (batch_size, num_heads, S, seq_len_q) float32
            split_logsumexp = Tensor(shape=(batch_size, num_heads, S, seq_len_q), dtype=DataType.F32, device=DeviceType.NVIDIA, device_id=device_id)
            # split_outputs: (batch_size, num_heads, S, seq_len_q, emb_dim) float16
            split_outputs = Tensor(shape=(batch_size, num_heads, S, seq_len_q, emb_dim), dtype=DataType.F16, device=DeviceType.NVIDIA, device_id=device_id)

            # final outputs: create final_o with hw2-style layout (batch_size, num_heads, seq_len_q, emb_dim)
            # dtype should match user-provided attn_val_out dtype so combine can cast correctly
            out_dt = DataType(LIB_LLAISYS.tensorGetDataType(_get_raw_ptr(attn_val_out)))
            final_o = Tensor(shape=(batch_size, num_heads, seq_len_q, emb_dim), dtype=out_dt, device=DeviceType.NVIDIA, device_id=device_id)
            # final_l as (batch_size * num_heads, seq_len_q) float32 to match combine ptr math
            final_l = Tensor(shape=(batch_size * num_heads, seq_len_q), dtype=DataType.F32, device=DeviceType.NVIDIA, device_id=device_id)

            # prepare strides mapping consistent with kernels
            s_q = q_wr.strides
            s_k = k_wr.strides
            s_v = v_wr.strides
            s_past_k = past_k_wr.strides
            s_past_v = past_v_wr.strides

            # map 3D LLAISYS (seq_len, heads, emb) -> kernel expected 4-stride args (z,h,m,k)
            strides_q_b = (0, s_q[1], s_q[0], s_q[2])
            strides_k_b = (0, s_k[1], s_k[0], s_k[2])
            strides_v_b = (0, s_v[1], s_v[0], s_v[2])
            strides_past_k_b = (0, s_past_k[1], s_past_k[0], s_past_k[2])
            strides_past_v_b = (0, s_past_v[1], s_past_v[0], s_past_v[2])

            # strides for split_outputs (5D)
            split_out_wr = LLAITensorAdapter(split_outputs.lib_tensor())
            so_strides = split_out_wr.strides

            # strides for final_o (we created as 3D to match attn_val_out)
            final_o_wr = LLAITensorAdapter(final_o.lib_tensor())
            fo_strides = final_o_wr.strides

            # compute binned sizes
            M_binned = triton.next_power_of_2(seq_len_q)
            N_binned = triton.next_power_of_2(seq_len_k_v)

            # choose block sizes
            BLOCK_SIZE_M = 64 if emb_dim >= 16 else 16
            BLOCK_SIZE_N = 64

            # call split kernel
            def grid1(meta):
                return (triton.cdiv(seq_len_q, meta['BLOCK_SIZE_M']), S, num_heads * batch_size)

            decode_kernels.split_kv_kernel[grid1](
                q_wr, k_wr, v_wr, past_k_wr, past_v_wr,
                scale_val,
                LLAITensorAdapter(split_logsumexp.lib_tensor()), LLAITensorAdapter(split_outputs.lib_tensor()),
                num_heads, num_groups,
                strides_q_b[0], strides_q_b[1], strides_q_b[2], strides_q_b[3],
                strides_k_b[0], strides_k_b[1], strides_k_b[2], strides_k_b[3],
                strides_v_b[0], strides_v_b[1], strides_v_b[2], strides_v_b[3],
                strides_past_k_b[0], strides_past_k_b[1], strides_past_k_b[2], strides_past_k_b[3],
                strides_past_v_b[0], strides_past_v_b[1], strides_past_v_b[2], strides_past_v_b[3],
                so_strides[0], so_strides[1], so_strides[2], so_strides[3], so_strides[4],
                seq_len_q,
                seq_len_k_v,
                current_seq_len,
                S,
                EMB_DIM=emb_dim, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, M_BINNED=M_binned, N_BINNED=N_binned, IS_CAUSAL=IS_CAUSAL,
            )

            # call combine kernel (it correctly handles S==1). Wrap tensors in adapters.
            def grid2(meta):
                return (triton.cdiv(seq_len_q, meta['BLOCK_SIZE_M']), num_heads, batch_size)

            sl_strides = split_out_wr.strides
            final_o_wr = LLAITensorAdapter(final_o.lib_tensor())
            fo_strides = final_o_wr.strides

            # call combine with LLAITensorAdapter wrappers for device tensors
            decode_kernels.combine_kv_splits_kernel[grid2](
                LLAITensorAdapter(split_outputs.lib_tensor()), LLAITensorAdapter(split_logsumexp.lib_tensor()),
                final_o_wr, LLAITensorAdapter(final_l.lib_tensor()),
                num_heads,
                sl_strides[0], sl_strides[1], sl_strides[2], sl_strides[3], sl_strides[4],
                fo_strides[0], fo_strides[1], fo_strides[2], fo_strides[3],
                seq_len_q, S,
                EMB_DIM=emb_dim, BLOCK_SIZE_M=BLOCK_SIZE_M, M_BINNED=M_binned,
            )

            # final_o now has shape (batch_size, num_heads, seq_len_q, emb_dim)
            # permute -> (batch_size, seq_len_q, num_heads, emb_dim) then collapse batch dim and copy to attn_val_out
            permuted = final_o.permute(0, 2, 1, 3)  # (B, SEQ, H, EMB)
            collapsed = permuted.view(batch_size * seq_len_q, num_heads, emb_dim)
            if batch_size == 1:
                final_view = collapsed.view(seq_len_q, num_heads, emb_dim)
            else:
                # general case: create a temporary contiguous tensor matching attn_val_out layout then copy
                final_view = collapsed.view(batch_size * seq_len_q, num_heads, emb_dim)

            # copy final_view -> attn_val_out (D2D)
            runtime = RuntimeAPI(DeviceType.NVIDIA)
            out_ptr = _get_raw_ptr(attn_val_out)
            dst_data = LIB_LLAISYS.tensorGetData(out_ptr)
            src_data = LIB_LLAISYS.tensorGetData(final_view.lib_tensor())
            elem_size = get_element_size(out_dt)
            size_bytes = int(seq_len_q) * int(num_heads) * int(emb_dim) * int(elem_size)
            runtime.memcpy_sync(ctypes.c_void_p(dst_data), ctypes.c_void_p(src_data), size_bytes, MemcpyKind.D2D)

    except Exception as e:
        import traceback
        print(f"[llaisys] Triton kernels raised exception: {e}")
        traceback.print_exc()
        
    return attn_val_out

def llaisysSwiGLU(out, gate, up):
    """
    [原创无 Torch 版] Launcher for Triton-backed SwiGLU.
    """
    # 1. 将 llaisys.Tensor 包装成 Triton 兼容的对象
    gate_wrapped = LLAITensorAdapter(gate)
    up_wrapped = LLAITensorAdapter(up)
    out_wrapped = LLAITensorAdapter(out)

    # 2. 直接将 Wrapper 对象传递给 Triton 内核的启动器
    swiglu_kernel.kernel(
        gate_wrapped, 
        up_wrapped, 
        out_wrapped, 
        BLOCK_SIZE=1024
    )

    return out


def llaisysROPE(out, inp, pos_ids, theta: float):
    """Torch-free launcher for RoPE.

    Minimal change: compute `pos` and `freqs` on host, upload as device tensors,
    and call the Triton kernel using `LLAITensorAdapter` wrappers.
    """
    # Wrap inputs/outputs
    inp_wr = LLAITensorAdapter(inp)
    out_wr = LLAITensorAdapter(out)
    pos_wr_raw = LLAITensorAdapter(pos_ids)

    # validate shapes
    seq_len, n_heads, head_dim = inp_wr.shape
    assert head_dim % 2 == 0
    half = head_dim // 2

    # coerce theta
    try:
        if hasattr(theta, "value"):
            theta_val = float(theta.value)
        else:
            theta_val = float(theta)
    except Exception:
        theta_val = float(theta)

    runtime = RuntimeAPI(DeviceType.NVIDIA)

    # --- read pos_ids to host (use float64 for high precision) ---
    pos_ptr = _get_raw_ptr(pos_ids)
    p_ndim = int(LIB_LLAISYS.tensorGetNdim(pos_ptr))
    p_shape_buf = (ctypes.c_size_t * p_ndim)()
    LIB_LLAISYS.tensorGetShape(pos_ptr, p_shape_buf)
    pos_numel = 1
    for i_ in range(p_ndim):
        pos_numel *= int(p_shape_buf[i_])

    src_pos_dt = DataType(LIB_LLAISYS.tensorGetDataType(pos_ptr))
    src_elem = get_element_size(src_pos_dt)
    host_pos_buf = runtime.malloc_host(pos_numel * src_elem)
    runtime.memcpy_sync(host_pos_buf, ctypes.c_void_p(LIB_LLAISYS.tensorGetData(pos_ptr)), pos_numel * src_elem, MemcpyKind.D2H)
    host_pos_addr = ctypes.cast(host_pos_buf, ctypes.c_void_p).value
    if src_pos_dt == DataType.I64:
        pos_host = _np.ctypeslib.as_array((ctypes.c_int64 * pos_numel).from_address(host_pos_addr)).astype(_np.float64)
    else:
        pos_host = _np.ctypeslib.as_array((ctypes.c_int32 * pos_numel).from_address(host_pos_addr)).astype(_np.float64)

    # --- compute freqs matrix and sin/cos on host in float64 for higher precision ---
    indices = _np.arange(0, half, dtype=_np.float64)
    freqs_1d = 1.0 / (theta_val ** (2.0 * indices / head_dim))
    # broadcast multiply to get (seq_len, half)
    freqs_mat = (pos_host.reshape(-1, 1) * freqs_1d.reshape(1, -1)).astype(_np.float64)

    # compute sin/cos in float64 then cast to float32 for device
    sin_host = _np.sin(freqs_mat).astype(_np.float32)
    cos_host = _np.cos(freqs_mat).astype(_np.float32)

    # flatten and upload sin and cos as temporary device tensors of shape (seq_len*half,)
    flat_len = sin_host.size
    sin_flat = sin_host.ravel()
    cos_flat = cos_host.ravel()

    sin_ptr = Tensor(shape=(flat_len,), dtype=DataType.F32, device=DeviceType.NVIDIA)
    cos_ptr = Tensor(shape=(flat_len,), dtype=DataType.F32, device=DeviceType.NVIDIA)
    # load data via host staging
    host_buf_sin = runtime.malloc_host(sin_flat.nbytes)
    ctypes.memmove(host_buf_sin, sin_flat.ctypes.data, sin_flat.nbytes)
    LIB_LLAISYS.tensorLoad(sin_ptr.lib_tensor(), host_buf_sin)
    runtime.free_host(host_buf_sin)

    host_buf_cos = runtime.malloc_host(cos_flat.nbytes)
    ctypes.memmove(host_buf_cos, cos_flat.ctypes.data, cos_flat.nbytes)
    LIB_LLAISYS.tensorLoad(cos_ptr.lib_tensor(), host_buf_cos)
    runtime.free_host(host_buf_cos)

    sin_wr = LLAITensorAdapter(sin_ptr.lib_tensor())
    cos_wr = LLAITensorAdapter(cos_ptr.lib_tensor())

    # 4. call Triton kernel with precomputed sin/cos flattened arrays
    rope_kernel.kernel(
        inp_wr,
        out_wr,
        sin_wr,
        cos_wr,
        BLOCK=128,
    )

    # destroy temporary host pos buffer
    runtime.free_host(host_pos_buf)

    return out

