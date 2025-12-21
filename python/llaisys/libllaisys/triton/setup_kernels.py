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


_rope_cache = {}

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

def create_kv_cache(total_seq_len: int, kv_heads: int, head_dim: int, dtype: DataType, device_type: DeviceType, device_id: int = 0) -> Tensor:
    """Create an empty KV cache tensor on the specified device.

    Returns a Tensor of shape (total_seq_len, kv_heads, head_dim).
    """
    shape = (total_seq_len, kv_heads, head_dim)
    return Tensor(shape=shape, dtype=dtype, device=device_type, device_id=device_id)


def kv_cache_write_slice(cache: Tensor, src: Tensor, dst_offset: int):
    """Write `src` into `cache` at sequence offset `dst_offset` along dim 0.

    Performs a device-to-device memcpy. `src` shape is (S, kv_heads, head_dim).
    """
    cache_wr = LLAITensorAdapter(cache)
    src_wr = LLAITensorAdapter(src)

    # infer device and runtime from cache
    try:
        in_ptr = _get_raw_ptr(cache)
        dev_type = DeviceType(LIB_LLAISYS.tensorGetDeviceType(in_ptr))
    except Exception:
        dev_type = DeviceType.NVIDIA

    runtime = RuntimeAPI(dev_type)

    # compute bytes
    elem_size = cache_wr.element_size()
    # row_elems = number of elements per sequence step (kv_heads * head_dim)
    seq_dim = src_wr.shape[0]
    if isinstance(seq_dim, tuple) or seq_dim is None:
        seq_dim = int(src_wr.shape[0])
    row_elems = 1
    for s in src_wr.shape[1:]:
        row_elems *= int(s)

    src_bytes = src_wr.numel() * elem_size
    offset_bytes = dst_offset * row_elems * elem_size

    dst_ptr = ctypes.c_void_p(cache_wr.data_ptr())
    src_ptr = ctypes.c_void_p(src_wr.data_ptr())

    dst_offset_ptr = ctypes.c_void_p(dst_ptr.value + offset_bytes)

    runtime.memcpy_sync(dst_offset_ptr, src_ptr, src_bytes, MemcpyKind.D2D)

    return
    
    
def llaisysAdd(out, a, b):
    """
    Launcher that bridges LLAISYS tensors to the Triton add kernel.
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
    Argmax Launcher
    """
    # 1. 包装输入输出
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
    # 需要保存 Stage 1 产生的 num_blocks 个局部最大值和索引
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
    # 刚才优化的 kernel_stage2 会处理 num_blocks 个输入
    argmax_kernel.kernel_stage2(
        p_vals_wr, 
        p_idx_wr, 
        val_out_wr, 
        idx_out_wr, 
        num_blocks
    )

    # 虽然已经写入 tensor 了，为了保持接口一致性返回一下
    return max_idx_out, max_val_out


def llaisysEmbedding(out, index, weight):
    """
    Launcher for Triton-backed embedding.
    """
    # 将 llaisys.Tensor 包装
    index_wrapped = LLAITensorAdapter(index)
    weight_wrapped = LLAITensorAdapter(weight)
    out_wrapped = LLAITensorAdapter(out)

    # N 是 index 的元素数量
    N = index_wrapped.numel() 
    # D 是 embedding 的维度
    D = weight_wrapped.shape[1]

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
    Triton Linear Launcher
    """
    # 1. 包装输入
    x_wr = LLAITensorAdapter(inp)
    w_wr = LLAITensorAdapter(weight)
    out_wr = LLAITensorAdapter(out)

    # 2. 形状检查
    if len(x_wr.shape) != 2 or len(w_wr.shape) != 2:
        raise RuntimeError(f"Triton linear expects 2D inputs")

    M, K = x_wr.shape
    N = w_wr.shape[0]

    # 3. 处理 Bias
    b_wr = None
    if bias is not None:
        b_wr = LLAITensorAdapter(bias)

    # 4. 获取 Strides
    stride_xm, stride_xk = x_wr.strides
    stride_wn, stride_wk = w_wr.strides
    
    s_out = out_wr.strides
    if len(s_out) == 2:
        stride_om, stride_on = s_out
    elif len(s_out) > 2:
        # 如果是 3D 或更高维，假设内部维度是连续的，取第一个和最后一个 stride
        # 对应于 Flatten 后的 (M, N)
        stride_om = s_out[0]
        stride_on = s_out[-1]
    else:
        raise RuntimeError(f"Linear output must be at least 2D, got {len(s_out)}")

    # 5. 确定输出精度
    w_dtype = w_wr._read_dtype_ll()
    OUT_FP32 = (w_dtype == DataType.F32)

    # 6. 设置固定的 Block Size
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
    Launcher for Triton-backed RMSNorm.
    """
    # 将 llaisys.Tensor 包装
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
            # print(f"[llaisysSelfAttention] PREFILL path: seq_len_q={seq_len_q} seq_len_k_v={k_wr.shape[0]} num_heads={num_heads} emb_dim={emb_dim}", flush=True)
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
            past_k_wr = LLAITensorAdapter(past_k)
            past_v_wr = LLAITensorAdapter(past_v)

            seq_len_q = q_wr.shape[0]
            current_seq_len = k_wr.shape[0]
            past_seq_len = past_k_wr.shape[0]
            seq_len_k_v = past_seq_len + current_seq_len
            num_groups = num_heads // num_heads_k

            # choose S
            S = get_optimal_s(batch_size, num_heads, seq_len_k_v, q_wr.device)
            # print("choose S =", S)

            # Common Setup
            s_q = q_wr.strides
            s_k = k_wr.strides
            s_v = v_wr.strides
            s_past_k = past_k_wr.strides
            s_past_v = past_v_wr.strides

            strides_q_b = (0, s_q[1], s_q[0], s_q[2])
            strides_k_b = (0, s_k[1], s_k[0], s_k[2])
            strides_v_b = (0, s_v[1], s_v[0], s_v[2])
            strides_past_k_b = (0, s_past_k[1], s_past_k[0], s_past_k[2])
            strides_past_v_b = (0, s_past_v[1], s_past_v[0], s_past_v[2])
            
            M_binned = triton.next_power_of_2(seq_len_q)
            N_binned = triton.next_power_of_2(seq_len_k_v)
            BLOCK_SIZE_M = 64 if emb_dim >= 16 else 16
            BLOCK_SIZE_N = 64

            # --- S=1 Optimization: Zero-Copy Fast Path ---
            if S == 1:
                # 但我们可以直接写入 attn_val_out，无需 split_outputs 和 combine 过程
                split_logsumexp = Tensor(shape=(batch_size, num_heads, 1, seq_len_q), dtype=DataType.F32, device=DeviceType.NVIDIA, device_id=device_id)
                
                s_out = out_wr.strides
                if len(s_out) == 3: # [Seq, Head, Dim]
                    o_stride_z = 0
                    o_stride_m = s_out[0]
                    o_stride_h = s_out[1]
                    o_stride_k = s_out[2]
                elif len(s_out) == 4: # [Batch, Seq, Head, Dim]
                    o_stride_z = s_out[0]
                    o_stride_m = s_out[1]
                    o_stride_h = s_out[2]
                    o_stride_k = s_out[3]
                else:
                    # Fallback: assume contiguous [Seq, Head, Dim]
                    o_stride_z = 0
                    o_stride_m = num_heads * emb_dim
                    o_stride_h = emb_dim
                    o_stride_k = 1
                
                o_stride_s = 0 # S=1，Split 维度 stride 设为 0 即可

                def grid_fast(meta):
                    return (triton.cdiv(seq_len_q, meta['BLOCK_SIZE_M']), 1, num_heads * batch_size)

                # 直接把 attn_val_out 作为输出传进去
                self_attention_kernel.split_kv_kernel[grid_fast](
                    q_wr, k_wr, v_wr, past_k_wr, past_v_wr,
                    scale_val,
                    LLAITensorAdapter(split_logsumexp.lib_tensor()), out_wr, # <--- 直接写 Out
                    num_heads, num_groups,
                    strides_q_b[0], strides_q_b[1], strides_q_b[2], strides_q_b[3],
                    strides_k_b[0], strides_k_b[1], strides_k_b[2], strides_k_b[3],
                    strides_v_b[0], strides_v_b[1], strides_v_b[2], strides_v_b[3],
                    strides_past_k_b[0], strides_past_k_b[1], strides_past_k_b[2], strides_past_k_b[3],
                    strides_past_v_b[0], strides_past_v_b[1], strides_past_v_b[2], strides_past_v_b[3],
                    o_stride_z, o_stride_h, o_stride_s, o_stride_m, o_stride_k, # <--- 适配后的 Strides
                    seq_len_q,
                    seq_len_k_v,
                    current_seq_len,
                    1, # S=1
                    EMB_DIM=emb_dim, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, M_BINNED=M_binned, N_BINNED=N_binned, IS_CAUSAL=IS_CAUSAL,
                )
                # S=1 时，split kernel 的输出已经是归一化后的结果，直接返回即可
                return attn_val_out

            # --- S > 1 Path: Standard Split & Combine ---
            else:
                # 1. Allocate Temp Buffers
                split_logsumexp = Tensor(shape=(batch_size, num_heads, S, seq_len_q), dtype=DataType.F32, device=DeviceType.NVIDIA, device_id=device_id)
                split_outputs = Tensor(shape=(batch_size, num_heads, S, seq_len_q, emb_dim), dtype=DataType.F16, device=DeviceType.NVIDIA, device_id=device_id)
                
                # Final output temp buffers (needs HW2 layout for combine kernel)
                out_dt = DataType(LIB_LLAISYS.tensorGetDataType(_get_raw_ptr(attn_val_out)))
                final_o = Tensor(shape=(batch_size, num_heads, seq_len_q, emb_dim), dtype=out_dt, device=DeviceType.NVIDIA, device_id=device_id)
                final_l = Tensor(shape=(batch_size * num_heads, seq_len_q), dtype=DataType.F32, device=DeviceType.NVIDIA, device_id=device_id)

                split_out_wr = LLAITensorAdapter(split_outputs.lib_tensor())
                so_strides = split_out_wr.strides

                # 2. Run Split Kernel
                def grid1(meta):
                    return (triton.cdiv(seq_len_q, meta['BLOCK_SIZE_M']), S, num_heads * batch_size)

                self_attention_kernel.split_kv_kernel[grid1](
                    q_wr, k_wr, v_wr, past_k_wr, past_v_wr,
                    scale_val,
                    LLAITensorAdapter(split_logsumexp.lib_tensor()), split_out_wr,
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

                # 3. Run Combine Kernel
                final_o_wr = LLAITensorAdapter(final_o.lib_tensor())
                fo_strides = final_o_wr.strides
                sl_strides = split_out_wr.strides

                def grid2(meta):
                    return (triton.cdiv(seq_len_q, meta['BLOCK_SIZE_M']), num_heads, batch_size)

                self_attention_kernel.combine_kv_splits_kernel[grid2](
                    LLAITensorAdapter(split_outputs.lib_tensor()), LLAITensorAdapter(split_logsumexp.lib_tensor()),
                    final_o_wr, LLAITensorAdapter(final_l.lib_tensor()),
                    num_heads,
                    sl_strides[0], sl_strides[1], sl_strides[2], sl_strides[3], sl_strides[4],
                    fo_strides[0], fo_strides[1], fo_strides[2], fo_strides[3],
                    seq_len_q, S,
                    EMB_DIM=emb_dim, BLOCK_SIZE_M=BLOCK_SIZE_M, M_BINNED=M_binned,
                )

                # 4. Final Copy / Permute (HW2 -> Output Layout)
                permuted = final_o.permute(0, 2, 1, 3)
                collapsed = permuted.view(batch_size * seq_len_q, num_heads, emb_dim)
                final_view = collapsed.view(batch_size * seq_len_q, num_heads, emb_dim) # redundant view but safe

                runtime = RuntimeAPI(DeviceType.NVIDIA)
                out_ptr = _get_raw_ptr(attn_val_out)
                dst_data = LIB_LLAISYS.tensorGetData(out_ptr)
                src_data = LIB_LLAISYS.tensorGetData(final_view.lib_tensor())
                elem_size = get_element_size(out_dt)
                size_bytes = int(seq_len_q) * int(num_heads) * int(emb_dim) * int(elem_size)
                runtime.memcpy_sync(ctypes.c_void_p(dst_data), ctypes.c_void_p(src_data), size_bytes, MemcpyKind.D2D)
                
    except Exception as e:
        import traceback
        # print(f"[llaisys] Triton kernels raised exception: {e}")
        traceback.print_exc()
        
    return attn_val_out

def llaisysSwiGLU(out, gate, up):
    # 1. 将 llaisys.Tensor 包装
    gate_wrapped = LLAITensorAdapter(gate)
    up_wrapped = LLAITensorAdapter(up)
    out_wrapped = LLAITensorAdapter(out)

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

    # infer device from input and obtain runtime for that device
    in_ptr = _get_raw_ptr(inp)
    try:
        device_type = DeviceType(LIB_LLAISYS.tensorGetDeviceType(in_ptr))
    except Exception:
        device_type = DeviceType.NVIDIA
    try:
        device_id = int(LIB_LLAISYS.tensorGetDeviceId(in_ptr))
    except Exception:
        device_id = 0

    runtime = RuntimeAPI(device_type)

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

    # reuse or allocate host staging buffer for pos_ids
    host_pos_bytes = pos_numel * src_elem
    cache_key = (int(device_type), device_id, pos_numel)
    cached = _rope_cache.get(cache_key)
    if cached and cached.get('host_pos_bytes') == host_pos_bytes and cached.get('host_pos_buf') is not None:
        host_pos_buf = cached['host_pos_buf']
    else:
        host_pos_buf = runtime.malloc_host(host_pos_bytes)
        if not cached:
            _rope_cache[cache_key] = {}
        _rope_cache[cache_key]['host_pos_buf'] = host_pos_buf
        _rope_cache[cache_key]['host_pos_bytes'] = host_pos_bytes

    runtime.memcpy_sync(host_pos_buf, ctypes.c_void_p(LIB_LLAISYS.tensorGetData(pos_ptr)), host_pos_bytes, MemcpyKind.D2H)
    host_pos_addr = ctypes.cast(host_pos_buf, ctypes.c_void_p).value
    if src_pos_dt == DataType.I64:
        pos_host_raw = _np.ctypeslib.as_array((ctypes.c_int64 * pos_numel).from_address(host_pos_addr))
    else:
        pos_host_raw = _np.ctypeslib.as_array((ctypes.c_int32 * pos_numel).from_address(host_pos_addr))
    pos_host = pos_host_raw.astype(_np.float64)

    # detect common case where pos_ids == arange(seq_len)
    try:
        pos_int = pos_host_raw.astype(_np.int64)
        is_arange = _np.array_equal(pos_int, _np.arange(seq_len, dtype=_np.int64))
    except Exception:
        is_arange = False

    # --- compute freqs matrix and sin/cos on host in float64 for higher precision ---
    indices = _np.arange(0, half, dtype=_np.float64)
    freqs_1d = 1.0 / (theta_val ** (2.0 * indices / head_dim))
    # broadcast multiply to get (seq_len, half)
    freqs_mat = (pos_host.reshape(-1, 1) * freqs_1d.reshape(1, -1)).astype(_np.float64)

    # compute sin/cos in float64 then cast to float32 for device
    sin_host = _np.sin(freqs_mat).astype(_np.float32)
    cos_host = _np.cos(freqs_mat).astype(_np.float32)

    # flatten and upload sin and cos as device tensors; reuse if cached
    flat_len = sin_host.size
    sin_flat = sin_host.ravel()
    cos_flat = cos_host.ravel()

    if is_arange:
        sin_cos_key = (int(device_type), device_id, seq_len, half)
        cached2 = _rope_cache.get(sin_cos_key)
        if cached2 and cached2.get('sin') is not None and cached2.get('cos') is not None:
            sin_ptr = cached2['sin']
            cos_ptr = cached2['cos']
        else:
            sin_ptr = Tensor(shape=(flat_len,), dtype=DataType.F32, device=device_type, device_id=device_id)
            cos_ptr = Tensor(shape=(flat_len,), dtype=DataType.F32, device=device_type, device_id=device_id)

            # use (or create) a host staging buffer for sin/cos loads
            host_stage_bytes = max(sin_flat.nbytes, cos_flat.nbytes)
            if cached2 is None:
                cached2 = {}
                _rope_cache[sin_cos_key] = cached2
            if cached2.get('host_stage_bytes') == host_stage_bytes and cached2.get('host_stage_buf') is not None:
                host_stage_buf = cached2['host_stage_buf']
            else:
                # allocate host staging buffer once
                host_stage_buf = runtime.malloc_host(host_stage_bytes)
                cached2['host_stage_buf'] = host_stage_buf
                cached2['host_stage_bytes'] = host_stage_bytes

            # copy sin
            ctypes.memmove(host_stage_buf, sin_flat.ctypes.data, sin_flat.nbytes)
            LIB_LLAISYS.tensorLoad(sin_ptr.lib_tensor(), host_stage_buf)

            # copy cos
            ctypes.memmove(host_stage_buf, cos_flat.ctypes.data, cos_flat.nbytes)
            LIB_LLAISYS.tensorLoad(cos_ptr.lib_tensor(), host_stage_buf)

            cached2['sin'] = sin_ptr
            cached2['cos'] = cos_ptr
    else:
        # non-standard pos_ids; compute sin/cos and upload but don't cache values
        sin_ptr = Tensor(shape=(flat_len,), dtype=DataType.F32, device=device_type, device_id=device_id)
        cos_ptr = Tensor(shape=(flat_len,), dtype=DataType.F32, device=device_type, device_id=device_id)
        # try to reuse a per-device host staging buffer if present
        stage_key = ('rope_stage', int(device_type), device_id)
        stage_cached = _rope_cache.get(stage_key)
        host_stage_bytes = max(sin_flat.nbytes, cos_flat.nbytes)
        if stage_cached and stage_cached.get('host_stage_bytes') == host_stage_bytes and stage_cached.get('host_stage_buf') is not None:
            host_stage_buf = stage_cached['host_stage_buf']
        else:
            host_stage_buf = runtime.malloc_host(host_stage_bytes)
            if stage_cached is None:
                stage_cached = {}
                _rope_cache[stage_key] = stage_cached
            stage_cached['host_stage_buf'] = host_stage_buf
            stage_cached['host_stage_bytes'] = host_stage_bytes

        ctypes.memmove(host_stage_buf, sin_flat.ctypes.data, sin_flat.nbytes)
        LIB_LLAISYS.tensorLoad(sin_ptr.lib_tensor(), host_stage_buf)
        ctypes.memmove(host_stage_buf, cos_flat.ctypes.data, cos_flat.nbytes)
        LIB_LLAISYS.tensorLoad(cos_ptr.lib_tensor(), host_stage_buf)

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

    return out

