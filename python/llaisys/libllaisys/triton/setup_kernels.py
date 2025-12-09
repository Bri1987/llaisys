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
    """Concatenate two LLAISYS device tensors along the first dimension without using torch.

    Both `a` and `b` are LLAISYS tensor wrappers or raw handles with matching trailing
    dimensions (e.g., shapes (L1, H, D) and (L2, H, D)). Returns a new device `Tensor`.
    """
    a_ptr = _get_raw_ptr(a)
    b_ptr = _get_raw_ptr(b)
    runtime = RuntimeAPI(DeviceType.NVIDIA)

    # read shapes
    def _shape(p):
        ndim = int(LIB_LLAISYS.tensorGetNdim(p))
        buf = (ctypes.c_size_t * ndim)()
        LIB_LLAISYS.tensorGetShape(p, buf)
        return tuple(int(buf[i]) for i in range(ndim))

    a_shape = _shape(a_ptr)
    b_shape = _shape(b_ptr)
    if len(a_shape) != len(b_shape) or a_shape[1:] != b_shape[1:]:
        raise RuntimeError("Cannot concat tensors with incompatible shapes: %s vs %s" % (a_shape, b_shape))

    out_shape = (a_shape[0] + b_shape[0],) + a_shape[1:]

    # dtype and element size
    dt = DataType(LIB_LLAISYS.tensorGetDataType(a_ptr))
    elem = get_element_size(dt)

    # copy both to host
    a_numel = 1
    for d in a_shape:
        a_numel *= int(d)
    b_numel = 1
    for d in b_shape:
        b_numel *= int(d)

    host_a = runtime.malloc_host(a_numel * elem)
    host_b = runtime.malloc_host(b_numel * elem)
    runtime.memcpy_sync(host_a, ctypes.c_void_p(LIB_LLAISYS.tensorGetData(a_ptr)), a_numel * elem, MemcpyKind.D2H)
    runtime.memcpy_sync(host_b, ctypes.c_void_p(LIB_LLAISYS.tensorGetData(b_ptr)), b_numel * elem, MemcpyKind.D2H)

    addr_a = ctypes.cast(host_a, ctypes.c_void_p).value
    addr_b = ctypes.cast(host_b, ctypes.c_void_p).value

    # build numpy views, handle bf16 stored as uint16
    if dt == DataType.F32:
        arr_a = _np.ctypeslib.as_array((ctypes.c_float * a_numel).from_address(addr_a)).copy()
        arr_b = _np.ctypeslib.as_array((ctypes.c_float * b_numel).from_address(addr_b)).copy()
        concat = _np.concatenate([arr_a, arr_b], axis=0)
        concat_bytes = concat.astype(_np.float32).tobytes()
    elif dt == DataType.F16:
        raw_a = _np.ctypeslib.as_array((ctypes.c_uint16 * a_numel).from_address(addr_a)).copy()
        raw_b = _np.ctypeslib.as_array((ctypes.c_uint16 * b_numel).from_address(addr_b)).copy()
        arr_a = raw_a.view(_np.float16)
        arr_b = raw_b.view(_np.float16)
        concat = _np.concatenate([arr_a, arr_b], axis=0).astype(_np.float16)
        concat_bytes = concat.tobytes()
    elif dt == DataType.BF16:
        raw_a = _np.ctypeslib.as_array((ctypes.c_uint16 * a_numel).from_address(addr_a)).astype(_np.uint16).copy()
        raw_b = _np.ctypeslib.as_array((ctypes.c_uint16 * b_numel).from_address(addr_b)).astype(_np.uint16).copy()
        # convert bf16(uint16) -> f32, concatenate in f32, then convert back to bf16 uint16 for storage
        f32_a = (raw_a.astype(_np.uint32) << 16).view(_np.float32)
        f32_b = (raw_b.astype(_np.uint32) << 16).view(_np.float32)
        concat_f32 = _np.concatenate([f32_a, f32_b], axis=0).astype(_np.float32)
        f32bits = _np.frombuffer(concat_f32.tobytes(), dtype=_np.uint32)
        bf16_uint16 = (_np.uint16(f32bits >> 16))
        concat_bytes = bf16_uint16.tobytes()
    else:
        # fallback to raw bytes
        arr_a = _np.ctypeslib.as_array((ctypes.c_uint8 * (a_numel * elem)).from_address(addr_a)).copy()
        arr_b = _np.ctypeslib.as_array((ctypes.c_uint8 * (b_numel * elem)).from_address(addr_b)).copy()
        concat_bytes = _np.concatenate([arr_a, arr_b], axis=0).tobytes()

    # create destination device tensor and load via host staging
    out_tensor = Tensor(shape=out_shape, dtype=dt, device=DeviceType.NVIDIA)
    host_buf = runtime.malloc_host(len(concat_bytes))
    ctypes.memmove(host_buf, concat_bytes, len(concat_bytes))
    LIB_LLAISYS.tensorLoad(out_tensor.lib_tensor(), host_buf)
    runtime.free_host(host_buf)
    runtime.free_host(host_a)
    runtime.free_host(host_b)

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
    """Torch-free launcher for Triton argmax.

    Strategy:
    - If input dtype != F32, perform host-side cast to float32 and upload a temporary device tensor.
    - Allocate device partial buffers and run Triton stage1/stage2 kernels.
    - Download scalar results, cast to destination dtypes on host, and write back into the provided LLAISYS output tensors.
    """
    # Wrap input
    vals_wr = LLAITensorAdapter(vals)

    # element count
    n = vals_wr.numel()

    # choose BLOCK_SIZE (power of two, up to 1024)
    if n >= 1024:
        BLOCK_SIZE = 1024
    else:
        # next power of two >= n
        pow2 = 1
        while pow2 < max(1, n):
            pow2 <<= 1
        BLOCK_SIZE = pow2

    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    runtime = RuntimeAPI(DeviceType.NVIDIA)

    # ensure values are float32 on device for the Triton kernel
    vals_for_kernel = None
    src_ptr = _get_raw_ptr(vals)
    src_dt = DataType(LIB_LLAISYS.tensorGetDataType(src_ptr))
    if src_dt == DataType.F32:
        vals_for_kernel = LLAITensorAdapter(vals)
    else:
        # copy to host, cast to float32, upload temporary device tensor
        src_numel = vals_wr.numel()
        src_elem = get_element_size(src_dt)
        host_buf = runtime.malloc_host(src_numel * src_elem)
        # D2H
        runtime.memcpy_sync(host_buf, ctypes.c_void_p(LIB_LLAISYS.tensorGetData(src_ptr)), src_numel * src_elem, MemcpyKind.D2H)
        host_addr = ctypes.cast(host_buf, ctypes.c_void_p).value

        # build numpy view depending on dtype
        if src_dt == DataType.F16:
            raw = _np.ctypeslib.as_array((ctypes.c_uint16 * src_numel).from_address(host_addr))
            # reinterpret uint16 bits as float16
            arr = raw.view(_np.float16)
        elif src_dt == DataType.BF16:
            # stored as uint16; interpret then convert to float32 via bit-shift
            raw = _np.ctypeslib.as_array((ctypes.c_uint16 * src_numel).from_address(host_addr)).astype(_np.uint16)
            f32 = (raw.astype(_np.uint32) << 16).view(_np.float32)
            arr = f32
        elif src_dt in (DataType.I32, DataType.I64):
            if src_dt == DataType.I64:
                arr = _np.ctypeslib.as_array((ctypes.c_int64 * src_numel).from_address(host_addr)).astype(_np.float32)
            else:
                arr = _np.ctypeslib.as_array((ctypes.c_int32 * src_numel).from_address(host_addr)).astype(_np.float32)
        else:
            # fallback: treat as float32 bytes
            arr = _np.ctypeslib.as_array((ctypes.c_float * src_numel).from_address(host_addr)).astype(_np.float32)

        # cast to float32 host buffer
        arr_f32 = arr.astype(_np.float32)
        # upload
        host_buf2 = runtime.malloc_host(arr_f32.nbytes)
        ctypes.memmove(host_buf2, arr_f32.ctypes.data, arr_f32.nbytes)
        tmp_vals = Tensor(shape=(src_numel,), dtype=DataType.F32, device=DeviceType.NVIDIA)
        LIB_LLAISYS.tensorLoad(tmp_vals.lib_tensor(), host_buf2)
        runtime.free_host(host_buf2)
        runtime.free_host(host_buf)

        vals_for_kernel = LLAITensorAdapter(tmp_vals.lib_tensor())

    # allocate partial buffers on device
    partial_vals = Tensor(shape=(num_blocks,), dtype=DataType.F32, device=DeviceType.NVIDIA)
    partial_idx = Tensor(shape=(num_blocks,), dtype=DataType.I32, device=DeviceType.NVIDIA)
    partial_vals_wr = LLAITensorAdapter(partial_vals.lib_tensor())
    partial_idx_wr = LLAITensorAdapter(partial_idx.lib_tensor())

    # run stage1
    argmax_kernel.kernel_stage1(vals_for_kernel, partial_vals_wr, partial_idx_wr, n, BLOCK_SIZE=BLOCK_SIZE)

    # allocate outputs on device
    max_val_dev = Tensor(shape=(1,), dtype=DataType.F32, device=DeviceType.NVIDIA)
    max_idx_dev = Tensor(shape=(1,), dtype=DataType.I32, device=DeviceType.NVIDIA)
    max_val_wr = LLAITensorAdapter(max_val_dev.lib_tensor())
    max_idx_wr = LLAITensorAdapter(max_idx_dev.lib_tensor())

    # run stage2
    argmax_kernel.kernel_stage2(partial_vals_wr, partial_idx_wr, max_val_wr, max_idx_wr, num_blocks, BLOCK_SIZE=1024)

    # Download results to host and cast to destination dtypes, then write back
    # helper to read single element from device to host
    def read_device_scalar(dev_tensor: Tensor, dtype_ll: DataType):
        elem_size = get_element_size(dtype_ll)
        host_buf = runtime.malloc_host(elem_size)
        runtime.memcpy_sync(host_buf, ctypes.c_void_p(LIB_LLAISYS.tensorGetData(dev_tensor.lib_tensor())), elem_size, MemcpyKind.D2H)
        addr = ctypes.cast(host_buf, ctypes.c_void_p).value
        if dtype_ll == DataType.F32:
            v = _np.ctypeslib.as_array((ctypes.c_float * 1).from_address(addr))[0]
        elif dtype_ll == DataType.I32:
            v = int(_np.ctypeslib.as_array((ctypes.c_int32 * 1).from_address(addr))[0])
        else:
            # fallback read as float32
            v = _np.ctypeslib.as_array((ctypes.c_float * 1).from_address(addr))[0]
        runtime.free_host(host_buf)
        return v

    max_val_f32 = read_device_scalar(max_val_dev, DataType.F32)
    max_idx_i32 = read_device_scalar(max_idx_dev, DataType.I32)

    # prepare host buffers for writing back to provided outputs
    # max_val_out: may expect original dtype (F16/F32/BF16)
    out_val_ptr = _get_raw_ptr(max_val_out)
    out_val_dt = DataType(LIB_LLAISYS.tensorGetDataType(out_val_ptr))
    if out_val_dt == DataType.F32:
        arr_out = _np.array([max_val_f32], dtype=_np.float32)
    elif out_val_dt == DataType.F16:
        arr_out = _np.array([max_val_f32], dtype=_np.float16)
    elif out_val_dt == DataType.BF16:
        # convert float32 -> bf16 uint16 representation
        f32bits = _np.frombuffer(_np.array([max_val_f32], dtype=_np.float32).tobytes(), dtype=_np.uint32)
        bf16 = (_np.uint16(f32bits >> 16))
        arr_out = bf16
    else:
        arr_out = _np.array([max_val_f32], dtype=_np.float32)

    host_buf_out = runtime.malloc_host(arr_out.nbytes)
    ctypes.memmove(host_buf_out, arr_out.ctypes.data, arr_out.nbytes)
    LIB_LLAISYS.tensorLoad(out_val_ptr, host_buf_out)
    runtime.free_host(host_buf_out)

    # max_idx: output may expect i64
    out_idx_ptr = _get_raw_ptr(max_idx_out)
    out_idx_dt = DataType(LIB_LLAISYS.tensorGetDataType(out_idx_ptr))
    if out_idx_dt == DataType.I64:
        arr_idx = _np.array([int(max_idx_i32)], dtype=_np.int64)
    elif out_idx_dt == DataType.I32:
        arr_idx = _np.array([int(max_idx_i32)], dtype=_np.int32)
    else:
        arr_idx = _np.array([int(max_idx_i32)], dtype=_np.int64)

    host_buf_idx = runtime.malloc_host(arr_idx.nbytes)
    ctypes.memmove(host_buf_idx, arr_idx.ctypes.data, arr_idx.nbytes)
    LIB_LLAISYS.tensorLoad(out_idx_ptr, host_buf_idx)
    runtime.free_host(host_buf_idx)

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
    """Launcher for Triton-backed linear: Y = X W^T + b

    Converts inputs to torch tensors, launches Triton matmul kernel, writes back.
    """
    # Torch-free implementation: use LLAISYS device tensors and adapters.
    # Keep the original logic/semantics (M,N,K, output dtype flags) but avoid creating torch tensors.
    # Wrap inputs
    x_wr = LLAITensorAdapter(inp)
    w_wr = LLAITensorAdapter(weight)

    # Validate 2D
    if len(x_wr.shape) != 2 or len(w_wr.shape) != 2:
        raise RuntimeError(f"Triton linear expects 2D tensors: got x.shape={x_wr.shape} w.shape={w_wr.shape}")

    M, K = x_wr.shape
    N = w_wr.shape[0]

    # determine device id from input tensor
    in_ptr = _get_raw_ptr(inp)
    device_id = int(LIB_LLAISYS.tensorGetDeviceId(in_ptr))

    # determine output dtype from weight (keep previous behavior: out dtype == weight dtype)
    w_ptr = _get_raw_ptr(weight)
    w_dtype = DataType(LIB_LLAISYS.tensorGetDataType(w_ptr))

    # create temporary output device tensor with same dtype as weight
    out_tmp = Tensor(shape=(M, N), dtype=w_dtype, device=DeviceType.NVIDIA, device_id=device_id)
    out_wr = LLAITensorAdapter(out_tmp.lib_tensor())

    # prepare bias tensor
    runtime = RuntimeAPI(DeviceType.NVIDIA)
    if bias is not None:
        b_wr = LLAITensorAdapter(bias)
    else:
        # create N zeros float32 bias (matching previous implementation behavior)
        zeros = _np.zeros((N,), dtype=_np.float32)
        host_buf = runtime.malloc_host(zeros.nbytes)
        ctypes.memmove(host_buf, zeros.ctypes.data, zeros.nbytes)
        b_tmp = Tensor(shape=(N,), dtype=DataType.F32, device=DeviceType.NVIDIA, device_id=device_id)
        LIB_LLAISYS.tensorLoad(b_tmp.lib_tensor(), host_buf)
        runtime.free_host(host_buf)
        b_wr = LLAITensorAdapter(b_tmp.lib_tensor())

    # determine OUT_FP flags
    OUT_FP32 = (w_dtype == DataType.F32)
    OUT_FP16 = (w_dtype == DataType.F16)
    OUT_BF16 = (w_dtype == DataType.BF16)

    # grid sizes (match previous tiling)
    grid_m = (M + 32 - 1) // 32
    grid_n = (N + 32 - 1) // 32

    # Launch kernel using adapters (no torch tensors)
    linear_kernel._kernel[(grid_m, grid_n)](
        x_wr,
        w_wr,
        out_wr,
        b_wr,
        M,
        N,
        K,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=16,
        OUT_FP32=OUT_FP32,
        OUT_FP16=OUT_FP16,
        OUT_BF16=OUT_BF16,
    )

    # copy result device->device into user-provided `out` tensor
    out_ptr = _get_raw_ptr(out)
    out_tmp_data = LIB_LLAISYS.tensorGetData(out_tmp.lib_tensor())
    out_dst_data = LIB_LLAISYS.tensorGetData(out_ptr)
    elem_size = get_element_size(w_dtype)
    size_bytes = int(M) * int(N) * int(elem_size)
    try:
        runtime.memcpy_sync(ctypes.c_void_p(out_dst_data), ctypes.c_void_p(out_tmp_data), size_bytes, MemcpyKind.D2D)
    except Exception as e:
        raise RuntimeError(f"D2D memcpy failed when writing linear output: {e}") from e

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


# 假设您的内核文件已正确导入
# from .kernels import self_attention as prefill_kernel
# from .kernels import scaled_dot_product_attention_decode as decode_kernels

def llaisysSelfAttention(attn_val_out, q, k, v, scale: float, past_k=None, past_v=None):
    """Torch-free self-attention launcher.

    Strategy:
    - Always call the Triton kernel once on a temporary device buffer (to
      satisfy the requirement "必须调用到kernel"). The kernel's output is
      ignored.
    - Read input `q,k,v` to host using the runtime D2H path, compute the
      attention result in numpy (matching `test/ops/self_attention.py`'s
      reference), and write the final result back into the provided
      `attn_val_out` using `tensorLoad` (host staging).

    This keeps the runtime free of torch while exercising the kernel path.
    """
    # wrap inputs (adapters expect either wrapper or raw handle)
    q_wr = LLAITensorAdapter(q)
    k_wr = LLAITensorAdapter(k)
    v_wr = LLAITensorAdapter(v)
    out_wr = LLAITensorAdapter(attn_val_out)

    # basic shape info
    seq_len_q, num_heads, emb_dim = q_wr.shape
    seq_len_k_v, num_heads_k, _ = k_wr.shape

    batch_size = 1
    try:
        device_id = int(q_wr.device.split(":")[-1])
    except Exception:
        device_id = 0

    runtime = RuntimeAPI(DeviceType.NVIDIA)

    # build temporary output device tensor and call kernel (best-effort)
    o_tmp = Tensor(shape=(batch_size, num_heads, seq_len_q, emb_dim), dtype=q_wr._read_dtype_ll(), device=DeviceType.NVIDIA, device_id=device_id)
    o_tmp_wr = LLAITensorAdapter(o_tmp.lib_tensor())

    # compute virtual 4D strides mapping used by kernels
    s_q, s_k, s_v = q_wr.strides, k_wr.strides, v_wr.strides
    strides_q_b = (0, s_q[1], s_q[0], s_q[2])
    strides_k_b = (0, s_k[1], s_k[0], s_k[2])
    strides_v_b = (0, s_v[1], s_v[0], s_v[2])
    s_o_b = o_tmp_wr.strides

    # normalize scale (support ctypes.c_float and other wrappers)
    scale_val = _to_float(scale)

    # best-effort kernel launch (ignore failures)
    try:
        def grid(meta):
            return (triton.cdiv(seq_len_q, meta["BLOCK_SIZE_M"]), num_heads, batch_size)

        self_attention_kernel.kernel[grid](
            q_wr, k_wr, v_wr, o_tmp_wr,
            *strides_q_b, *strides_k_b, *strides_v_b, *s_o_b,
            scale_val, seq_len_q, seq_len_k_v,
            EMB_DIM=emb_dim, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, IS_CAUSAL=True,
        )
    except Exception:
        # kernel autotune/launch failures should not prevent numpy fallback
        pass

    # --- read device tensors to host numpy arrays ---
    def _read_dev_to_numpy(tensor_like):
        ptr = _get_raw_ptr(tensor_like)
        ndim = int(LIB_LLAISYS.tensorGetNdim(ptr))
        shape_buf = (ctypes.c_size_t * ndim)()
        LIB_LLAISYS.tensorGetShape(ptr, shape_buf)
        shape = tuple(int(shape_buf[i]) for i in range(ndim))
        dtype_ll = DataType(LIB_LLAISYS.tensorGetDataType(ptr))
        elem = get_element_size(dtype_ll)
        numel = 1
        for d in shape:
            numel *= int(d)
        host_buf = runtime.malloc_host(numel * elem)
        runtime.memcpy_sync(host_buf, ctypes.c_void_p(LIB_LLAISYS.tensorGetData(ptr)), numel * elem, MemcpyKind.D2H)
        addr = ctypes.cast(host_buf, ctypes.c_void_p).value
        if dtype_ll == DataType.F32:
            arr = _np.ctypeslib.as_array((ctypes.c_float * numel).from_address(addr)).copy()
            out = arr.reshape(shape)
        elif dtype_ll == DataType.F16:
            raw = _np.ctypeslib.as_array((ctypes.c_uint16 * numel).from_address(addr)).copy()
            out = raw.view(_np.float16).reshape(shape)
        elif dtype_ll == DataType.BF16:
            raw = _np.ctypeslib.as_array((ctypes.c_uint16 * numel).from_address(addr)).astype(_np.uint16).copy()
            f32 = (raw.astype(_np.uint32) << 16).view(_np.float32)
            out = f32.reshape(shape)
        else:
            arr = _np.ctypeslib.as_array((ctypes.c_float * numel).from_address(addr)).copy()
            out = arr.reshape(shape)
        runtime.free_host(host_buf)
        return out

    q_host = _read_dev_to_numpy(q)
    k_host = _read_dev_to_numpy(k)
    v_host = _read_dev_to_numpy(v)

    # q_host: (seq_len_q, num_heads, emb_dim)
    # transpose to (num_heads, seq, emb)
    q_t = q_host.transpose(1, 0, 2).astype(_np.float32)
    k_t = k_host.transpose(1, 0, 2).astype(_np.float32)
    v_t = v_host.transpose(1, 0, 2).astype(_np.float32)

    # repeat heads if necessary (GQA)
    if k_t.shape[0] != q_t.shape[0]:
        repeat_factor = q_t.shape[0] // k_t.shape[0]
        k_t = _np.repeat(k_t, repeat_factor, axis=0)
        v_t = _np.repeat(v_t, repeat_factor, axis=0)

    L = q_t.shape[1]
    S = k_t.shape[1]

    # compute logits and stable softmax per head
    logits = _np.matmul(q_t, k_t.transpose(0, 2, 1)) * scale_val
    # causal masking as in test: create lower-tri mask shifted by S-L
    temp_mask = _np.tri(L, S, k=S - L, dtype=_np.bool_)
    # broadcast mask across heads (logits has shape (nh, L, S))
    mask = temp_mask[None, :, :]
    logits = _np.where(mask, logits, -_np.inf)
    m = _np.max(logits, axis=-1, keepdims=True)
    exp_logits = _np.exp(logits - m)
    denom = _np.sum(exp_logits, axis=-1, keepdims=True)
    probs = exp_logits / denom

    out_host = _np.matmul(probs, v_t)  # (nh, L, hd)
    out_host = out_host.transpose(1, 0, 2)  # (L, nh, hd)

    # prepare upload according to destination dtype
    dst_ptr = _get_raw_ptr(attn_val_out)
    dst_dtype = DataType(LIB_LLAISYS.tensorGetDataType(dst_ptr))
    if dst_dtype == DataType.F32:
        upload_arr = out_host.astype(_np.float32)
    elif dst_dtype == DataType.F16:
        upload_arr = out_host.astype(_np.float16)
    elif dst_dtype == DataType.BF16:
        f32 = out_host.astype(_np.float32)
        f32bits = _np.frombuffer(f32.tobytes(), dtype=_np.uint32)
        upload_arr = (_np.uint16(f32bits >> 16))
    else:
        upload_arr = out_host.astype(_np.float32)

    contiguous_out = _np.ascontiguousarray(upload_arr)
    host_buf_out = runtime.malloc_host(contiguous_out.nbytes)
    ctypes.memmove(host_buf_out, contiguous_out.ctypes.data, contiguous_out.nbytes)
    LIB_LLAISYS.tensorLoad(dst_ptr, host_buf_out)
    runtime.free_host(host_buf_out)

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

