from .kernels import add as add_kernel
from .kernels import argmax as argmax_kernel
from .kernels import embedding as embedding_kernel
from .kernels import linear as linear_kernel
from .kernels import rearrange as rearrange_kernel
from .kernels import rms_norm as rms_norm_kernel    
from .kernels import rope as rope_kernel
from .kernels import self_attention as self_attention_kernel
from .kernels import swiglu as swiglu_kernel

# import torch

import ctypes
import numpy as _np
import torch

from llaisys.runtime import RuntimeAPI
from llaisys.libllaisys import DeviceType, MemcpyKind, DataType, LIB_LLAISYS


def _get_raw_ptr(x):
    # Accept either a Python wrapper (with lib_tensor) or a raw handle
    if hasattr(x, "lib_tensor"):
        return x.lib_tensor()
    return x


def to_torch_tensor(x):
    """Convert a LLAISYS tensor (wrapper or raw handle) into a CUDA torch.Tensor.

    Preserves dtype for f32/f16/bf16 and places the result on the original device id.
    """
    ptr = _get_raw_ptr(x)
    runtime = RuntimeAPI(DeviceType.NVIDIA)

    def _ndim(p):
        return int(LIB_LLAISYS.tensorGetNdim(p))

    def _shape(p):
        ndim = _ndim(p)
        buf = (ctypes.c_size_t * ndim)()
        LIB_LLAISYS.tensorGetShape(p, buf)
        return tuple(buf[i] for i in range(ndim))

    def _dtype(p):
        return DataType(LIB_LLAISYS.tensorGetDataType(p))

    def _device_id(p):
        return int(LIB_LLAISYS.tensorGetDeviceId(p))

    def _data_ptr(p):
        return LIB_LLAISYS.tensorGetData(p)

    shape = _shape(ptr)
    dtype = _dtype(ptr)
    device_id = _device_id(ptr)

    if dtype == DataType.F32:
        td = torch.float32
        np_dtype = _np.float32
    elif dtype == DataType.F16:
        td = torch.float16
        np_dtype = _np.float16
    elif dtype == DataType.BF16:
        td = torch.bfloat16
        np_dtype = _np.uint16
    else:
        raise TypeError(f"Unsupported dtype for Triton kernel: {dtype}")

    numel = 1
    for d in shape:
        numel *= int(d)
    elem_size = _np.dtype(np_dtype).itemsize
    size_bytes = numel * elem_size

    host_ptr = runtime.malloc_host(size_bytes)
    try:
        runtime.memcpy_sync(host_ptr, ctypes.c_void_p(_data_ptr(ptr)), size_bytes, MemcpyKind.D2H)
        raw = ctypes.string_at(host_ptr, size_bytes)
        if dtype == DataType.BF16:
            arr_u16 = _np.frombuffer(raw, dtype=_np.uint16).copy()
            arr_u16 = arr_u16.reshape(tuple(shape))
            arr_u32 = (arr_u16.astype(_np.uint32) << 16)
            arr_f32 = arr_u32.view(_np.float32)
            th = torch.from_numpy(arr_f32).to(device=f"cuda:{device_id}", dtype=td)
        else:
            arr = _np.frombuffer(raw, dtype=np_dtype).copy()
            arr = arr.reshape(tuple(shape))
            th = torch.from_numpy(arr).to(device=f"cuda:{device_id}", dtype=td)
        try:
            if th.numel() <= 64:
                print("[triton.setup] converted tensor (device):", th)
        except Exception:
            pass
        return th
    finally:
        runtime.free_host(host_ptr)


def from_torch_to_ptr(th, out):
    """Write a CPU-contiguous torch tensor `th` back into an output LLAISYS tensor handle or wrapper."""
    out_ptr = _get_raw_ptr(out)
    cpu = th.to("cpu").contiguous()

    class _View:
        def __init__(self, ptr):
            self._ptr = ptr

        def load(self, data):
            LIB_LLAISYS.tensorLoad(self._ptr, data)

    view = _View(out_ptr)
    view.load(ctypes.c_void_p(cpu.data_ptr()))
    
    
# add examples:
def llaisysAdd(out, a, b):
    """Launcher that bridges LLAISYS tensors to Triton kernel.

    Note: `Ops.add` calls this as `llaisysAdd(out, a, b)`, so we accept
    `(out, a, b)` and invoke the Triton kernel as `kernel(a, b, out)`.
    """
    print("Using Triton add kernel")
    # Convert inputs to torch tensors (if they are LLAISYS handles)
    a_t = to_torch_tensor(a) if not isinstance(a, torch.Tensor) else a
    b_t = to_torch_tensor(b) if not isinstance(b, torch.Tensor) else b

    # allocate output torch tensor on same device/dtype as inputs
    c_t = torch.empty_like(a_t)

    # call Triton kernel with (a_t, b_t, c_t)
    add_kernel.kernel(a_t, b_t, c_t, BLOCK_SIZE=1024)

    # write back to LLAISYS output if needed
    try:
        if not isinstance(out, torch.Tensor):
            from_torch_to_ptr(c_t, out)
        else:
            # if caller provided a torch tensor as out, copy into it
            out.copy_(c_t)
    except Exception:
        # best-effort: ignore write-back errors here
        pass

    return out

# implement other operators below

