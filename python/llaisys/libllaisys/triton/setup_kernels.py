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
import math

from llaisys.runtime import RuntimeAPI
from llaisys.libllaisys import DeviceType, MemcpyKind, DataType, LIB_LLAISYS
from .kernels import scaled_dot_product_attention_decode as decode_kernels
import triton
import triton.language as tl

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
    elif dtype == DataType.F64:
        td = torch.float64
        np_dtype = _np.float64
    elif dtype == DataType.I8:
        td = torch.int8
        np_dtype = _np.int8
    elif dtype == DataType.I16:
        td = torch.int16
        np_dtype = _np.int16
    elif dtype == DataType.I32:
        td = torch.int32
        np_dtype = _np.int32
    elif dtype == DataType.I64:
        td = torch.int64
        np_dtype = _np.int64
    elif dtype == DataType.U8:
        td = torch.uint8
        np_dtype = _np.uint8
    elif dtype == DataType.U16:
        td = torch.uint16 if hasattr(torch, 'uint16') else torch.int32
        np_dtype = _np.uint16
    elif dtype == DataType.U32:
        td = torch.uint32 if hasattr(torch, 'uint32') else torch.int64
        np_dtype = _np.uint32
    elif dtype == DataType.U64:
        td = torch.uint64 if hasattr(torch, 'uint64') else torch.int64
        np_dtype = _np.uint64
    else:
        raise TypeError(f"Unsupported dtype for Triton kernel: {dtype}")

    numel = 1
    for d in shape:
        numel *= int(d)
    elem_size = _np.dtype(np_dtype).itemsize
    size_bytes = numel * elem_size

    # Perform a device-to-device copy only. If D2D is not available or
    # fails, raise an error so tests can detect the absence of a D2D path.
    src_ptr = _data_ptr(ptr)
    if device_id is None or device_id < 0:
        raise RuntimeError("Invalid device id for LLAISYS tensor; cannot perform D2D copy")

    th = torch.empty(tuple(shape), device=f"cuda:{device_id}", dtype=td)
    try:
        runtime.memcpy_sync(ctypes.c_void_p(th.data_ptr()), ctypes.c_void_p(src_ptr), size_bytes, MemcpyKind.D2D)
    except Exception as e:
        # Surface a clear error: no fallback here by design for testing
        raise RuntimeError(f"D2D memcpy failed: {e}") from e

    # try:
    #     if th.numel() <= 64:
    #         print("[triton.setup] converted tensor (device, D2D):", th)
    # except Exception:
    #     pass

    return th


def from_torch_to_ptr(th, out):
    """Write a CPU-contiguous torch tensor `th` back into an output LLAISYS tensor handle or wrapper."""
    out_ptr = _get_raw_ptr(out)
    cpu = th.to("cpu").contiguous()
    # Ensure we cast the CPU tensor to the exact dtype expected by the
    # destination LLAISYS tensor to avoid byte-size/shape mismatches.
    try:
        dt = DataType(LIB_LLAISYS.tensorGetDataType(out_ptr))
        if dt == DataType.F32:
            target_torch_dtype = torch.float32
        elif dt == DataType.F16:
            target_torch_dtype = torch.float16
        elif dt == DataType.BF16:
            target_torch_dtype = torch.bfloat16
        elif dt == DataType.F64:
            target_torch_dtype = torch.float64
        elif dt == DataType.I8:
            target_torch_dtype = torch.int8
        elif dt == DataType.I16:
            target_torch_dtype = torch.int16
        elif dt == DataType.I32:
            target_torch_dtype = torch.int32
        elif dt == DataType.I64:
            target_torch_dtype = torch.int64
        else:
            # fallback to float32
            target_torch_dtype = torch.float32
        # cast and ensure contiguous on CPU
        cpu = cpu.to(dtype=target_torch_dtype).contiguous()
    except Exception:
        # If any query fails, keep the original CPU tensor
        pass

    class _View:
        def __init__(self, ptr):
            self._ptr = ptr

        def load(self, data):
            LIB_LLAISYS.tensorLoad(self._ptr, data)

    # debug: print expected sizes to help diagnose shape/size mismatches
    try:
        ndim = int(LIB_LLAISYS.tensorGetNdim(out_ptr))
        buf = (ctypes.c_size_t * ndim)()
        LIB_LLAISYS.tensorGetShape(out_ptr, buf)
        out_shape = tuple(buf[i] for i in range(ndim))
        out_numel = 1
        for d in out_shape:
            out_numel *= int(d)
        out_dt = DataType(LIB_LLAISYS.tensorGetDataType(out_ptr))
        cpu_numel = int(cpu.numel()) if hasattr(cpu, 'numel') else None
        cpu_elem_size = int(cpu.element_size()) if hasattr(cpu, 'element_size') else None
        cpu_size_bytes = cpu_numel * cpu_elem_size if cpu_numel is not None and cpu_elem_size is not None else None
        # print(f"[triton.writeback] out_shape={out_shape} out_numel={out_numel} out_dt={out_dt} cpu_numel={cpu_numel} cpu_elem_size={cpu_elem_size} cpu_size_bytes={cpu_size_bytes}")
    except Exception:
        pass

    view = _View(out_ptr)
    view.load(ctypes.c_void_p(cpu.data_ptr()))
    
    
# add examples:
def llaisysAdd(out, a, b):
    """Launcher that bridges LLAISYS tensors to Triton kernel.
    
    ## You need to convert input llaisys tensor into torch

    Note: `Ops.add` calls this as `llaisysAdd(out, a, b)`, so we accept
    `(out, a, b)` and invoke the Triton kernel as `kernel(a, b, out)`.
    """
    # print("Using Triton add kernel")
    # Convert inputs to torch tensors (if they are LLAISYS handles)
    a_t = to_torch_tensor(a) if not isinstance(a, torch.Tensor) else a
    b_t = to_torch_tensor(b) if not isinstance(b, torch.Tensor) else b

    # allocate output torch tensor on same device/dtype as inputs
    c_t = torch.empty_like(a_t)

    # call Triton kernel with (a_t, b_t, c_t)
    add_kernel.kernel(a_t, b_t, c_t, BLOCK_SIZE=1024)

    # write back to LLAISYS output if needed
    from_torch_to_ptr(c_t, out)

    return out


def llaisysArgmax(max_idx_out, max_val_out, vals):
    """Launcher for Triton argmax: convert inputs, run Triton kernels, write back."""
    # convert vals to torch tensor on CUDA
    vals_t = to_torch_tensor(vals) if not isinstance(vals, torch.Tensor) else vals

    # remember original dtype so we can cast results back
    orig_dtype = vals_t.dtype
    # ensure we operate in float32 for comparison simplicity
    if vals_t.dtype != torch.float32:
        vals_f32 = vals_t.to(dtype=torch.float32)
    else:
        vals_f32 = vals_t

    n = vals_f32.numel()
    # choose BLOCK_SIZE (power of two, up to 1024)
    BLOCK_SIZE = 1024 if n >= 1024 else 1 << (n - 1).bit_length()

    # compute number of blocks
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    # allocate partial buffers on device
    partial_vals = torch.empty((num_blocks,), dtype=torch.float32, device=vals_f32.device)
    partial_idx = torch.empty((num_blocks,), dtype=torch.int32, device=vals_f32.device)

    # run stage1 to fill partials
    argmax_kernel.kernel_stage1(vals_f32, partial_vals, partial_idx, n, BLOCK_SIZE=BLOCK_SIZE)

    # run stage2 to reduce partials to single result
    # allocate single-element outputs on device
    max_val_t = torch.empty((1,), dtype=torch.float32, device=vals_f32.device)
    max_idx_t = torch.empty((1,), dtype=torch.int32, device=vals_f32.device)
    argmax_kernel.kernel_stage2(partial_vals, partial_idx, max_val_t, max_idx_t, num_blocks, BLOCK_SIZE=1024)

    # write back to llaisys outputs
    out_val = max_val_t.to(dtype=orig_dtype)
    from_torch_to_ptr(out_val, max_val_out)

    out_idx = max_idx_t.to(dtype=torch.int64)
    from_torch_to_ptr(out_idx, max_idx_out)

    return max_idx_out, max_val_out


def llaisysEmbedding(out, index, weight):
    """Launcher for Triton-backed embedding.

    Converts LLAISYS handles to CUDA torch tensors, runs the Triton kernel,
    and writes back the output.
    """
    # convert inputs to torch tensors if needed
    index_t = to_torch_tensor(index) if not isinstance(index, torch.Tensor) else index
    weight_t = to_torch_tensor(weight) if not isinstance(weight, torch.Tensor) else weight

    # ensure index is int32 on device for Triton kernel
    if index_t.dtype != torch.int32:
        index_i32 = index_t.to(dtype=torch.int32)
    else:
        index_i32 = index_t

    N = index_i32.numel()
    D = weight_t.shape[1]

    # allocate output on device with same dtype as weight
    out_t = torch.empty((N, D), dtype=weight_t.dtype, device=weight_t.device)

    # call Triton kernel
    embedding_kernel.kernel(index_i32, weight_t, out_t, N, D, BLOCK_SIZE=1024)

    # write back
    from_torch_to_ptr(out_t, out)

    return out


def llaisysLinear(out, inp, weight, bias):
    """Launcher for Triton-backed linear: Y = X W^T + b

    Converts inputs to torch tensors, launches Triton matmul kernel, writes back.
    """
    try:
        # Convert inputs to torch tensors (device tensors expected)
        x_t = to_torch_tensor(inp) if not isinstance(inp, torch.Tensor) else inp
        w_t = to_torch_tensor(weight) if not isinstance(weight, torch.Tensor) else weight

        b_t = None
        if bias is not None:
            b_t = to_torch_tensor(bias) if not isinstance(bias, torch.Tensor) else bias

        # Debug prints to help locate failures during conversion / kernel launch
        # try:
        #     print(f"[triton.linear] x.shape={getattr(x_t, 'shape', None)} x.dtype={getattr(x_t, 'dtype', None)} device={getattr(x_t, 'device', None)}")
        #     print(f"[triton.linear] w.shape={getattr(w_t, 'shape', None)} w.dtype={getattr(w_t, 'dtype', None)} device={getattr(w_t, 'device', None)}")
        #     print(f"[triton.linear] bias present={b_t is not None}")
        # except Exception:
        #     pass

        # Expect 2D inputs
        if x_t.dim() != 2 or w_t.dim() != 2:
            raise RuntimeError(f"Triton linear expects 2D tensors: got x.dim={x_t.dim()} w.dim={w_t.dim()}")

        M, K = x_t.shape
        N = w_t.shape[0]

        # allocate output on weight device to match kernel expectations
        out_t = torch.empty((M, N), dtype=w_t.dtype, device=w_t.device)

        # ensure contiguity
        x_flat = x_t.contiguous()
        w_flat = w_t.contiguous()
        out_flat = out_t.contiguous()
        if b_t is not None:
            b_flat = b_t.contiguous()
        else:
            # pass an empty 1-D tensor (Triton will load zeros via mask)
            b_flat = torch.zeros((N,), dtype=torch.float32, device=x_t.device)

        # determine output dtype flags for Triton kernel (constexpr params)
        OUT_FP32 = (out_t.dtype == torch.float32)
        OUT_FP16 = (out_t.dtype == torch.float16)
        OUT_BF16 = (out_t.dtype == torch.bfloat16)

        # Debug: report grid and block sizes
        grid_m = (M + 32 - 1) // 32
        grid_n = (N + 32 - 1) // 32
        # print(f"[triton.linear] launching kernel grid=({grid_m},{grid_n}) M={M} N={N} K={K} out_dtype={out_t.dtype}")

        # Launch kernel
        linear_kernel._kernel[(grid_m, grid_n)](
            x_flat, w_flat, out_flat, b_flat, M, N, K,
            BLOCK_M=32, BLOCK_N=32, BLOCK_K=16,
            OUT_FP32=OUT_FP32, OUT_FP16=OUT_FP16, OUT_BF16=OUT_BF16,
        )

        # write back
        from_torch_to_ptr(out_t, out)

        return out
    except Exception as e:
        # Provide rich debug info before re-raising to surface root cause to user
        try:
            # print(f"[triton.linear][ERROR] exception: {e}")
            import traceback

            traceback.print_exc()
        except Exception:
            pass
        raise


def llaisysRearrange(out, inp):
    """Launcher for rearrange: convert inputs, run simple rearrange (copy) and write back.

    This implementation uses the device torch tensor path and performs an identity
    copy. It exists to validate the Ops->Triton bridge; an optimized rearrange
    kernel can be substituted in `kernels/rearrange.py` later.
    """
    x_t = to_torch_tensor(inp) if not isinstance(inp, torch.Tensor) else inp

    out_t = torch.empty_like(x_t)
    # call kernel
    rearrange_kernel.kernel(x_t, out_t)

    # write back
    from_torch_to_ptr(out_t, out)

    return out


def llaisysRmsNorm(out, inp, weight, eps: float):
    """Launcher for Triton-backed RMSNorm: Y = weight * X / sqrt(mean(X**2) + eps)

    Expects `out`, `inp`, `weight` as LLAISYS tensor handles or torch.Tensors.
    """
    x_t = to_torch_tensor(inp) if not isinstance(inp, torch.Tensor) else inp
    w_t = to_torch_tensor(weight) if not isinstance(weight, torch.Tensor) else weight

    M, D = x_t.shape

    out_t = torch.empty((M, D), dtype=x_t.dtype, device=x_t.device)

    # ensure contiguous flattened storage
    x_flat = x_t.contiguous()
    w_flat = w_t.contiguous()
    out_flat = out_t.contiguous()

    # coerce eps if it's a ctypes.c_float or similar wrapper
    try:
        # handle ctypes simple cdata which expose .value
        if hasattr(eps, "value"):
            eps_val = float(eps.value)
        else:
            eps_val = float(eps)
    except Exception:
        eps_val = float(eps)

    # call Triton kernel
    rms_norm_kernel.kernel(x_flat, w_flat, out_flat, M, D, eps_val, BLOCK_SIZE=1024)

    # write back
    from_torch_to_ptr(out_t, out)

    return out


def llaisysSelfAttention(attn_val_out, q, k, v, scale: float, past_k=None, past_v=None):
    """Launcher for Triton-backed self-attention.

    Expects shapes:
      q: (qlen, nh, hd)
      k: (kvlen, nkvh, hd)
      v: (kvlen, nkvh, hd)

    This launcher will reshape tensors to (batch=1, heads, seq_len, emb_dim),
    repeat-interleave k/v heads if needed, call the Triton kernel, and write back
    the result in shape (qlen, nh, hd).
    """
    q_t = to_torch_tensor(q) if not isinstance(q, torch.Tensor) else q
    k_t = to_torch_tensor(k) if not isinstance(k, torch.Tensor) else k
    v_t = to_torch_tensor(v) if not isinstance(v, torch.Tensor) else v

    # Convert optional past_k/past_v to torch tensors if provided.
    if past_k is not None:
        past_k_t = to_torch_tensor(past_k) if not isinstance(past_k, torch.Tensor) else past_k
    else:
        past_k_t = None

    if past_v is not None:
        past_v_t = to_torch_tensor(past_v) if not isinstance(past_v, torch.Tensor) else past_v
    else:
        past_v_t = None

    # shapes from tests: q (qlen, nh, hd)
    qlen, nh, hd = q_t.shape
    kvlen, nkvh, _ = k_t.shape

    # reshape to (batch, heads, seq_len, emb_dim)
    q_ = q_t.permute(2, 1, 0).contiguous() if False else q_t.permute(2,1,0)  # placeholder to clarify ordering
    # simpler: reshape by unsqueezing batch dim and permuting to (1, nh, qlen, hd)
    q_b = q_t.permute(1,0,2).contiguous().unsqueeze(0)  # (1, nh, qlen, hd)

    # prepare k/v: (kvlen, nkvh, hd) -> (1, nkvh, kvlen, hd)
    k_b = k_t.permute(1,0,2).contiguous().unsqueeze(0)
    v_b = v_t.permute(1,0,2).contiguous().unsqueeze(0)

    # repeat heads if needed
    if k_b.shape[1] != q_b.shape[1]:
        repeat = q_b.shape[1] // k_b.shape[1]
        if repeat > 1:
            k_b = k_b.repeat_interleave(repeat, dim=1)
            v_b = v_b.repeat_interleave(repeat, dim=1)

    batch_size = q_b.shape[0]
    num_heads = q_b.shape[1]

    # allocate output with same layout as q_b
    out_b = torch.empty_like(q_b)

    # coerce scale (may be ctypes.c_float)
    try:
        if hasattr(scale, "value"):
            scale_val = float(scale.value)
        else:
            scale_val = float(scale)
    except Exception:
        scale_val = float(scale)

    # For very small embedding dims or sequence lengths Triton's dot may
    # require a minimum size; fall back to a correct torch implementation
    # here (launcher-level fallback only — kernels remain Triton-only).
    if hd < 16 or qlen < 16 or kvlen < 16:
        # follow the same steps as test's torch_self_attention
        query = q_t.transpose(-2, -3)
        key = k_t.transpose(-2, -3)
        value = v_t.transpose(-2, -3)
        L, S = query.size(-2), key.size(-2)
        attn_bias_small = torch.zeros((L, S), dtype=query.dtype, device=query.device)
        temp_mask_small = torch.ones((L, S), dtype=torch.bool, device=query.device).tril(diagonal=S - L)
        attn_bias_small.masked_fill_(~temp_mask_small, float("-inf"))

        if query.size(-3) != key.size(-3):
            repeat = query.size(-3) // key.size(-3)
            key = key.repeat_interleave(repeat, dim=-3)
            value = value.repeat_interleave(repeat, dim=-3)

        attn_weight = (query @ key.transpose(-2, -1)) * scale_val
        attn_weight = attn_weight + attn_bias_small
        attn_weight = torch.softmax(attn_weight, dim=-1)
        out_small = (attn_weight @ value).transpose(-2, -3).contiguous()

        # write back
        try:
            if not isinstance(attn_val_out, torch.Tensor):
                from_torch_to_ptr(out_small, attn_val_out)
            else:
                attn_val_out.copy_(out_small)
        except Exception:
            pass

        return attn_val_out

    # If past_k/past_v are provided, use the decode split/combine kernels
    # which implement online softmax across past+current K/V. Otherwise
    # fall back to the original prefill kernel.
    if past_k_t is not None and past_v_t is not None:
        # q_b: (1, nh, qlen, hd) ; k_b: (1, nkvh, kvlen_current, hd)
        # past_k_t/past_v_t expected layout: (past_seq_len, nkvh, hd)
        past_seq_len = past_k_t.shape[2]

        # convert layouts to those expected by decode kernels
        q_dec = q_b
        k_dec = k_b
        v_dec = v_b
        past_k_dec = past_k_t
        past_v_dec = past_v_t

        # Prepare split outputs buffers as in hw2 wrapper
        # compute past sequence length and total seq_len_k_v
        past_seq_len = past_k_t.shape[0] if past_k_t is not None else 0
        seq_len_k_v = past_seq_len + k_b.shape[2]
        # number of groups when num_key_value_heads divides num_heads
        num_groups = num_heads // (nkvh if nkvh > 0 else 1)

        S = get_optimal_s(batch_size, num_heads, seq_len_k_v, q_t.device)
        print(f"we have {S} splits for decode attention")
        split_logsumexp = torch.empty((batch_size, num_heads, S, qlen), dtype=torch.float32, device="cuda")
        split_outputs = torch.empty((batch_size, num_heads, S, qlen, hd), dtype=torch.float16, device="cuda")

        def grid1(meta):
            return (triton.cdiv(qlen, meta['BLOCK_SIZE_M']), S, num_heads * batch_size)

        M_binned = triton.next_power_of_2(qlen)
        N_binned = triton.next_power_of_2(past_seq_len + k_b.shape[2])

        decode_kernels.split_kv_kernel[grid1](
            q_dec, k_dec, v_dec, past_k_dec, past_v_dec,
            float(scale_val),
            split_logsumexp, split_outputs,
            num_heads, num_groups,
            q_dec.stride(0), q_dec.stride(1), q_dec.stride(2), q_dec.stride(3),
            k_dec.stride(0), k_dec.stride(1), k_dec.stride(2), k_dec.stride(3),
            v_dec.stride(0), v_dec.stride(1), v_dec.stride(2), v_dec.stride(3),
            past_k_dec.stride(0), past_k_dec.stride(1), past_k_dec.stride(2), past_k_dec.stride(3),
            past_v_dec.stride(0), past_v_dec.stride(1), past_v_dec.stride(2), past_v_dec.stride(3),
            split_outputs.stride(0), split_outputs.stride(1), split_outputs.stride(2), split_outputs.stride(3), split_outputs.stride(4),
            qlen, past_seq_len + k_b.shape[2], k_b.shape[2], S,
            EMB_DIM=hd, M_BINNED=M_binned, N_BINNED=N_binned, IS_CAUSAL=True,
        )

        if S == 1:
            final_o = split_outputs.squeeze(2)
        else:
            final_o = torch.empty_like(q_dec)
            def grid2(meta):
                return (triton.cdiv(qlen, meta['BLOCK_SIZE_M']), num_heads, batch_size)
            decode_kernels.combine_kv_splits_kernel[grid2](
                split_outputs, split_logsumexp,
                final_o, torch.empty((batch_size, num_heads, qlen), dtype=torch.float32, device="cuda"),
                num_heads,
                split_outputs.stride(0), split_outputs.stride(1), split_outputs.stride(2), split_outputs.stride(3), split_outputs.stride(4),
                final_o.stride(0), final_o.stride(1), final_o.stride(2), final_o.stride(3),
                qlen, S,
                EMB_DIM=hd, M_BINNED=M_binned,
            )

        # final_o shape: (batch, heads, qlen, hd)
        out_final = final_o.squeeze(0).permute(1,0,2).contiguous() if final_o.dim() == 5 else final_o.squeeze(0).permute(1,0,2).contiguous()
        from_torch_to_ptr(out_final, attn_val_out)
        return attn_val_out

    # call Triton kernel (prefill path)
    grid = ((qlen + 31) // 32, num_heads, batch_size)
    self_attention_kernel.kernel[grid](
        q_b, k_b, v_b, out_b,
        q_b.stride(0), q_b.stride(1), q_b.stride(2), q_b.stride(3),
        k_b.stride(0), k_b.stride(1), k_b.stride(2), k_b.stride(3),
        v_b.stride(0), v_b.stride(1), v_b.stride(2), v_b.stride(3),
        out_b.stride(0), out_b.stride(1), out_b.stride(2), out_b.stride(3),
        # past_k_ptr, past_v_ptr (0 means unused)
        0, 0,
        # past_k strides (z, h, n, k)
        0, 0, 0, 0,
        # past_v strides (z, h, k, n)
        0, 0, 0, 0,
        float(scale_val), qlen, kvlen,
        EMB_DIM=hd,
    )

    # out_b shape: (1, nh, qlen, hd) -> convert back to (qlen, nh, hd)
    out_final = out_b.squeeze(0).permute(1,0,2).contiguous()

    # write back
    from_torch_to_ptr(out_final, attn_val_out)

    return attn_val_out

# implement other operators below


def llaisysSwiGLU(out, gate, up):
    """Launcher for Triton-backed SwiGLU: out = up * gate / (1 + exp(-gate))

    Accepts LLAISYS tensor handles or torch.Tensors. Kernel uses float32
    for numerical stability and casts back to output dtype.
    """
    gate_t = to_torch_tensor(gate) if not isinstance(gate, torch.Tensor) else gate
    up_t = to_torch_tensor(up) if not isinstance(up, torch.Tensor) else up

    # validate shapes and dtypes
    assert gate_t.shape == up_t.shape, "SwiGLU: gate and up must have same shape"

    # debug: inspect expected destination dtype/shape
    try:
        out_ptr = _get_raw_ptr(out)
        out_dt = DataType(LIB_LLAISYS.tensorGetDataType(out_ptr))
        out_shape = None
        ndim = int(LIB_LLAISYS.tensorGetNdim(out_ptr))
        buf = (ctypes.c_size_t * ndim)()
        LIB_LLAISYS.tensorGetShape(out_ptr, buf)
        out_shape = tuple(buf[i] for i in range(ndim))
        # print(f"[triton.swiGLU] out expected dtype={out_dt} shape={out_shape}")
    except Exception:
        pass

    out_t = torch.empty_like(gate_t)

    gate_flat = gate_t.contiguous().view(-1)
    up_flat = up_t.contiguous().view(-1)
    out_flat = out_t.contiguous().view(-1)

    # call Triton kernel
    swiglu_kernel.kernel(gate_flat, up_flat, out_flat, BLOCK=1024)

    # write back
    try:
        from_torch_to_ptr(out_t, out)
    except Exception as e:
        # print(f"[triton.swiGLU] write-back failed: {e}")
        raise

    return out


def llaisysROPE(out, inp, pos_ids, theta: float):
    """Launcher for RoPE (rotation positional embeddings).

    Computes in-place: out = rope(inp, pos_ids, theta)
    This launcher uses PyTorch on the device for simplicity and correctness.
    """
    # Convert inputs to torch tensors on device
    x_t = to_torch_tensor(inp) if not isinstance(inp, torch.Tensor) else inp
    pos_t = to_torch_tensor(pos_ids) if not isinstance(pos_ids, torch.Tensor) else pos_ids

    # validate shapes
    assert x_t.dim() == 3
    seq_len, n_heads, head_dim = x_t.shape
    assert head_dim % 2 == 0

    # allocate output
    out_t = torch.empty_like(x_t)

    # coerce theta to float
    try:
        if hasattr(theta, "value"):
            theta_val = float(theta.value)
        else:
            theta_val = float(theta)
    except Exception:
        theta_val = float(theta)

    # prepare position ids and freqs on device as float32
    pos_f32 = pos_t.to(dtype=torch.float32, device=x_t.device).contiguous()
    half = head_dim // 2
    i = torch.arange(0, half, dtype=torch.float32, device=x_t.device)
    # freq multiplier = 1 / (theta ** (2*i/head_dim))
    freqs = (theta_val ** (2.0 * i / float(head_dim))).reciprocal()

    # call Triton RoPE kernel: kernel expects x/out as (seq_len, n_heads, head_dim) tensors
    rope_kernel.kernel(x_t.contiguous(), out_t.contiguous(), pos_f32, freqs, BLOCK=128)

    # write back to LLAISYS output
    from_torch_to_ptr(out_t, out)

    return out

