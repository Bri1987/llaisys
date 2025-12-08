import os
from .libllaisys import LIB_LLAISYS
from .libllaisys import triton as Triton
from .tensor import Tensor
from ctypes import c_float, c_int


os.environ["ENABLE_Triton"] = "True"
if os.environ.get("ENABLE_Triton") == "True":
    _CURRENT_LIB = Triton
    print("LLAISYS Ops: Using Triton for accelerated kernels.")
else:
    _CURRENT_LIB = LIB_LLAISYS
    print("LLAISYS Ops: Using default LIB_LLAISYS kernels.")


class Ops:
    @staticmethod
    def add(c: Tensor, a: Tensor, b: Tensor):
        _CURRENT_LIB.llaisysAdd(c.lib_tensor(), a.lib_tensor(), b.lib_tensor())

    @staticmethod
    def argmax(max_idx: Tensor, max_val: Tensor, vals: Tensor):
        _CURRENT_LIB.llaisysArgmax(max_idx.lib_tensor(), max_val.lib_tensor(), vals.lib_tensor())

    @staticmethod
    def embedding(out: Tensor, index: Tensor, weight: Tensor):
        _CURRENT_LIB.llaisysEmbedding(
            out.lib_tensor(), index.lib_tensor(), weight.lib_tensor()
        )

    @staticmethod
    def linear(out: Tensor, inp: Tensor, weight: Tensor, bias: Tensor):
        if bias is None:
            _CURRENT_LIB.llaisysLinear(out.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), None)
        else:
            _CURRENT_LIB.llaisysLinear(
                out.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), bias.lib_tensor()
            )

    @staticmethod
    def rearrange(out: Tensor, inp: Tensor):
        _CURRENT_LIB.llaisysRearrange(out.lib_tensor(), inp.lib_tensor())

    @staticmethod
    def rms_norm(out: Tensor, inp: Tensor, weight: Tensor, eps: float):
        _CURRENT_LIB.llaisysRmsNorm(
            out.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), c_float(eps)
        )

    @staticmethod
    def rope(out: Tensor, inp: Tensor, pos_ids: Tensor, theta: float):
        _CURRENT_LIB.llaisysROPE(
            out.lib_tensor(), inp.lib_tensor(), pos_ids.lib_tensor(), c_float(theta)
        )

    @staticmethod
    def self_attention(attn_val: Tensor, q: Tensor, k: Tensor, v: Tensor, scale: float, past_k: Tensor = None, past_v: Tensor = None):
        """Run self-attention; optionally supply `past_k` and `past_v` LLAISYS Tensors.

        This will attempt to call the Triton launcher with the extended
        arguments; if the underlying lib does not support the extended
        signature (C backend), it falls back to the original 5-arg call.
        """
        try:
            # Preferred: call the backend launcher with LLAISYS Tensor objects
            _CURRENT_LIB.llaisysSelfAttention(
                attn_val,
                q,
                k,
                v,
                c_float(scale),
                past_k,
                past_v,
            )
        except TypeError:
            # Fallback for backends that expect raw llaisysTensor handles
            _CURRENT_LIB.llaisysSelfAttention(
                attn_val.lib_tensor(),
                q.lib_tensor(),
                k.lib_tensor(),
                v.lib_tensor(),
                c_float(scale),
            )

    @staticmethod
    def swiglu(out: Tensor, gate: Tensor, up: Tensor):
        _CURRENT_LIB.llaisysSwiGLU(out.lib_tensor(), gate.lib_tensor(), up.lib_tensor())