from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType

from pathlib import Path
import safetensors
import numpy as _np
from ctypes import c_void_p
from ..tensor import Tensor
import numpy as _np
from ctypes import c_void_p
from ..tensor import Tensor
from ..libllaisys import DataType


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        """Load model weights from safetensors into memory.

        This constructor currently loads all safetensors arrays into
        `self.params` as numpy arrays. Later we will convert them to
        `llaisys.Tensor` objects and build the transformer blocks.
        """

        model_path = Path(model_path)
        self.device = device
        self.params = {}

        # iterate all .safetensors files (sorted for determinism)
        for file in sorted(model_path.glob("*.safetensors")):
            # safe_open returns a SafeTensorsFile-like object exposing keys and get_tensor
            data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name_ in data_.keys():
                try:
                    arr = data_.get_tensor(name_)
                except Exception:
                    # fallback: some versions expose indexing
                    arr = data_[name_]
                # store numpy array; conversion to llaisys.Tensor is deferred
                self.params[name_] = arr

        # basic metadata guess
        self.config = {}
        # attempt to infer vocabulary size / embedding dim from common keys
        for k, v in self.params.items():
            if k.endswith("embedding.weight") or k.endswith("tok_embeddings.weight") or k.endswith("wte.weight"):
                self.config.setdefault("vocab_size", v.shape[0])
                self.config.setdefault("d_model", v.shape[1])
                break

        # Convert numpy arrays to llaisys.Tensor and load into device memory (or host depending on device)
        self.tensors = {}
        for name, arr in list(self.params.items()):
            # ensure numpy array
            if not isinstance(arr, _np.ndarray):
                arr = _np.array(arr)
            # choose DataType
            if arr.dtype == _np.float32:
                dt = DataType.F32
            elif arr.dtype == _np.float16:
                dt = DataType.F16
            else:
                # try bfloat16 name, fallback to F32 for unknown types
                try:
                    if str(arr.dtype) == "bfloat16":
                        dt = DataType.BF16
                    else:
                        dt = DataType.F32
                except Exception:
                    dt = DataType.F32

            # create Tensor and load data
            try:
                arr_c = _np.ascontiguousarray(arr)
                t = Tensor(shape=arr_c.shape, dtype=dt, device=device)
                t.load(arr_c.ctypes.data_as(c_void_p))
                self.tensors[name] = t
            except Exception:
                # if any tensor fails to load, keep numpy version and continue
                self.tensors[name] = arr

        # Convert numpy arrays into llaisys.Tensor objects for runtime use
        self.tensors = {}

        def _np_to_dtype(npdtype):
            if npdtype == _np.float32:
                return DataType.F32
            if npdtype == _np.float16:
                return DataType.F16
            if npdtype == _np.bool_:
                return DataType.BOOL
            if npdtype == _np.int8:
                return DataType.I8
            if npdtype == _np.int32:
                return DataType.I32
            # bf16 may appear as dtype('bfloat16') in newer numpy
            if str(npdtype) == "bfloat16" or str(npdtype) == "<bf16":
                return DataType.BF16
            # fallback to F32
            return DataType.F32

        for name, arr in list(self.params.items()):
            if not isinstance(arr, _np.ndarray):
                continue
            dtype = _np_to_dtype(arr.dtype)
            # ensure contiguous
            arr_c = _np.ascontiguousarray(arr)
            shape = tuple(int(s) for s in arr_c.shape)
            try:
                t = Tensor(shape=shape, dtype=dtype, device=device)
                # load from host pointer
                ptr = c_void_p(arr_c.ctypes.data)
                t.load(ptr)
                self.tensors[name] = t
            except Exception:
                # keep numpy fallback if conversion fails
                self.tensors[name] = arr_c

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        """Placeholder generate implementation.

        The full transformer forward (using `llaisys.Ops`) is not
        implemented yet. For now raise an explicit error so the test
        harness can proceed after we implement the inference path.
        """

        raise NotImplementedError(
            "Qwen2.generate not implemented yet — transformer forward/decoding to be implemented using llaisys.Ops (Triton/ninetoothed)."
        )
