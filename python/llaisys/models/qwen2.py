from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType

from pathlib import Path
import safetensors
import numpy as _np
from ctypes import c_void_p
from ..tensor import Tensor
from ..libllaisys import DataType
from .. import Ops
import numpy as _np
from ..libllaisys import DeviceType as _DeviceType


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
        temp_skipped = []

        # iterate all .safetensors files (sorted for determinism)
        for file in sorted(model_path.glob("*.safetensors")):
            # try numpy backend first
            data_np = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name_ in data_np.keys():
                try:
                    arr = data_np.get_tensor(name_)
                    self.params[name_] = arr
                    continue
                except Exception:
                    # numpy backend failed (likely bfloat16). Try PyTorch backend
                    try:
                        data_pt = safetensors.safe_open(file, framework="pt", device="cpu")
                        arr_pt = data_pt.get_tensor(name_)
                        # try to convert torch tensor to numpy without importing torch here
                        try:
                            arr_conv = arr_pt.cpu().numpy()
                            self.params[name_] = arr_conv
                        except Exception:
                            # keep the torch tensor object if we can't convert now
                            self.params[name_] = arr_pt
                        continue
                    except Exception:
                        temp_skipped.append(name_)
                        continue

        # basic metadata guess
        self.config = {}
        # attempt to infer vocabulary size / embedding dim from common keys
        for k, v in self.params.items():
            if k.endswith("embedding.weight") or k.endswith("tok_embeddings.weight") or k.endswith("wte.weight"):
                self.config.setdefault("vocab_size", v.shape[0])
                self.config.setdefault("d_model", v.shape[1])
                break

        # Convert numpy arrays into llaisys.Tensor objects for runtime use
        # Note: we avoid importing torch here. Some safetensors entries may
        # use bfloat16 which numpy doesn't understand; we will skip those
        # keys for now and record them in `self.skipped_keys` for later
        # handling (e.g., via a backend loader that supports bf16).
        self.tensors = {}
        self.skipped_keys = temp_skipped

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
            # If we have a torch.Tensor object (returned by safetensors pt backend)
            # try to convert it to numpy without importing torch explicitly.
            if not isinstance(arr, _np.ndarray):
                # try duck-typed conversion (torch.Tensor has cpu() and numpy())
                try:
                    if hasattr(arr, "cpu") and hasattr(arr, "numpy"):
                        try:
                            arr = arr.cpu().numpy()
                        except Exception:
                            # numpy conversion may fail for bfloat16; try loading
                            # directly from the torch tensor memory via data_ptr
                            if hasattr(arr, "data_ptr"):
                                # keep arr as the original torch tensor object; conversion
                                # below will detect and load via data_ptr
                                pass
                            else:
                                raise
                    else:
                        raise TypeError("unsupported tensor object")
                except Exception:
                    # unable to obtain a numpy view; we'll attempt pointer-based
                    # loading below if possible, otherwise skip
                    if not hasattr(arr, "data_ptr"):
                        self.skipped_keys.append(name)
                        continue
            # check dtype support
            try:
                dtype = _np_to_dtype(arr.dtype)
            except Exception:
                self.skipped_keys.append(name)
                continue

            # ensure contiguous
            try:
                arr_c = _np.ascontiguousarray(arr)
            except Exception:
                self.skipped_keys.append(name)
                continue

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
                continue

        # Second pass: for any parameters that are not numpy arrays but are
        # torch.Tensor objects (we stored them earlier), try pointer-based load
        for name, arr in list(self.params.items()):
            if name in self.tensors:
                continue
            # if it's a torch tensor like object with data_ptr, try load directly
            if hasattr(arr, "data_ptr"):
                try:
                    # map common torch dtypes by string
                    td = str(getattr(arr, 'dtype', ''))
                    if 'bfloat16' in td:
                        dt = DataType.BF16
                    elif 'float32' in td or 'float' in td:
                        dt = DataType.F32
                    elif 'float16' in td or 'half' in td:
                        dt = DataType.F16
                    else:
                        dt = DataType.F32

                    shape = tuple(int(s) for s in arr.shape)
                    t = Tensor(shape=shape, dtype=dt, device=device)
                    ptr = c_void_p(int(arr.data_ptr()))
                    t.load(ptr)
                    self.tensors[name] = t
                except Exception:
                    self.skipped_keys.append(name)
                    continue

        # Build convenient references for embedding and lm_head and per-layer params
        # Prefer common key names seen in the safetensors for Qwen2
        self.embed_weight = self.tensors.get("model.embed_tokens.weight") or self.tensors.get("model.embed_tokens.weight")
        self.lm_head_weight = self.tensors.get("lm_head.weight")
        self.lm_head_bias = self.tensors.get("lm_head.bias") if "lm_head.bias" in self.tensors else None

        # determine number of layers from keys or config
        num_layers = None
        if "num_hidden_layers" in self.config:
            num_layers = int(self.config["num_hidden_layers"])
        else:
            # infer from keys
            layer_idxs = set()
            for k in self.tensors.keys():
                if k.startswith("model.layers."):
                    try:
                        idx = int(k.split(".")[2])
                        layer_idxs.add(idx)
                    except Exception:
                        pass
            if layer_idxs:
                num_layers = max(layer_idxs) + 1
            else:
                num_layers = 0

        self.layers = []
        for i in range(num_layers):
            base = f"model.layers.{i}."
            layer = {
                "ln1": self.tensors.get(base + "input_layernorm.weight"),
                "q_w": self.tensors.get(base + "self_attn.q_proj.weight"),
                "q_b": self.tensors.get(base + "self_attn.q_proj.bias"),
                "k_w": self.tensors.get(base + "self_attn.k_proj.weight"),
                "k_b": self.tensors.get(base + "self_attn.k_proj.bias"),
                "v_w": self.tensors.get(base + "self_attn.v_proj.weight"),
                "v_b": self.tensors.get(base + "self_attn.v_proj.bias"),
                "o_w": self.tensors.get(base + "self_attn.o_proj.weight"),
                "o_b": self.tensors.get(base + "self_attn.o_proj.bias"),
                "ln2": self.tensors.get(base + "post_attention_layernorm.weight"),
                "mlp_down_w": self.tensors.get(base + "mlp.down_proj.weight"),
                "mlp_gate_w": self.tensors.get(base + "mlp.gate_proj.weight"),
                "mlp_up_w": self.tensors.get(base + "mlp.up_proj.weight"),
            }
            self.layers.append(layer)

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

        # Simplified greedy generation using only embedding -> lm_head linear -> argmax.
        # This is a minimal runnable path to validate Ops and device (GPU) pipeline.
        if max_new_tokens is None:
            max_new_tokens = 0

        # inputs: sequence of token ids
        seq = list(inputs)
        d_model = None
        if self.embed_weight is None:
            raise RuntimeError("embed weight not found")
        # infer d_model and vocab
        try:
            d_model = self.embed_weight.shape()[1]
        except Exception:
            # fallback to config
            d_model = int(self.config.get("d_model", 1536))

        # vocab size from lm_head weight
        if self.lm_head_weight is None:
            raise RuntimeError("lm_head weight not found")
        try:
            vocab = self.lm_head_weight.shape()[0]
        except Exception:
            vocab = int(self.config.get("vocab_size", 50257))

        # create index tensor for initial tokens (on device)
        n = len(seq)
        idx_np = _np.asarray(seq, dtype=_np.int64)
        idx_t = Tensor(shape=(n,), dtype=DataType.I64, device=self.device)
        idx_t.load(c_void_p(idx_np.ctypes.data))

        # embedding output
        emb_out = Tensor(shape=(n, d_model), dtype=self.embed_weight.dtype(), device=self.device)
        Ops.embedding(emb_out, idx_t, self.embed_weight)

        # prepare lm bias if missing
        if self.lm_head_bias is None:
            # create zero bias on model device
            zero_np = _np.zeros((vocab,), dtype=_np.float32)
            bias_t = Tensor(shape=(vocab,), dtype=DataType.F32, device=self.device)
            bias_t.load(c_void_p(zero_np.ctypes.data))
        else:
            bias_t = self.lm_head_bias

        outputs = []

        # greedy decode loop
        for step in range(max_new_tokens):
            # take last token embedding
            last = emb_out.view(1, d_model)

            # logits: [1, vocab]
            logits = Tensor(shape=(1, vocab), dtype=self.lm_head_weight.dtype(), device=self.device)
            Ops.linear(logits, last, self.lm_head_weight, bias_t)

            # argmax -> write back to CPU tensors
            max_idx = Tensor(shape=(1,), dtype=DataType.I64, device=DeviceType.CPU)
            max_val = Tensor(shape=(1,), dtype=DataType.F32, device=DeviceType.CPU)
            Ops.argmax(max_idx, max_val, logits)

            # read max_idx value from CPU memory
            from ctypes import cast, POINTER, c_int64

            ptr = max_idx.data_ptr()
            val = cast(ptr, POINTER(c_int64)).contents.value
            token_id = int(val)
            outputs.append(token_id)

            # append embedding for next token (single)
            next_idx_np = _np.asarray([token_id], dtype=_np.int64)
            next_idx = Tensor(shape=(1,), dtype=DataType.I64, device=self.device)
            next_idx.load(c_void_p(next_idx_np.ctypes.data))
            next_emb = Tensor(shape=(1, d_model), dtype=self.embed_weight.dtype(), device=self.device)
            Ops.embedding(next_emb, next_idx, self.embed_weight)

            # append row to emb_out: simple reallocation (inefficient but acceptable for test)
            # create new emb_out2 of shape (n+1, d_model) and copy
            try:
                new_emb = Tensor(shape=(n + 1, d_model), dtype=emb_out.dtype(), device=self.device)
                # copy existing emb_out into new_emb via Ops.add with zero trick or via linear rearrange not available
                # For simplicity, write existing embeddings by converting to torch and back using triton bridge (slow)
                # We'll use runtime copy via to_torch_tensor/from_torch_to_ptr implicitly via llaisysLinear by preparing a view
                # Fallback: just replace emb_out with last-only (works for autoregressive chain of single-step model)
                emb_out = next_emb
                n = 1
            except Exception:
                emb_out = next_emb
                n = 1

        return outputs
