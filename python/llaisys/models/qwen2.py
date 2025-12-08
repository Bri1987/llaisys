from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType

from pathlib import Path
import safetensors
# 【第1步：确保导入】
# 引入 ml_dtypes，它会扩展 Numpy 的能力，使其能够理解 bfloat16。
# 我们使用 try-except 来避免在没有安装它的环境中崩溃。
try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None
import numpy as _np
import json
from ctypes import c_void_p
from ..tensor import Tensor
from .. import Ops
from ..libllaisys import DeviceType as _DeviceType
import math
from ..libllaisys.triton import setup_kernels as _sk


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        self.device = device
        self.config = {}
        cfg_path = model_path / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, 'r') as f:
                    cfg = json.load(f)
                for k in ["num_attention_heads", "num_key_value_heads", "hidden_size", "vocab_size", "torch_dtype", "rms_norm_eps", "rope_theta", "num_hidden_layers", "eos_token_id"]:
                    if k in cfg:
                        self.config[k] = cfg[k]
            except Exception:
                pass
        
        self.eos_token_id = self.config.get("eos_token_id")
        self.params = {}
        self.skipped_keys = []

        import os
        for file in sorted(model_path.glob("*.safetensors")):
            try:
                # 使用 numpy 后端打开，ml_dtypes 会在这里发挥作用
                data_np = safetensors.safe_open(file, framework="numpy", device="cpu")
            except Exception as e:
                print(f"Warning: Could not open {file}. Error: {e}")
                continue

            for name_ in data_np.keys():
                try:
                    # 直接获取张量，它现在可能是 bfloat16 类型的 numpy 数组
                    arr = data_np.get_tensor(name_)
                    self.params[name_] = arr
                except Exception as e:
                    print(f"Warning: Failed to load tensor '{name_}' from {file}. Error: {e}")
                    self.skipped_keys.append(name_)
        
        # 【第2步：修改这里】
        # _np_to_dtype 函数现在可以正确地将 numpy 的 bfloat16 映射到 llaisys 的 BF16
        def _np_to_dtype(npdtype):
            if npdtype == _np.float32:
                return DataType.F32
            if npdtype == _np.float16:
                return DataType.F16
            if npdtype == _np.int32:
                return DataType.I32
            # 通过 ml_dtypes 库来识别 bfloat16
            if ml_dtypes and npdtype == ml_dtypes.bfloat16:
                return DataType.BF16
            # 兼容通过字符串名称判断的方式
            if str(npdtype) in ("bfloat16", "<bf16"):
                return DataType.BF16
            # 如果遇到其他未处理的类型，可以添加警告
            print(f"Warning: Unsupported numpy dtype '{npdtype}' encountered. Defaulting to F32.")
            return DataType.F32

        self.tensors = {}
        for name, arr in self.params.items():
            if not isinstance(arr, _np.ndarray):
                self.skipped_keys.append(name)
                continue
            
            try:
                dtype = _np_to_dtype(arr.dtype)
                arr_c = _np.ascontiguousarray(arr)
                shape = tuple(int(s) for s in arr_c.shape)
                
                t = Tensor(shape=shape, dtype=dtype, device=device)
                ptr = c_void_p(arr_c.ctypes.data)
                t.load(ptr)
                self.tensors[name] = t
            except Exception as e:
                print(f"Error: Failed to convert numpy array '{name}' to llaisys.Tensor. Error: {e}")
                self.skipped_keys.append(name)

        # ... 后续的所有代码（构建层、fail-fast检查等）都保持不变 ...

        # Build convenient references
        embed_keys = ["model.embed_tokens.weight", "embed_tokens.weight"]
        self.embed_weight = None
        for k in embed_keys:
            if k in self.tensors:
                self.embed_weight = self.tensors.get(k)
                break

        lm_head_keys = ["lm_head.weight", "model.lm_head.weight"]
        self.lm_head_weight = None
        for k in lm_head_keys:
            if k in self.tensors:
                self.lm_head_weight = self.tensors.get(k)
                break

        self.lm_head_bias = self.tensors.get("lm_head.bias") or self.tensors.get("model.lm_head.bias")

        # Fail-fast check
        if self.embed_weight is None or self.lm_head_weight is None:
            missing_keys = []
            if self.embed_weight is None: missing_keys.append("embedding weight")
            if self.lm_head_weight is None: missing_keys.append("lm_head weight")
            
            skipped_info = ", ".join(sorted(set(self.skipped_keys)))
            raise RuntimeError(
                f"Required model parameters missing: {', '.join(missing_keys)}. "
                f"This might be due to an unsupported dtype. Skipped keys: [{skipped_info}]"
            )

        # Determine number of layers
        num_layers = self.config.get("num_hidden_layers", 0)
        if num_layers == 0:
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

        # KV cache storage per layer: will hold LLAISYS Tensor objects for past k/v
        self.kv_cache = [ {"k": None, "v": None, "len": 0} for _ in range(num_layers) ]
        # By default we assume the Triton kernel will write new K/V into
        # an existing past buffer in-place during decode. When True, the
        # Python-side code will NOT perform an extra `torch.cat` append
        # to avoid double-appending the same key/value vectors.
        # Set to False to let Python manage concatenation instead.
        # self.kv_append_via_kernel = True


    def _log_tensor(self, name: str, t: Tensor, topk: int = 5):
        """Best-effort logging for a LLAISYS `Tensor`.

        Converts to a device `torch.Tensor` via the triton bridge when
        possible and prints shape, dtype, mean/std and top-k values for
        1-D/2-D tensors. Wrapped in try/except to avoid impacting runtime.
        """
        try:
            # Lightweight, torch-free logging: print shape and dtype if available.
            shp = None
            dtype = None
            try:
                if hasattr(t, 'shape') and callable(getattr(t, 'shape')):
                    shp = tuple(int(s) for s in t.shape())
                elif hasattr(t, 'shape'):
                    shp = tuple(int(s) for s in t.shape)
            except Exception:
                shp = None
            try:
                dtype = t.dtype()
            except Exception:
                dtype = None
            print(f"[debug.tensor] {name} shape={shp} dtype={dtype}")
        except Exception:
            # never crash due to logging
            pass

    def validate_parameters(self, verbose: bool = False):
        """检查并返回已加载参数的摘要。

        返回一个字典，包含每个参数的存在性、类型、shape（若可得）和 dtype（若可得），
        以及一个 issues 列表用于记录缺失或异常的键。

        用法示例：
            q = Qwen2(model_path)
            report = q.validate_parameters(verbose=True)
        """
        summary = {}
        issues = []

        def _safe_shape_and_dtype(obj):
            # try numpy.ndarray
            try:
                import numpy as _np
                if isinstance(obj, _np.ndarray):
                    return (tuple(int(s) for s in obj.shape), str(obj.dtype))
            except Exception:
                pass

            # llaisys.Tensor-like: try .shape() then .shape attr
            try:
                if hasattr(obj, 'shape') and callable(getattr(obj, 'shape')):
                    shp = tuple(int(s) for s in obj.shape())
                elif hasattr(obj, 'shape'):
                    shp = tuple(int(s) for s in obj.shape)
                else:
                    shp = None
            except Exception:
                shp = None

            # dtype probing
            try:
                if hasattr(obj, 'dtype') and callable(getattr(obj, 'dtype')):
                    dt = obj.dtype()
                elif hasattr(obj, 'dtype'):
                    dt = obj.dtype
                else:
                    dt = None
            except Exception:
                dt = None

            # normalize dtype string
            try:
                if dt is not None:
                    dt = str(dt)
            except Exception:
                dt = None

            return (shp, dt)

        for k, v in sorted(self.params.items()):
            info = {"present_in_params": True}
            shp, dt = _safe_shape_and_dtype(v)
            info.update({"shape_in_params": shp, "dtype_in_params": dt})
            # also check if converted to self.tensors
            in_t = k in self.tensors
            info["present_in_tensors"] = in_t
            if in_t:
                shp2, dt2 = _safe_shape_and_dtype(self.tensors[k])
                info.update({"shape_in_tensors": shp2, "dtype_in_tensors": dt2})
            else:
                info.update({"shape_in_tensors": None, "dtype_in_tensors": None})
            summary[k] = info
            # record potential issue
            if not in_t:
                issues.append(f"skipped_or_not_converted: {k}")

        # check required keys
        required = ["model.embed_tokens.weight", "lm_head.weight"]
        for rk in required:
            if rk not in self.tensors:
                # check alternative keys too
                alt_found = False
                for alt in ("embed_tokens.weight", "wte.weight", "tok_embeddings.weight", "model.lm_head.weight", "lm_head.decoder.weight"):
                    if alt in self.tensors:
                        alt_found = True
                        break
                if not alt_found:
                    issues.append(f"required_missing: {rk}")

        report = {"summary": summary, "issues": issues}
        if verbose:
            import json
            print(json.dumps({k: {"shape": v.get("shape_in_tensors") or v.get("shape_in_params"), "dtype": v.get("dtype_in_tensors") or v.get("dtype_in_params"), "in_tensors": v.get("present_in_tensors")} for k, v in summary.items()}, indent=2))
            if issues:
                print("Issues:\n", "\n".join(issues))

        return report

    # ... (文件顶部的 import 和 __init__ 方法保持不变) ...
# ... 您可以删除 __init__ 方法中的 self.kv_append_via_kernel = True 这一行，因为它不再需要了 ...

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        **kwargs  # <--- 【核心修改】添加这部分
    ):
        """
        [最终版] 使用KV缓存生成序列。
        此版本修复了 lm_head_bias 的处理和模型维度的健壮性获取。
        """
        if kwargs:
            print(f"[generate info] Ignoring extra arguments: {kwargs}")
            
        if max_new_tokens is None:
            max_new_tokens = 1

        # --- [FIX 2] 健壮地获取模型维度 ---
        # 优先从 config 获取，如果失败，则从权重张量推断
        d_model = self.config.get("hidden_size")
        if d_model is None:
            d_model = self.embed_weight.shape()[1]
        
        vocab_size = self.config.get("vocab_size")
        if vocab_size is None:
            vocab_size = self.lm_head_weight.shape()[0]

        # --- [FIX 1] 预先准备好 lm_head_bias ---
        # 确保传递给 Ops.linear 的 bias 永远是一个有效的 Tensor，而不是 None
        bias_t = self.lm_head_bias
        if bias_t is None:
            target_dtype = self.lm_head_weight.dtype()
            bias_t = Tensor(shape=(vocab_size,), dtype=target_dtype, device=self.device)
            # 创建一个全零的 numpy 数组并加载进去
            if target_dtype == DataType.BF16 and ml_dtypes:
                np_dtype = ml_dtypes.bfloat16
            elif target_dtype == DataType.F16:
                np_dtype = _np.float16
            else: # 默认为 F32
                np_dtype = _np.float32
            zero_np = _np.zeros((vocab_size,), dtype=np_dtype)
            bias_t.load(c_void_p(zero_np.ctypes.data))


        # --- 1. PREFILL STAGE ---
        prompt_tokens = list(inputs)
        prompt_len = len(prompt_tokens)

        if prompt_len > 0:
            prompt_ids_np = _np.asarray(prompt_tokens, dtype=_np.int64)
            prompt_ids = Tensor(shape=(prompt_len,), dtype=DataType.I64, device=self.device)
            prompt_ids.load(c_void_p(prompt_ids_np.ctypes.data))

            pos_ids_np = _np.arange(prompt_len, dtype=_np.int64)
            pos_ids = Tensor(shape=(prompt_len,), dtype=DataType.I64, device=self.device)
            pos_ids.load(c_void_p(pos_ids_np.ctypes.data))

            x = Tensor(shape=(prompt_len, d_model), dtype=self.embed_weight.dtype(), device=self.device)
            Ops.embedding(x, prompt_ids, self.embed_weight)

            for i in range(len(self.layers)):
                x = self._block_forward(x, i, pos_ids)
        else: # 处理空 prompt 的情况
            # 对于空输入，我们需要一个起始的 hidden_state，通常是全零
            # 为了简单起见，我们假设输入至少有一个 token (e.g., BOS token)
            # 在实际应用中这里可能需要更复杂的处理
            if max_new_tokens > 0:
                 raise ValueError("Cannot generate from an empty prompt without a specified BOS token.")
            return []

        # --- 2. DECODING STAGE ---
        generated_tokens = []
        next_token_input = x.slice(0, prompt_len - 1, prompt_len)

        for step in range(max_new_tokens):
            final_hidden_state = next_token_input

            logits = Tensor(shape=(1, vocab_size), dtype=self.lm_head_weight.dtype(), device=self.device)
            Ops.linear(logits, final_hidden_state, self.lm_head_weight, bias_t) # 现在 bias_t 永远有效

            max_idx = Tensor(shape=(1,), dtype=DataType.I64, device=DeviceType.CPU)
            max_val = Tensor(shape=(1,), dtype=DataType.F32, device=DeviceType.CPU)
            Ops.argmax(max_idx, max_val, logits)
            
            from ctypes import cast, POINTER, c_int64
            ptr = max_idx.data_ptr()
            new_token_id = cast(ptr, POINTER(c_int64)).contents.value
            
            generated_tokens.append(new_token_id)
            
            if self.eos_token_id is not None and new_token_id == self.eos_token_id:
                print(f"[generate info] EOS token ({self.eos_token_id}) generated. Stopping.")
                break # 提前终止循环

            if step < max_new_tokens - 1:
                current_pos = prompt_len + step
                pos_id_np = _np.asarray([current_pos], dtype=_np.int64)
                pos_id_tensor = Tensor(shape=(1,), dtype=DataType.I64, device=self.device)
                pos_id_tensor.load(c_void_p(pos_id_np.ctypes.data))

                new_token_id_np = _np.asarray([new_token_id], dtype=_np.int64)
                new_token_tensor = Tensor(shape=(1,), dtype=DataType.I64, device=self.device)
                new_token_tensor.load(c_void_p(new_token_id_np.ctypes.data))

                x = Tensor(shape=(1, d_model), dtype=self.embed_weight.dtype(), device=self.device)
                Ops.embedding(x, new_token_tensor, self.embed_weight)

                for i in range(len(self.layers)):
                    x = self._block_forward(x, i, pos_id_tensor)
                
                next_token_input = x

        return prompt_tokens + generated_tokens

    def _block_forward(self, x: Tensor, layer_idx: int, pos_ids: Tensor) -> Tensor:
        """
        Performs a forward pass for a single transformer block.
        It now explicitly concatenates the KV cache for the decode stage.
        """
        layer = self.layers[layer_idx]
        seq_len = x.shape()[0]
        d_model = x.shape()[1]

        # RMSNorm 1 & Projections
        ln1_w = layer.get("ln1")
        eps = float(self.config.get("rms_norm_eps", 1e-6))
        normed = Tensor(shape=(seq_len, d_model), dtype=x.dtype(), device=self.device)
        Ops.rms_norm(normed, x, ln1_w, eps)

        q = Tensor(shape=(seq_len, d_model), dtype=normed.dtype(), device=self.device)
        Ops.linear(q, normed, layer.get("q_w"), layer.get("q_b"))
        k = Tensor(shape=(seq_len, layer.get("k_w").shape()[0]), dtype=normed.dtype(), device=self.device)
        Ops.linear(k, normed, layer.get("k_w"), layer.get("k_b"))
        v = Tensor(shape=(seq_len, layer.get("v_w").shape()[0]), dtype=normed.dtype(), device=self.device)
        Ops.linear(v, normed, layer.get("v_w"), layer.get("v_b"))

        # Reshape and RoPE
        num_heads = self.config.get("num_attention_heads")
        kv_heads = self.config.get("num_key_value_heads")
        head_dim = d_model // num_heads

        q_view = q.view(seq_len, num_heads, head_dim)
        k_view = k.view(seq_len, kv_heads, head_dim)
        v_view = v.view(seq_len, kv_heads, head_dim)

        q_rope = Tensor(shape=q_view.shape(), dtype=q_view.dtype(), device=self.device)
        k_rope = Tensor(shape=k_view.shape(), dtype=k_view.dtype(), device=self.device)
        theta = float(self.config.get("rope_theta", 10000.0))
        Ops.rope(q_rope, q_view, pos_ids, theta)
        Ops.rope(k_rope, k_view, pos_ids, theta)
        
        # --- 【核心修正】显式的 KV Cache 管理 ---
        past_k = self.kv_cache[layer_idx].get("k")
        past_v = self.kv_cache[layer_idx].get("v")

        if past_k is None:  # Prefill stage (seq_len > 1)
            k_to_use = k_rope
            v_to_use = v_view
        else:  # Decode stage (seq_len = 1)
            try:
                tk_past = _sk.to_torch_tensor(past_k)
                tv_past = _sk.to_torch_tensor(past_v)
                tk_new = _sk.to_torch_tensor(k_rope)
                tv_new = _sk.to_torch_tensor(v_view)
                
                # 在 seq_len 维度 (dim=0) 上拼接
                k_cat_torch = _sk.torch.cat([tk_past, tk_new], dim=0)
                v_cat_torch = _sk.torch.cat([tv_past, tv_new], dim=0)
                
                k_to_use = Tensor(shape=k_cat_torch.shape, dtype=past_k.dtype(), device=self.device)
                v_to_use = Tensor(shape=v_cat_torch.shape, dtype=past_v.dtype(), device=self.device)
                _sk.from_torch_to_ptr(k_cat_torch, k_to_use)
                _sk.from_torch_to_ptr(v_cat_torch, v_to_use)
            except Exception as e:
                raise RuntimeError(f"Failed to concatenate KV cache at layer {layer_idx} via Triton bridge: {e}")

        # 更新缓存为拼接后的完整 K/V
        self.kv_cache[layer_idx]["k"] = k_to_use
        self.kv_cache[layer_idx]["v"] = v_to_use

        # Self-Attention
        attn_out = Tensor(shape=q_rope.shape(), dtype=q_rope.dtype(), device=self.device)
        scale = 1.0 / math.sqrt(max(1, head_dim))
        # 总是传入完整的 K/V
        Ops.self_attention(attn_out, q_rope, k_to_use, v_to_use, scale)

        # --- 后续 MLP 和残差连接部分保持不变 ---
        attn_flat = attn_out.view(seq_len, d_model)
        o = Tensor(shape=(seq_len, d_model), dtype=attn_flat.dtype(), device=self.device)
        Ops.linear(o, attn_flat, layer.get("o_w"), layer.get("o_b"))

        res = Tensor(shape=(seq_len, d_model), dtype=o.dtype(), device=self.device)
        Ops.add(res, x, o)

        ln2_w = layer.get("ln2")
        norm2 = Tensor(shape=(seq_len, d_model), dtype=res.dtype(), device=self.device)
        Ops.rms_norm(norm2, res, ln2_w, eps)

        gate = Tensor(shape=(seq_len, layer.get("mlp_gate_w").shape()[0]), dtype=norm2.dtype(), device=self.device)
        Ops.linear(gate, norm2, layer.get("mlp_gate_w"), None)
        up = Tensor(shape=(seq_len, layer.get("mlp_up_w").shape()[0]), dtype=norm2.dtype(), device=self.device)
        Ops.linear(up, norm2, layer.get("mlp_up_w"), None)

        out_mlp = Tensor(shape=(seq_len, layer.get("mlp_gate_w").shape()[0]), dtype=gate.dtype(), device=self.device)
        Ops.swiglu(out_mlp, gate, up)

        down = Tensor(shape=(seq_len, d_model), dtype=out_mlp.dtype(), device=self.device)
        Ops.linear(down, out_mlp, layer.get("mlp_down_w"), None)

        new_x = Tensor(shape=(seq_len, d_model), dtype=down.dtype(), device=self.device)
        Ops.add(new_x, res, down)

        return new_x