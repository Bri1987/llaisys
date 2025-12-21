from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType

from pathlib import Path
import safetensors
# 引入 ml_dtypes，它会扩展 Numpy 的能力，使其能够理解 bfloat16
import ml_dtypes
import numpy as _np
import json
from ctypes import c_void_p
from ..tensor import Tensor
from .. import Ops
from ..libllaisys import DeviceType as _DeviceType, MemcpyKind
from ..runtime import RuntimeAPI
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

        # 每一层的 KV 缓存：用于存放过去生成的 key/value 张量
        self.kv_cache = [ {"k": None, "v": None, "len": 0} for _ in range(num_layers) ]
        # decode模式下，我们默认 Triton 内核会直接将新的 K/V 写入（in-place）到缓存中

        # 为每一层准备的临时计算空间（scratch tensors），避免在每次前向传播时重复申请内存
        self.layer_scratch = [dict() for _ in range(num_layers)]

        # 用于解码的可重用张量：采用懒加载（lazily initialized）方式，
        # 避免在生成每个 token 时都重复创建 logits 和 argmax 这些输出张量
        self._reusable_logits = None
        self._reusable_argmax_idx = None
        self._reusable_argmax_val = None
        # 缓存的 lm_head 偏置（如果模型没有提供）
        self._cached_lm_head_bias = None


    def _get_scratch(self, layer_idx: int, name: str, shape, dtype):
        """Get or allocate a reusable scratch Tensor for `layer_idx`.

        - `shape` may be a tuple of ints or a Tensor-style shape()
        - `dtype` is passed to `Tensor` constructor (usually a DataType)
        """
        # normalize shape to tuple of ints
        try:
            shp = tuple(int(s) for s in shape)
        except Exception:
            try:
                shp = tuple(int(s) for s in shape())
            except Exception:
                shp = tuple(shape)

        cache = self.layer_scratch[layer_idx].get(name)
        if cache is not None:
            try:
                # cache.shape may be method or attribute
                existing = None
                if callable(getattr(cache, 'shape', None)):
                    existing = cache.shape()
                else:
                    existing = getattr(cache, 'shape', None)
                if existing is not None:
                    existing_tup = tuple(int(s) for s in existing)
                    if existing_tup == shp:
                        return cache
            except Exception:
                pass

        t = Tensor(shape=shp, dtype=dtype, device=self.device)
        self.layer_scratch[layer_idx][name] = t
        return t


    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        **kwargs  # 兼容旧的额外参数
    ):
        # 在每次调用 generate 时，必须重置 KV 缓存的已写入长度。
        for layer_cache in self.kv_cache:
            layer_cache['len'] = 0
            
        token_callback = None
        if kwargs:
            token_callback = kwargs.pop("token_callback", None)
            if kwargs:
                print(f"[generate info] Ignoring extra arguments: {kwargs}")

        if max_new_tokens is None:
            max_new_tokens = 1

        # 优先从 config 获取，如果失败，则从权重张量推断
        d_model = self.config.get("hidden_size")
        if d_model is None:
            d_model = self.embed_weight.shape()[1]
        
        vocab_size = self.config.get("vocab_size")
        if vocab_size is None:
            vocab_size = self.lm_head_weight.shape()[0]

        # --- 预先准备好 lm_head_bias
        if self.lm_head_bias is not None:
            bias_t = self.lm_head_bias
        else:
            if self._cached_lm_head_bias is None or tuple(int(s) for s in self._cached_lm_head_bias.shape()) != (vocab_size,):
                target_dtype = self.lm_head_weight.dtype()
                bt = Tensor(shape=(vocab_size,), dtype=target_dtype, device=self.device)
                # create zero numpy buffer
                if target_dtype == DataType.BF16 and ml_dtypes:
                    np_dtype = ml_dtypes.bfloat16
                elif target_dtype == DataType.F16:
                    np_dtype = _np.float16
                else:
                    np_dtype = _np.float32
                zero_np = _np.zeros((vocab_size,), dtype=np_dtype)
                bt.load(c_void_p(zero_np.ctypes.data))
                self._cached_lm_head_bias = bt
            bias_t = self._cached_lm_head_bias


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

            # Prepare per-layer KV caches for decode: allocate capacity = prompt_len + max_new_tokens
            total_capacity = prompt_len + max_new_tokens
            num_heads = self.config.get("num_attention_heads")
            kv_heads = self.config.get("num_key_value_heads") or num_heads
            head_dim = d_model // num_heads
            for li in range(len(self.layers)):
                # 如果还没分配，或已分配但容量不足，则分配/重分配 k/v cache
                existing_k = self.kv_cache[li].get('k')
                need_alloc = False
                if existing_k is None:
                    need_alloc = True
                else:
                    try:
                        if callable(getattr(existing_k, 'shape', None)):
                            existing_cap = int(existing_k.shape()[0])
                        else:
                            existing_cap = int(existing_k.shape[0])
                    except Exception:
                        existing_cap = 0
                    if existing_cap < total_capacity:
                        need_alloc = True

                if need_alloc:
                    k_cache = Tensor(shape=(total_capacity, kv_heads, head_dim), dtype=self.layers[li]['k_w'].dtype(), device=self.device)
                    v_cache = Tensor(shape=(total_capacity, kv_heads, head_dim), dtype=self.layers[li]['v_w'].dtype(), device=self.device)
                    self.kv_cache[li]['k'] = k_cache
                    self.kv_cache[li]['v'] = v_cache
            
            for i in range(len(self.layers)):
                x = self._block_forward(x, i, pos_ids)
        else: # 处理空 prompt 的情况
            if max_new_tokens > 0:
                 raise ValueError("Cannot generate from an empty prompt without a specified BOS token.")
            return []

        # --- 2. DECODING STAGE ---
        generated_tokens = []
        next_token_input = x.slice(0, prompt_len - 1, prompt_len)

        for step in range(max_new_tokens):
            final_hidden_state = next_token_input

            # reuse or lazily allocate logits and argmax output tensors
            if self._reusable_logits is None or tuple(int(s) for s in self._reusable_logits.shape()) != (1, vocab_size):
                self._reusable_logits = Tensor(shape=(1, vocab_size), dtype=self.lm_head_weight.dtype(), device=self.device)

            Ops.linear(self._reusable_logits, final_hidden_state, self.lm_head_weight, bias_t)

            new_token_id = None
            
            runtime = RuntimeAPI(self.device)
            import ctypes

            if self.should_sample(do_sample, temperature, top_k, top_p):
                # Copy logits from device -> host and perform sampling in numpy
                logits_dtype = self._reusable_logits.dtype()
                if logits_dtype == DataType.F32:
                    elem_size, HostArrayType = ctypes.sizeof(ctypes.c_float), ctypes.c_float
                else:
                    elem_size, HostArrayType = ctypes.sizeof(ctypes.c_uint16), ctypes.c_uint16

                host_ptr = runtime.malloc_host(elem_size * vocab_size)
                src_dev_ptr = ctypes.c_void_p(self._reusable_logits.data_ptr())
                runtime.memcpy_sync(host_ptr, src_dev_ptr, elem_size * vocab_size, MemcpyKind.D2H)

                if HostArrayType is ctypes.c_float:
                    arr = ctypes.cast(host_ptr, ctypes.POINTER(ctypes.c_float * vocab_size)).contents
                    logits_np = _np.ctypeslib.as_array(arr).astype(_np.float32)
                else:
                    arr = ctypes.cast(host_ptr, ctypes.POINTER(ctypes.c_uint16 * vocab_size)).contents
                    u16 = _np.ctypeslib.as_array(arr).astype(_np.uint16)
                    if logits_dtype == DataType.BF16: logits_np = (u16.astype(_np.uint32) << 16).view(_np.float32)
                    else: logits_np = u16.view(_np.float16).astype(_np.float32)

                if temperature != 1.0 and temperature > 0.0: 
                    logits_np /= float(temperature)
                
                logits_np -= _np.max(logits_np)
                probs = _np.exp(logits_np)
                probs /= _np.sum(probs)

                if top_k > 1 and top_k < vocab_size:
                    kth_prob = _np.partition(probs, -top_k)[-top_k]
                    probs[probs < kth_prob] = 0.0
                
                if top_p > 0.0 and top_p < 1.0:
                    sorted_indices = _np.argsort(probs)[::-1]
                    # 对排序后的概率计算累积和
                    sorted_probs = probs[sorted_indices]
                    cumulative_probs = _np.cumsum(sorted_probs)
                    
                    # 找到要移除的 token 的索引
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
                    # 向右移动一位，这样第一个超过阈值的 token 就不会被标记为移除了
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
                    sorted_indices_to_remove[0] = False # 确保第一个 token 永远不会因为移位被移除
                    
                    # 获取原始词表中需要被移除的 token 的索引
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0.0

                final_probs = probs / (_np.sum(probs) + 1e-9)
                new_token_id = int(_np.random.choice(vocab_size, p=final_probs))
                runtime.free_host(host_ptr)
            else:
                # Greedy path
                if self._reusable_argmax_idx is None: self._reusable_argmax_idx = Tensor(shape=(1,), dtype=DataType.I64, device=self.device)
                if self._reusable_argmax_val is None: self._reusable_argmax_val = Tensor(shape=(1,), dtype=DataType.F32, device=self.device)
                Ops.argmax(self._reusable_argmax_idx, self._reusable_argmax_val, self._reusable_logits)
                
                host_ptr = runtime.malloc_host(ctypes.sizeof(ctypes.c_int64))
                src_dev_ptr = ctypes.c_void_p(self._reusable_argmax_idx.data_ptr())
                runtime.memcpy_sync(host_ptr, src_dev_ptr, ctypes.sizeof(ctypes.c_int64), MemcpyKind.D2H)
                new_token_id = ctypes.cast(host_ptr, ctypes.POINTER(ctypes.c_int64)).contents.value
                runtime.free_host(host_ptr)
            
            generated_tokens.append(new_token_id)
            if token_callback is not None:
                try: token_callback(int(new_token_id))
                except Exception: pass
            
            if self.eos_token_id is not None and new_token_id == self.eos_token_id:
                break

            if step < max_new_tokens - 1:
                current_pos = prompt_len + step
                # Lazy-initialize reusable tensors for single-token decoding steps
                if not hasattr(self, '_decode_pos_tensor'):
                    self._decode_pos_np = _np.zeros((1,), dtype=_np.int64)
                    self._decode_pos_tensor = Tensor(shape=(1,), dtype=DataType.I64, device=self.device)
                    self._decode_token_np = _np.zeros((1,), dtype=_np.int64)
                    self._decode_token_tensor = Tensor(shape=(1,), dtype=DataType.I64, device=self.device)

                self._decode_pos_np[0] = current_pos
                self._decode_pos_tensor.load(c_void_p(self._decode_pos_np.ctypes.data))
                self._decode_token_np[0] = new_token_id
                self._decode_token_tensor.load(c_void_p(self._decode_token_np.ctypes.data))

                x = self._get_scratch(0, "embedding_temp", (1, d_model), self.embed_weight.dtype())
                Ops.embedding(x, self._decode_token_tensor, self.embed_weight)

                for i in range(len(self.layers)):
                    x = self._block_forward(x, i, self._decode_pos_tensor)
                
                next_token_input = x

        return prompt_tokens + generated_tokens

    def _block_forward(self, x: Tensor, layer_idx: int, pos_ids: Tensor) -> Tensor:
        """
        In-Place K Update & Zero-Copy V Update.
        Eliminates 'k_temp' scratch buffer entirely.
        """
        layer = self.layers[layer_idx]
        seq_len = x.shape()[0]
        d_model = x.shape()[1]

        # 1. RMSNorm
        ln1_w = layer.get("ln1")
        eps = float(self.config.get("rms_norm_eps", 1e-6))
        normed = Tensor(shape=(seq_len, d_model), dtype=x.dtype(), device=self.device)
        Ops.rms_norm(normed, x, ln1_w, eps)

        # 2. 准备 KV Cache 的写入位置 (Views)
        cache_k = self.kv_cache[layer_idx].get("k")
        cache_v = self.kv_cache[layer_idx].get("v")
        start_pos = int(self.kv_cache[layer_idx].get("len", 0))
        
        num_heads = self.config.get("num_attention_heads")
        kv_heads = self.config.get("num_key_value_heads")
        head_dim = d_model // num_heads

        if cache_k is not None:
            # 获取 Cache 的切片引用
            k_dest = cache_k.slice(0, start_pos, start_pos + seq_len)
            v_dest = cache_v.slice(0, start_pos, start_pos + seq_len)
        else:
            k_dest = self._get_scratch(layer_idx, "k_dest", (seq_len, kv_heads, head_dim), normed.dtype())
            v_dest = self._get_scratch(layer_idx, "v_dest", (seq_len, kv_heads, head_dim), normed.dtype())

        # 3. 计算 Q (Q 还是需要 Scratch，因为它不是 Cache 的一部分)
        q = self._get_scratch(layer_idx, "q", (seq_len, d_model), normed.dtype())
        Ops.linear(q, normed, layer.get("q_w"), layer.get("q_b"))

        # 4. 计算 V -> 【直接写入 Cache】 (保持之前的优化)
        # 依赖 Ops.linear 的新逻辑处理 3D Strides
        Ops.linear(v_dest, normed, layer.get("v_w"), layer.get("v_b"))

        # 5. 计算 K -> 【直接写入 Cache (RAW K)】
        # 优化点：不再写入 k_temp，而是直接把未旋转的 K 写入 Cache
        Ops.linear(k_dest, normed, layer.get("k_w"), layer.get("k_b"))

        # 6. RoPE
        q_view = q.view(seq_len, num_heads, head_dim)
        # k_dest 已经是 (Seq, KV_Heads, Head_Dim) 形状，无需 view

        q_rope = self._get_scratch(layer_idx, "q_rope", q_view.shape(), q_view.dtype())
        theta = float(self.config.get("rope_theta", 10000.0))
        
        # Q 的 RoPE (Out-of-place: q -> q_rope)
        Ops.rope(q_rope, q_view, pos_ids, theta)
        
        # K 的 RoPE -> 【In-Place 原地旋转】
        # 输入和输出都是 k_dest。Triton Kernel 读取 Cache 中的值，旋转后写回 Cache。
        # 这一步极其高效，没有额外的显存分配和搬运。
        Ops.rope(k_dest, k_dest, pos_ids, theta)

        # 7. 准备 Attention 参数
        past_k = None
        past_v = None

        if cache_k is not None:
            self.kv_cache[layer_idx]["len"] = start_pos + seq_len
            if start_pos > 0:
                past_k = cache_k.slice(0, 0, start_pos)
                past_v = cache_v.slice(0, 0, start_pos)
        
        # 8. Self-Attention
        attn_out = self._get_scratch(layer_idx, "attn_out", q_rope.shape(), q_rope.dtype())
        scale = 1.0 / math.sqrt(max(1, head_dim))
        
        Ops.self_attention(attn_out, q_rope, k_dest, v_dest, scale, past_k, past_v)

        # --- MLP ---
        attn_flat = attn_out.view(seq_len, d_model)
        o = self._get_scratch(layer_idx, "o", (seq_len, d_model), attn_flat.dtype())
        Ops.linear(o, attn_flat, layer.get("o_w"), layer.get("o_b"))

        res = self._get_scratch(layer_idx, "res", (seq_len, d_model), o.dtype())
        Ops.add(res, x, o)

        ln2_w = layer.get("ln2")
        norm2 = self._get_scratch(layer_idx, "norm2", (seq_len, d_model), res.dtype())
        Ops.rms_norm(norm2, res, ln2_w, eps)

        gate = self._get_scratch(layer_idx, "gate", (seq_len, layer.get("mlp_gate_w").shape()[0]), norm2.dtype())
        Ops.linear(gate, norm2, layer.get("mlp_gate_w"), None)
        up = self._get_scratch(layer_idx, "up", (seq_len, layer.get("mlp_up_w").shape()[0]), norm2.dtype())
        Ops.linear(up, norm2, layer.get("mlp_up_w"), None)

        out_mlp = self._get_scratch(layer_idx, "out_mlp", (seq_len, layer.get("mlp_gate_w").shape()[0]), gate.dtype())
        Ops.swiglu(out_mlp, gate, up)

        down = self._get_scratch(layer_idx, "down", (seq_len, d_model), out_mlp.dtype())
        Ops.linear(down, out_mlp, layer.get("mlp_down_w"), None)

        new_x = self._get_scratch(layer_idx, "new_x", (seq_len, d_model), down.dtype())
        Ops.add(new_x, res, down)

        return new_x

    @staticmethod
    def should_sample(do_sample: bool, temperature: float, top_k: int = None, top_p: float = None) -> bool:
        # 1. 若调用方显式关闭采样，直接返回 False
        if not bool(do_sample):
            return False

        # 2. temperature == 0 明确表示贪婪
        if temperature is not None and float(temperature) == 0.0:
            return False

        # 3. top_k == 1 等价于 greedy
        if top_k is not None and int(top_k) == 1:
            return False

        # 4. top_p == 0.0 也被视为不进行 nucleus 截断（退化为 greedy）
        if top_p is not None and float(top_p) == 0.0:
            return False

        # 5. 若 temperature 未设置（None），则尊重 do_sample（上面已检查）
        # 默认按 do_sample 处理
        return True