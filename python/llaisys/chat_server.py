from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import os
import json
import threading
import queue
import time
from typing import List, Dict, Any

# lazy import of model to avoid heavy imports on import time
# 假设模型文件和设备类型定义在这些模块中
from .models.qwen2 import Qwen2
from .libllaisys import DeviceType

app = FastAPI(title="llaisys Chat-Completion Compat Server")

# Single-user model holder
class SingleModel:
    def __init__(self, model_path: str, device: DeviceType = None):
        dev = device
        if dev is None:
            try:
                dev = DeviceType.NVIDIA
            except Exception:
                dev = None
        self.model = Qwen2(model_path, device=dev)
        self.lock = threading.Lock()

MODEL_PATH = os.environ.get("LLAISYS_MODEL_PATH", None)
_model_holder: SingleModel = None
if MODEL_PATH:
    try:
        print(f"Loading model from LLAISYS_MODEL_PATH: {MODEL_PATH}")
        _model_holder = SingleModel(MODEL_PATH)
    except Exception as e:
        print(f"Warning: failed to load model at startup: {e}")

# --- 修正后的 Tokenizer 加载逻辑 ---
_tokenizer = None
if MODEL_PATH:
    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, 
            use_fast=True,
            trust_remote_code=True
        )
        print(f"Successfully loaded tokenizer using transformers.AutoTokenizer from {MODEL_PATH}")
    except Exception as e:
        _tokenizer = None
        print(f"Warning: failed to load tokenizer with AutoTokenizer: {e}")


def decode_ids(ids: List[int]) -> str:
    """解码token ID列表为字符串。"""
    if _tokenizer is None:
        raise RuntimeError("No tokenizer available to decode ids.")
    return _tokenizer.decode(ids, skip_special_tokens=True)


@app.get("/health")
async def health():
    """健康检查端点。"""
    return {"status": "ok", "model_loaded": _model_holder is not None}


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    """
    一个兼容 OpenAI ChatCompletion 格式的最小化端点。
    """
    if _model_holder is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set LLAISYS_MODEL_PATH and restart server.")

    try:
        data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    # (此处省略了请求解析部分的代码，与上一版相同)
    input_ids = data.get("input_ids")
    if input_ids is None:
        prompt = data.get("prompt") or data.get("messages")
        if prompt is not None:
            if isinstance(prompt, list):
                try:
                    prompt_text = "\n".join([m.get("content", "") for m in prompt if isinstance(m, dict) and "content" in m])
                except Exception:
                    prompt_text = str(prompt)
            else:
                prompt_text = str(prompt)
            if not _tokenizer:
                raise HTTPException(status_code=400, detail="No tokenizer available to encode 'prompt'.")
            try:
                formatted_prompt = _tokenizer.apply_chat_template(
                    conversation=[{"role": "user", "content": prompt_text}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt"
                )
                input_ids = formatted_prompt.tolist()[0]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Tokenization failed: {e}")
        else:
            raise HTTPException(status_code=400, detail="Request must include either 'input_ids' (list of ints) or 'prompt'/'messages' (text).")
    if not isinstance(input_ids, list) or not all(isinstance(x, int) for x in input_ids):
        raise HTTPException(status_code=400, detail="'input_ids' must be a list of integers.")
    max_tokens = int(data.get("max_tokens", 128))
    temperature = float(data.get("temperature", 1.0))
    top_k = int(data.get("top_k", 50))
    top_p = float(data.get("top_p", 1.0))
    stream = bool(data.get("stream", False))
    model = _model_holder.model

    if stream:
        q = queue.Queue()

        def token_cb(tid: int):
            q.put(tid)

        def run_generate():
            with _model_holder.lock:
                try:
                    do_sample = temperature != 1.0 or top_k > 1 or (0.0 < top_p < 1.0)
                    model.generate(input_ids, max_new_tokens=max_tokens, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, token_callback=token_cb)
                except Exception as e:
                    q.put({"__error__": str(e)})
                finally:
                    q.put(None)

        thread = threading.Thread(target=run_generate, daemon=True)
        thread.start()

        def event_stream():
            # 在流开始时打印标题
            print("\n--- [Streaming] Model Generated Content ---")
            acc_ids = []
            prev_text = ""
            while True:
                item = q.get()
                if item is None:
                    # 在流结束后换行，使终端显示更整洁
                    print("\n-----------------------------------------\n")
                    yield 'data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}) + '\n\n'
                    break
                if isinstance(item, dict) and item.get("__error__"):
                    yield 'data: ' + json.dumps({"error": item.get("__error__")}) + '\n\n'
                    break
                
                try:
                    acc_ids.append(int(item))
                    full_text = decode_ids(acc_ids)
                    delta = full_text[len(prev_text):]
                    prev_text = full_text
                    if delta:
                        # ===> 在这里打印流式内容的每个增量 <===
                        print(delta, end='', flush=True)
                        chunk = {"choices": [{"delta": {"content": delta}, "finish_reason": None}]}
                        yield 'data: ' + json.dumps(chunk) + '\n\n'
                except Exception:
                    chunk = {"choices": [{"delta": {"content": str(item)}, "finish_reason": None}]}
                    yield 'data: ' + json.dumps(chunk) + '\n\n'

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        with _model_holder.lock:
            do_sample = temperature != 1.0 or top_k > 1 or (0.0 < top_p < 1.0)
            out = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p)
        
        output_ids = out[len(input_ids):]
        content = decode_ids(output_ids)
        
        # ===> 在这里打印完整的非流式内容 <===
        print("\n--- [Non-Streaming] Model Generated Content ---")
        print(content)
        print("---------------------------------------------\n")
        
        resp = {
            "id": f"cmpl-llaisys-{time.time()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_PATH,
            "choices": [
                {
                    "index": 0, 
                    "message": {"role": "assistant", "content": content}, 
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(output_ids),
                "total_tokens": len(out)
            }
        }
        return JSONResponse(content=resp)