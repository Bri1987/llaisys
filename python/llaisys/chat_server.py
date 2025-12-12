from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import os
import json
import threading
import queue
import time
from typing import List, Dict, Any, Union

# lazy import of model to avoid heavy imports on import time
# 假设模型文件和设备类型定义在这些模块中
from .models.qwen2 import Qwen2
from .libllaisys import DeviceType

app = FastAPI(title="llaisys Chat-Completion Compat Server")

# --- [新增] 对话历史管理 ---
# NOTE: 这是一个用于单用户演示的全局内存存储。
# 对于多用户应用，您需要一个更复杂的解决方案，例如一个将 session_id 映射到历史记录的字典。
CONVERSATION_HISTORY: List[Dict[str, str]] = []
# 设置一个最大历史轮次，防止上下文无限增长超出模型限制
# 一轮 = 1次用户提问 + 1次助手回答
MAX_HISTORY_TURNS = 10 
# 用于保护全局历史记录的线程锁
history_lock = threading.Lock()


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
    if _tokenizer is None:
        raise RuntimeError("No tokenizer available to decode ids.")
    return _tokenizer.decode(ids, skip_special_tokens=True)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model_holder is not None}


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    global CONVERSATION_HISTORY
    if _model_holder is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set LLAISYS_MODEL_PATH and restart server.")

    try:
        data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    # --- [核心修改] 上下文处理逻辑 ---
    
    # 1. 从请求中解析出对话消息
    messages: List[Dict[str, str]] = data.get("messages")
    if messages is None:
        # 兼容旧的 'prompt' 格式，并将其转换为 'messages' 格式
        prompt_text = data.get("prompt")
        if prompt_text:
            messages = [{"role": "user", "content": str(prompt_text)}]
        else:
            raise HTTPException(status_code=400, detail="Request must include 'messages' (list of dicts).")

    # 2. 更新并整理对话历史
    with history_lock:
        # 将用户的新消息添加到历史记录中
        # 通常，客户端会发送包含历史的完整消息列表，我们只取最新的用户消息
        # 为简化，这里假设`messages`只包含最新的用户提问
        # 一个更健壮的实现会检查`messages`的内容
        if messages[-1]['role'] == 'user':
            CONVERSATION_HISTORY.append(messages[-1])
        else:
            # 如果请求的最后一条不是 user，可能是一个格式错误，或者客户端自己管理历史
            # 为简单起见，我们直接使用客户端发来的完整历史
            CONVERSATION_HISTORY = messages

        # 3. 控制历史长度，防止溢出
        # 如果历史轮次超过限制，就从头开始删除旧的对话（一问一答）
        while len(CONVERSATION_HISTORY) > MAX_HISTORY_TURNS * 2:
            CONVERSATION_HISTORY.pop(0)
            CONVERSATION_HISTORY.pop(0)
        
        # 将要传递给模板的最终对话内容
        conversation_for_prompt = list(CONVERSATION_HISTORY)

    # 4. 使用模板和 Tokenizer 对完整的对话历史进行编码
    if not _tokenizer:
        raise HTTPException(status_code=500, detail="No tokenizer available to encode prompt.")
    try:
        # apply_chat_template 需要一个对话列表，而不是一个扁平的字符串
        input_ids_tensor = _tokenizer.apply_chat_template(
            conversation=conversation_for_prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        )
        input_ids = input_ids_tensor.tolist()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenization with chat template failed: {e}")

    # (采样参数解析部分保持不变)
    max_tokens = int(data.get("max_tokens", 256))
    temperature = float(data.get("temperature", 0))
    top_k = int(data.get("top_k", 1))
    top_p = float(data.get("top_p", 0.9))
    stream = bool(data.get("stream", False))
    model = _model_holder.model

    if stream:
        q = queue.Queue()
        # 用于在流结束后更新历史记录的容器
        final_generated_ids = []

        def token_cb(tid: int):
            q.put(tid)
            final_generated_ids.append(tid)

        def run_generate():
            with _model_holder.lock:
                try:
                    do_sample = temperature != 1.0 or top_k > 0 or (0.0 < top_p < 1.0)
                    model.generate(input_ids, max_new_tokens=max_tokens, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, token_callback=token_cb)
                except Exception as e:
                    q.put({"__error__": str(e)})
                finally:
                    q.put(None)

        thread = threading.Thread(target=run_generate, daemon=True)
        thread.start()

        def event_stream():
            # 在流结束后，将模型的完整回答添加到历史记录
            # nonlocal CONVERSATION_HISTORY 
            # (在Python 3.x的嵌套函数中修改外部非全局变量需要nonlocal，但这里我们直接在函数结束后修改全局变量)
            try:
                print("\n--- [Streaming] Model Generated Content ---")
                acc_ids = []
                prev_text = ""
                while True:
                    item = q.get() # 阻塞等待直到有新项
                    if item is None:
                        print("\n-----------------------------------------\n")
                        yield 'data: [DONE]\n\n'
                        break
                    
                    if isinstance(item, dict) and item.get("__error__"):
                        yield 'data: ' + json.dumps({"error": item.get("__error__")}) + '\n\n'
                        yield 'data: [DONE]\n\n'
                        break
                    
                    try:
                        acc_ids.append(int(item))
                        full_text = decode_ids(acc_ids)
                        delta = full_text[len(prev_text):]
                        prev_text = full_text
                        if delta:
                            print(delta, end='', flush=True)
                            chunk = {"choices": [{"delta": {"content": delta}, "finish_reason": None}]}
                            yield 'data: ' + json.dumps(chunk) + '\n\n'
                    except Exception:
                        continue # 忽略单个解码错误
            finally:
                # 无论流是正常结束还是异常中断，都尝试更新历史
                if final_generated_ids:
                    response_text = decode_ids(final_generated_ids)
                    with history_lock:
                        CONVERSATION_HISTORY.append({"role": "assistant", "content": response_text})
        
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        # 非流式生成
        with _model_holder.lock:
            do_sample = temperature != 1.0 or top_k > 0 or (0.0 < top_p < 1.0)
            out_ids = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p)
        
        output_ids = out_ids[len(input_ids):]
        content = decode_ids(output_ids)
        
        print("\n--- [Non-Streaming] Model Generated Content ---")
        print(content)
        print("---------------------------------------------\n")

        # 将模型的回答添加到历史记录中
        with history_lock:
            CONVERSATION_HISTORY.append({"role": "assistant", "content": content})
        
        resp = {
            "id": f"cmpl-llaisys-{time.time()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_PATH,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(input_ids), "completion_tokens": len(output_ids), "total_tokens": len(out_ids)}
        }
        return JSONResponse(content=resp)