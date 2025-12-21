# 文件名: llaisys/models/chat_server.py (或您的 api.py)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import os
import json
import time
from typing import List, Dict
# [核心改动] 引入 multiprocessing
import multiprocessing as mp
# queue 在 multiprocessing 中有自己的版本
from multiprocessing import Queue

# --- 模型工作函数 (将在一个独立的子进程中运行) ---
# 这个函数包含了所有与模型和tokenizer相关的操作
def model_worker_func(
    model_path: str,
    input_ids: List[int],
    gen_params: Dict,
    result_queue: Queue
):
    """
    这是一个独立的函数，将在一个新的进程中被执行。
    它加载模型和tokenizer，执行一次生成，然后退出。
    """
    try:
        # 1. 在子进程内部导入所有相关库
        from .qwen2 import Qwen2
        from ..libllaisys import DeviceType
        from transformers import AutoTokenizer

        # print(f"[Worker PID: {os.getpid()}] Loading model and tokenizer...")
        
        # 2. 加载模型和Tokenizer
        # 注意：这里的 device 可以根据您的环境设置
        model = Qwen2(model_path, device=DeviceType.NVIDIA)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        
        # 3. 定义 token 回调函数
        def token_callback(token_id: int):
            result_queue.put(int(token_id))

        # 4. 执行模型生成
        # print(f"[Worker PID: {os.getpid()}] Starting generation...")
        model.generate(
            inputs=input_ids,
            token_callback=token_callback,
            **gen_params
        )

    except Exception as e:
        import traceback
        # 将错误信息放入队列，以便主进程知道
        error_msg = f"Worker process failed: {e}\n{traceback.format_exc()}"
        result_queue.put({"__error__": error_msg})
    finally:
        # 5. 放置结束标记，然后进程会自动退出
        # print(f"[Worker PID: {os.getpid()}] Generation finished. Worker exiting.")
        result_queue.put(None)


# --- FastAPI 应用 (主进程) ---
app = FastAPI(title="llaisys Chat-Completion Compat Server (Process-Isolated)")

# 主进程只管理对话历史和Web服务
CONVERSATION_HISTORY: List[Dict[str, str]] = []
MAX_HISTORY_TURNS = 10
history_lock = mp.Lock() # 使用 multiprocessing 的锁

# [重要] 主进程不加载模型，只记录路径
MODEL_PATH = os.environ.get("LLAISYS_MODEL_PATH", None)
if not MODEL_PATH:
    raise RuntimeError("LLAISYS_MODEL_PATH environment variable must be set.")

# 主进程只加载一次 Tokenizer 用于编码
_tokenizer = None
try:
    from transformers import AutoTokenizer
    print("Loading tokenizer in main process for encoding...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    print("Tokenizer loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer in main process: {e}")

def decode_ids(ids: List[int]) -> str:
    if _tokenizer is None:
        return "[Tokenizer not available for decoding]"
    return _tokenizer.decode(ids, skip_special_tokens=True)

@app.get("/health")
async def health():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    global CONVERSATION_HISTORY
    
    try:
        data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    # 1. 在主进程中准备输入
    messages: List[Dict[str, str]] = data.get("messages", [])
    with history_lock:
        if messages and messages[-1]['role'] == 'user': CONVERSATION_HISTORY.append(messages[-1])
        while len(CONVERSATION_HISTORY) > MAX_HISTORY_TURNS * 2: CONVERSATION_HISTORY.pop(0); CONVERSATION_HISTORY.pop(0)
        conversation_for_prompt = list(CONVERSATION_HISTORY)

    try:
        input_ids_tensor = _tokenizer.apply_chat_template(
            conversation=conversation_for_prompt, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
        input_ids = input_ids_tensor.tolist()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenization failed: {e}")

    gen_params = {
        "max_new_tokens": int(data.get("max_tokens", 256)),
        "temperature": float(data.get("temperature", 1.0)),
        "top_k": int(data.get("top_k", 0)),
        "top_p": float(data.get("top_p", 0.0)),
        "do_sample": data.get("do_sample", False),
    }

    stream = bool(data.get("stream", False))

    if not stream:
        raise HTTPException(status_code=400, detail="This server implementation only supports streaming mode for stability.")

    # 2. 创建队列和子进程
    result_queue = Queue()
    process = mp.Process(
        target=model_worker_func,
        args=(MODEL_PATH, input_ids, gen_params, result_queue)
    )
    process.start()

    def event_stream():
        final_generated_ids = []
        try:
            print("\n--- [Streaming] Waiting for model response from worker ---")
            acc_ids = []
            prev_text = ""
            while True:
                item = result_queue.get()
                if item is None:
                    print("\n----------------------------------------------------\n")
                    yield 'data: [DONE]\n\n'
                    break
                
                if isinstance(item, dict) and item.get("__error__"):
                    print(f"Error from worker process: {item['__error__']}")
                    yield 'data: ' + json.dumps({"error": "Model worker process failed."}) + '\n\n'
                    yield 'data: [DONE]\n\n'
                    break
                
                final_generated_ids.append(item)
                acc_ids.append(item)
                full_text = decode_ids(acc_ids)
                delta = full_text[len(prev_text):]
                prev_text = full_text
                if delta:
                    print(delta, end='', flush=True)
                    chunk = {"choices": [{"delta": {"content": delta}, "finish_reason": None}]}
                    yield 'data: ' + json.dumps(chunk) + '\n\n'
        finally:
            # 确保子进程被清理
            process.join(timeout=5)
            if process.is_alive():
                print("Warning: Worker process did not terminate gracefully. Terminating.")
                process.terminate()

            # 更新对话历史
            if final_generated_ids:
                response_text = decode_ids(final_generated_ids)
                with history_lock:
                    CONVERSATION_HISTORY.append({"role": "assistant", "content": response_text})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == '__main__':
    # [重要] 确保在Windows/MacOS上多进程的安全性
    mp.set_start_method("fork") # 在Linux上fork是默认且高效的
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)