# simple_server.py
import os
import multiprocessing as mp
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs

MODEL = os.environ.get("MODEL_PATH", "/root/Qwen3-VL-30B-A3B-Thinking")  # change via env MODEL_PATH

args = ServerArgs(
    model_path=MODEL,
    tokenizer_path=MODEL,
    host="0.0.0.0",
    port=10000,
    tp_size=1,                # 1 GPU
    trust_remote_code=True,
    disable_cuda_graph=True,  # skip CUDA-graph warmup
    # attention_backend="torch_native",
    # prefill_attention_backend="torch_native",
    # decode_attention_backend="torch_native",
    mem_fraction_static=0.5,
    max_running_requests=128,
    tool_call_parser="qwen",  # <-- set here 
    grammar_backend="xgrammar", # <-- set here 
)

# parity with slime defaults
os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
os.environ.setdefault("SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
os.environ.setdefault("SGLANG_MEMORY_SAVER_CUDA_GRAPH", "false")



# ````
# curl -sS -i http://127.0.0.1:10000/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -d '{
#     "model":"/root/Qwen3-VL-30B-A3B-Thinking",
#     "messages":[{"role":"user","content":"use a tool"}],
#     "tools":[{"type":"function","function":{
#       "name":"computer",
#       "description":"",
#       "parameters":{
#         "type":"object",
#         "properties":{
#           "action":{
#             "type":"string",
#             "enum":["screenshot","left_click","right_click","double_click","triple_click","middle_click","mouse_move","left_click_drag","type","key","scroll","wait","cursor_position","left_mouse_down","left_mouse_up","hold_key"]
#           },
#           "text":{"type":"string"}
#         },
#         "required":["action"]
#       }
#     }}],
#     "tool_choice":"auto"
#   }'
# ````


def tool_call_once(base: str, model_path: str, image_url: str):
    # OpenAI-compatible tool call with image
    data = {
        "model": model_path,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Only respond with a tool call. Scroll up to find the reviews."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "computer",
                    "description": "Basic GUI actions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["scroll", "screenshot"]},
                            "scroll_direction": {"type": "string", "enum": ["up", "down"]},
                            "scroll_amount": {"type": "integer", "minimum": 1},
                            "text": {"type": "string"}
                        },
                        "required": ["action"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 128,
    }
    with httpx.Client(timeout=None) as c:
        r = c.post(f"{base}/v1/chat/completions", json=data)
        r.raise_for_status()
        return r.json()

if __name__ == "__main__":
    wait_server_ready(BASE)  # blocks until the server is healthy
    # Optional: simple generate
    out = generate_once(BASE, MODEL, "Describe NVLink in one sentence.")
    print(out)
    # Tool call with image
    IMAGE_URL = os.environ.get(
        "IMAGE_URL",
        "https://theseus-model-traces.s3.amazonaws.com/slime/14a989a1-2b61-4172-893d-aa65d8f20fbf/turn_024.jpg",
    )
    tool_out = tool_call_once(BASE, MODEL, IMAGE_URL)
    print(tool_out)