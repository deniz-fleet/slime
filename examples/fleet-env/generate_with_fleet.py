import asyncio
import json
from typing import Any, Dict, List

import fleet
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from slime.utils.http_utils import post
from slime.utils.types import Sample
from .utils import build_user_message_from_sample, list_mcp_tools, build_tools_param, save_data_url_to_image_path

async def _list_mcp_tools(session: ClientSession):
    listed = await session.list_tools()
    return listed.tools


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    """Multi-turn Fleet tool-calling using SGLang Tool Parser (OpenAI-compatible API).

    - Uses tool_choice='required' to force tool calls (no system hints, no prompt schema).
    - Executes MCP tools and feeds role='tool' messages back to the model.
    - Supports multimodal via image_url content items for Qwen-VL.
    """
    chat_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
    user_message = build_user_message_from_sample(sample)
    print(f"{args.sglang_router_ip=}")
    print(f"{args.sglang_router_port=}")

    # Create Fleet env and MCP session; list tools once
    env_key = getattr(args, "fleet_env", None) or getattr(args, "fleet_env_key", None)
    if not env_key:
        raise ValueError("--fleet-env is required for Fleet tool loop")

    print(f"making fleet env call..")
    env = await fleet.env.make_async(env_key=env_key, image_type="mcp", ttl_seconds=3600)
    mcp_url = f"{env.urls.root}api/v1/mcp"

    try:
        async with streamablehttp_client(url=mcp_url) as streams:
            async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                await session.initialize()
                tools = await list_mcp_tools(session)
                #print(f"{tools=}")
                tools_param = build_tools_param(tools)
                #print(f"{tools_param=}")

                messages: List[Dict[str, Any]] = [user_message]

                max_turns = getattr(args, "max_tool_turns", 4)
                tool_trace: List[Dict[str, Any]] = []

                for turn in range(max_turns):
                    req = {
                        "model":"/root/Qwen2.5-VL-7B-Instruct",
                        "messages": messages,
                        "tools": tools_param,
                        "tool_choice": "required",
                    }
                    print(f"{req=}")
                    # model call.
                    resp = await post(chat_url, req)
                    print(f"{resp=}")
                    choice = (resp.get("choices") or [{}])[0]
                    msg = choice.get("message") or {}
                    tool_calls = msg.get("tool_calls") or []

                    # Append assistant message with tool_calls to maintain context
                    messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls})

                    if not tool_calls:
                        break

                    # Execute tool calls and add role='tool' messages
                    # forcing a single tool call for now
                    for tc in tool_calls[:1]:
                        name = (tc.get("function") or {}).get("name")
                        arguments = (tc.get("function") or {}).get("arguments") or "{}"
                        try:
                            parsed_args = json.loads(arguments) if isinstance(arguments, str) else arguments
                        except Exception:
                            parsed_args = {}
                        print(f"calling tool {name} with args {parsed_args}")
                        result = await session.call_tool(name, parsed_args)
                        # Avoid printing raw/binary blobs
                        # Extract textual observation and optional screenshot
                        result_str = None
                        base64_data_url = None
                        try:
                            if getattr(result, "content", None):
                                for c in result.content:
                                    if hasattr(c, "text") and c.text and not result_str:
                                        result_str = c.text
                                    # Some MCP tools pack JSON in text; try to pull base64_image
                                    if hasattr(c, "text") and c.text and ("base64_image" in c.text):
                                        try:
                                            parsed = json.loads(c.text)
                                            if isinstance(parsed, dict) and isinstance(parsed.get("base64_image"), str):
                                                base64_data_url = parsed.get("base64_image")
                                        except Exception:
                                            pass
                                result_str = result_str or json.dumps([c.model_dump() for c in result.content])
                            else:
                                result_str = str(result)
                        except Exception:
                            result_str = str(result)

                        # Log safe text preview only
                        try:
                            preview = (result_str or "")[:512]
                            suffix = "…" if (result_str and len(result_str) > 512) else ""
                            print(f"tool result text: {preview}{suffix}")
                        except Exception:
                            pass

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "content": result_str,
                        })

                        trace_entry = {
                            "turn": turn,
                            "name": name,
                            "arguments": parsed_args,
                            "result_text": result_str,
                        }

                        # If we have a screenshot, attach inline (data URL) to avoid FS coupling
                        if isinstance(base64_data_url, str) and base64_data_url.startswith("data:"):
                            try:
                                messages.append({
                                    "role": "user",
                                    "content": [
                                        {"type": "image_url", "image_url": {"url": base64_data_url}},
                                        {"type": "text", "text": "Observation screenshot"},
                                    ],
                                })
                                trace_entry["image_inline"] = True
                                print("tool screenshot: inline data URL attached")
                            except Exception:
                                pass

                        tool_trace.append(trace_entry)
    finally:
        try:
            await env.close()
        except Exception:
            pass
    print(f"{tool_trace=}")
    # Finalize sample
    sample.response = json.dumps({"steps": tool_trace}, ensure_ascii=False)
    sample.status = Sample.Status.COMPLETED
    sample.metadata.setdefault("tool_trace", tool_trace)
    sample.metadata.setdefault("env_key", env_key)

    return sample


