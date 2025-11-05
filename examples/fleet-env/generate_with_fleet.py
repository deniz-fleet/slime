import asyncio
import json
from typing import Any, Dict, List

import fleet
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from slime.utils.http_utils import post
from slime.utils.types import Sample
from .utils import build_user_message_from_sample, list_mcp_tools, build_tools_param

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
                print(f"{tools=}")
                tools_param = build_tools_param(tools)
                print(f"{tools_param=}")

                messages: List[Dict[str, Any]] = [user_message]

                max_turns = getattr(args, "max_tool_turns", 4)
                tool_trace: List[Dict[str, Any]] = []

                for turn in range(max_turns):
                    req = {
                        "model": "/root/Qwen2.5-VL-7B-Instruct",
                        "messages": messages,
                        "tools": tools_param,
                        "tool_choice": "required",
                    }
                    print(f"{req=}")
                    # Hardcoded toy payload (no tools) per SGLang OpenAI chat completions docs
                    # req = {
                    #     "model": "/root/Qwen2.5-VL-7B-Instruct",
                    #     "messages": [
                    #         {"role": "user", "content": "Say this is a test"}
                    #     ],
                    #     "tools": tools_param,
                    # }
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
                    for tc in tool_calls:
                        name = (tc.get("function") or {}).get("name")
                        arguments = (tc.get("function") or {}).get("arguments") or "{}"
                        try:
                            parsed_args = json.loads(arguments) if isinstance(arguments, str) else arguments
                        except Exception:
                            parsed_args = {}

                        result = await session.call_tool(name, parsed_args)
                        # Convert result to a concise string for message content
                        result_str = None
                        try:
                            if getattr(result, "content", None):
                                for c in result.content:
                                    if hasattr(c, "text") and c.text:
                                        result_str = c.text
                                        break
                                result_str = result_str or json.dumps([c.model_dump() for c in result.content])
                            else:
                                result_str = str(result)
                        except Exception:
                            result_str = str(result)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "content": result_str,
                        })

                        tool_trace.append({
                            "turn": turn,
                            "name": name,
                            "arguments": parsed_args,
                            "result": result_str,
                        })
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


