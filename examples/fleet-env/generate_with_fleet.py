import asyncio
import json
import os
import re
from typing import Any, Dict, List
import aiohttp

import fleet
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from slime.utils.http_utils import post
from slime.utils.types import Sample
from .utils import (
    build_user_message_from_sample,
    list_mcp_tools,
    build_tools_param,
    manage_context,
    upload_image_to_s3,
)

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
    

    def _pp(msg: str):
        task_key = (getattr(sample, "metadata", {}) or {}).get("task_key", "unknown")
        print(f"[pid={os.getpid()}][task_key={task_key}] {msg}")


    # Create Fleet env and MCP session; list tools once
    env_key = getattr(args, "fleet_env", None) or getattr(args, "fleet_env_key", None)
    if not env_key:
        raise ValueError("--fleet-env is required for Fleet tool loop")

    _pp("making fleet env call..")
    env = await fleet.env.make_async(env_key=env_key, image_type="mcp", ttl_seconds=3600)
    mcp_url = f"{env.urls.root}api/v1/mcp"
    api_key = getattr(args, "fleet_api_key", None) or os.getenv("FLEET_API_KEY")
    
    try:
        async with streamablehttp_client(url=mcp_url) as streams:
            async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                await session.initialize()
                async with aiohttp.ClientSession() as http_session:
                    tools = await list_mcp_tools(session)
                    _pp(f"{tools=}")
                    tools_param = build_tools_param(tools)
                    user_message = build_user_message_from_sample(sample, tools)
                
                    _pp(f"{sample=}")
                    _pp(f"{user_message=}")
                    #print(f"{tools_param=}")

                    messages: List[Dict[str, Any]] = [user_message]

                    # Support both max_tool_turns and max_turns for configuration
                    max_turns = 40
                    tool_trace: List[Dict[str, Any]] = []
                    final_answer: str = ""

                    finished = False
                    done_summary: str = ""
                    for turn in range(max_turns):
                        def _ppt(msg: str):
                            _pp(f"[turn={turn}] {msg}")
                        retain_n_turns = getattr(args, "retain_n_turns", 3)
                        window_messages = manage_context(messages, retain_n_turns)
                        req = {
                            "model":"/root/Qwen2.5-VL-7B-Instruct",
                            "messages": window_messages,
                            "tools": tools_param,
                            "tool_choice": "auto",
                            "max_tokens": 512,
                        }
                        # model call.
                        mosresp = await post(chat_url, req)
                        _ppt(f"{mosresp=}")
                        choice = (mosresp.get("choices") or [{}])[0]
                        msg = choice.get("message") or {}
                        tool_calls = msg.get("tool_calls") or []
                        

                        # Fallback: synthesize a tool call if model emitted raw JSON in content
                        fallback = False
                        if not tool_calls:
                            content_text = msg.get("content")
                            if isinstance(content_text, str):
                                try:
                                    raw = json.loads(content_text)
                                    if isinstance(raw, dict):
                                        # Qwen often outputs direct action JSON. Map to 'computer' tool when available.
                                        has_computer = any(getattr(t, "name", "") == "computer" for t in tools)
                                        if has_computer and ("action" in raw or "scroll_direction" in raw or "text" in raw):
                                            tool_calls = [{
                                                "id": f"fallback_{turn}",
                                                "type": "function",
                                                "function": {"name": "computer", "arguments": json.dumps(raw)},
                                            }]
                                            
                                        elif "name" in raw and "arguments" in raw:
                                            tool_calls = [{
                                                "id": f"fallback_{turn}",
                                                "type": "function",
                                                "function": {"name": raw.get("name"), "arguments": json.dumps(raw.get("arguments", {}))},
                                            }]
                                        elif "summary" in raw:
                                            tool_calls = [{
                                                "id": f"fallback_{turn}",
                                                "type": "function",
                                                "function": {"name": "done", "arguments": json.dumps({"summary": raw.get("summary")})},
                                            }]
                                        if len(tool_calls) > 0:
                                            fallback = True
                                        
                                except Exception:
                                    pass

                        _ppt(f"{fallback=} {tool_calls=}")

                        # Append assistant message with tool_calls to maintain context
                        messages.append({"role": "assistant", "content": (msg.get("content") or ""), "tool_calls": tool_calls})

                        if not tool_calls:
                            _ppt("no tool_calls; continuing to next turn")
                            continue

                        # Execute tool calls and add role='tool' messages
                        # forcing a single tool call for now
                        result_str = ""
                        for tc in tool_calls[:1]:
                            name = (tc.get("function") or {}).get("name")
                            arguments = (tc.get("function") or {}).get("arguments") or "{}"
                            try:
                                parsed_args = json.loads(arguments) if isinstance(arguments, str) else arguments
                            except Exception:
                                parsed_args = {}
                            _ppt(f"calling tool {name} with args {parsed_args}")
                            # Intercept synthetic 'done' tool to finish the loop gracefully.
                            if name == "done":
                                done_summary = str(parsed_args.get("summary", "")).strip()
                                
                                sample.metadata["final_answer"] = done_summary
                                print(f"{sample.metadata['final_answer']=}")
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.get("id"),
                                    "content": done_summary,
                                })
                                tool_trace.append({
                                    "turn": turn,
                                    "name": name,
                                    "arguments": parsed_args,
                                    "result_text": done_summary,
                                })
                                finished = True
                                break
                            
                            elif name == 'wait':
                                duration = parsed_args.get("duration", 2)
                                _ppt(f"waiting for {duration} seconds")
                                await asyncio.sleep(duration)
                                continue
                            
                            result = await session.call_tool(name, parsed_args)
                            # Extract textual observation and optional screenshot
                            base64_data_url = None
                            
                            if getattr(result, "content", None):
                                for c in result.content:
                                    if hasattr(c, "text") and c.text and not result_str:
                                        # Prefer the first textual observation that is not a base64 blob
                                        if "base64_image" in c.text or c.text.startswith("data:image"):
                                            # skip setting result_str from this chunk
                                            pass
                                        else:
                                            result_str = c.text                            
                                            preview = (result_str or "")[:512]
                                            suffix = "…" if (result_str and len(result_str) > 512) else ""
                                            _ppt(f"tool result text: {preview}{suffix}")
                                        
                                    # Some MCP tools pack JSON in text; try to pull base64_image
                                    if hasattr(c, "text") and c.text and ("base64_image" in c.text):
                                        try:
                                            parsed = json.loads(c.text)
                                            if isinstance(parsed, dict) and isinstance(parsed.get("base64_image"), str):
                                                base64_data_url = parsed.get("base64_image")
                                        except Exception:
                                            pass
                            
                            print(f"{result_str=}")
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
                        
                            # Upload screenshot to S3 via Fleet API and add URL to trace
                            if isinstance(base64_data_url, str) and api_key:
                                # Extract pure base64 payload from possible data URL
                                b64_payload = base64_data_url.split(",", 1)[-1] if "," in base64_data_url else base64_data_url
                                task_key = (getattr(sample, "metadata", {}) or {}).get("task_key", "unknown")
                                s3_path = f"slime/{task_key}/turn_{turn:03d}.jpg"
                                uploaded_url = await upload_image_to_s3(http_session, b64_payload, s3_path, api_key)
                                if isinstance(uploaded_url, str):
                                    trace_entry["image_url"] = uploaded_url
                                    print(f"{trace_entry['image_url']=}")
                            elif isinstance(base64_data_url, str) and not api_key:
                                _ppt("FLEET_API_KEY not provided; skipping image upload")
                        
                            tool_trace.append(trace_entry)

                        if finished:
                            _ppt("received 'done' signal; stopping tool loop")
                            break
    finally:
        try:
            await env.close()
        except Exception:
            pass
    # Finalize sample
    _pp(f"finalizing sample with {len(tool_trace)} tool calls")
    if sample.metadata.get("final_answer") is None:
        sample.metadata["final_answer"] = result_str
    sample.response = json.dumps({"steps": tool_trace}, ensure_ascii=False)
    sample.status = Sample.Status.COMPLETED
    sample.metadata.setdefault("tool_trace", tool_trace)
    sample.metadata.setdefault("env_key", env_key)

    return sample


