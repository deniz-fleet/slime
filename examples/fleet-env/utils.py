import base64
from pathlib import Path
import json
import mimetypes
import os
import re
from typing import Any, Dict, List, Union, Optional

from pydantic import BaseModel


COMPUTER_TOOL_USAGE_GUIDE = """## Computer Control Tool Usage Guide

### Critical Wait Rule (hard stop)

- If the screen is black/blank or shows a spinner/progress, do NOT call `done`.
- First do {"action": "wait", "duration": 2} then {"action": "screenshot"} and reassess.
- If still loading or black after two cycles, repeat once more with {"duration": 5}.
- You must not call `done` for any "still loading/black screen" reason.

### Screen Information

- Resolution: 1366x768 pixels
- Coordinate system: (0,0) is top-left, (1365,767) is bottom-right
- All coordinates must be integers within valid ranges

### First-Turn Protocol (hard rule)

1) Always start with {"action": "screenshot"}.
2) If anything is still loading (spinner, progress, or blank areas), do NOT call `done`. Instead:
   - {"action": "wait", "duration": 2}, then {"action": "screenshot"} and reassess.
3) Only after the page is stable, perform the next action (click/type/scroll/drag).
4) Never call `done` on the first turn. `done` requires at least one valid non-screenshot action that advances the task and a stable follow-up screenshot.

### Action Parameters (strict)

- Mouse actions (require coordinate [x, y], x in 0–1365 and y in 0–767):
  - `left_click`, `right_click`, `double_click`, `triple_click`, `middle_click`, `mouse_move`
  - Required: "coordinate": [x, y]
  - Forbidden: "start_coordinate", "text", "duration", "scroll_direction", "scroll_amount"

- Drag operations (require both):
  - `left_click_drag`
  - Required: "start_coordinate": [x, y] AND "coordinate": [x, y]
  - Forbidden: "text", "duration", "scroll_direction", "scroll_amount"

- Keyboard actions:
  - `type`: Required "text": your words or phrase; Forbidden: "coordinate", "start_coordinate", "duration"
  - `key`: Required "text": one special key from [
    "Return", "Enter", "Escape", "Tab", "Backspace", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Control", "Shift", "Alt", "Meta"
    ]
    Use `type` for words or phrases. Do not send words to `key`.
  - `hold_key`: Required "text": one of ["Control", "Shift", "Alt", "Meta"], and required "duration" (seconds); Forbidden: coordinates

- Scrolling:
  - `scroll`: Required "scroll_direction" in ["up", "down", "left", "right"], and required "scroll_amount" (integer ≥ 1);
    Forbidden: "coordinate", "start_coordinate", "text", "duration"

- Utility actions:
  - `screenshot`: No parameters at all. Do not add "duration" or coordinates.
  - `wait`: Required "duration" (seconds). No coordinates.
  - `cursor_position`: No parameters.

### Safety Checks (before every call)

- Coordinates within bounds: x in [0, 1365], y in [0, 767]
- Do not pass empty lists for coordinates.
- Do not include parameters that the action does not accept.
- For drags, both "start_coordinate" and "coordinate" must be present.

### Strict Completion Gate (must satisfy ALL)

1) You executed at least one non-screenshot action (click/type/scroll/drag/key).
2) You observed a subsequent screenshot that is not black and not a loading spinner.
3) The stated task objective is achieved, or there is a clear terminal state with no further beneficial actions.

### Forbidden reasons to call `done` (these are always incorrect)

- "The screen is black/blank" → Wait + Screenshot again instead.
- "There is a spinner/progress/it is still loading" → Wait + Screenshot again.
- "The search/navigation has not been performed yet" → Perform the action, do not `done`.
- "No visible elements to interact with yet" → Wait + Screenshot until visible, then act.

### Wrong vs Correct

Wrong (trying to type words with key):
{"action":"key","text":"search this"}  ❌
Correct:
{"action":"type","text":"search this"}  ✅

Wrong (missing start_coordinate on drag):
{"action":"left_click_drag","coordinate":[500,300]}  ❌
Correct:
{"action":"left_click_drag","start_coordinate":[100,100],"coordinate":[500,300]}  ✅

Wrong (`done` while black screen/spinner):
{"action":"done","summary":"The screen is black / still loading"}  ❌
Correct (staged waiting and retry):
{"action":"wait","duration":2} → {"action":"screenshot"}  ✅ (repeat if needed)

Wrong (adding params to screenshot):
{"action":"screenshot","duration":3}  ❌
Correct:
{"action":"screenshot"}  ✅

Wrong (coordinates with type/key/screenshot/wait):
{"action":"type","text":"hello","coordinate":[100,100]}  ❌

### Minimal Flow Example

{"action":"screenshot"}
{"action":"wait","duration":2}
{"action":"screenshot"}
{"action":"left_click","coordinate":[500,300]}
{"action":"type","text":"hello world"}
{"action":"key","text":"Return"}

### Completion

- `done`: Only when ALL Strict Completion Gate conditions are satisfied.
- Never use `done` to report loading states or lack of content; use `wait` then `screenshot` until stable, then act.
- Final Answer Requirement: when you call `done`, the `summary` MUST contain the direct, explicit answer to the user’s question (no meta statements like “identified” or “completed”). Include all requested fields in plain text.
  - Example (Amazon reviews): provide exactly three lines, each with "Reviewer — Rating — Headline".
  - Do not include tool logs or rationale in the summary; only the final answer.

### Tool Call Output (Qwen 2.5 parser)

- Emit exactly ONE tool call per assistant turn.
- Wrap it EXACTLY as below. This is an example of a tool call:
<tool_call>
{"name":"computer","arguments":{"action":"left_click","coordinate":[500,300]}}
</tool_call>
- Reason first, then act: write 1–2 concise sentences of rationale, then emit the tool call block. Avoid extra text after the block.
- Use valid, concise JSON for `arguments`.
- If multiple actions are needed, emit one call now; wait for the tool result next turn before emitting another.
- To finish, use the same format with `{"name":"done","arguments":{"summary":"<FINAL ANSWER HERE>"}}`. The summary must directly answer the user.

### Tool Call Grammar Enforcement (clarification)

- The grammar strictly enforces ONLY the content inside `<tool_call> … </tool_call>`. Any text outside the tag is unconstrained and ignored by the tool caller.
- If you do NOT output the `<tool_call>` block, no tool will run. Actions written as plain JSON or code fences outside the tag are ignored.
- Always use real tool names (e.g., `"computer"`, `"done"`). Never output placeholders like `"TOOL_NAME"`.
- Do NOT wrap the tool call block in markdown code fences. Emit the tag and JSON exactly, with no backticks or extra formatting.

Wrong (plain JSON in prose, no tag — ignored):
{"action":"left_click","coordinate":[500,300]}

Wrong (placeholder tool name — ignored or rejected):
<tool_call>
{"name":"computer","arguments":{"action":"left_click","coordinate":[500,300]}}
</tool_call>

Correct (tag + real tool name + valid arguments):
<tool_call>
{"name":"computer","arguments":{"action":"left_click","coordinate":[500,300]}}
</tool_call>

Reason-then-Act example (correct sequencing):
I see the Amazon home page with the search bar focused. I will type the query to begin the search.
<tool_call>
{"name":"computer","arguments":{"action":"type","text":"MOTU M2 Audio Interface - Portable and Powerful"}}
</tool_call>
"""


 


class TextContent(BaseModel):
    type: str = "text"
    text: str


class ImageUrl(BaseModel):
    url: str


class ImageUrlContent(BaseModel):
    type: str = "image_url"
    image_url: ImageUrl


class UserMessage(BaseModel):
    role: str = "user"
    content: List[Union[TextContent, ImageUrlContent]]


def img_path_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def normalize_image_reference_to_image_url(image_ref: str) -> Optional[str]:
    """Normalize various image references into an OpenAI-style image_url.url string.

    The OpenAI-compatible chat API expects images to be provided via
    `{"type": "image_url", "image_url": {"url": STRING}}` where STRING is a
    network URL or a data URL. This helper takes whatever an MCP tool or caller
    returns and converts it into that STRING.

    Accepts:
    - data URLs (returned as-is)
    - http/https URLs (returned as-is)
    - file:// URLs (converted to data URL if file exists)
    - absolute local paths (converted to data URL if file exists)
    - raw base64 strings (wrapped as data:image/jpeg;base64,...)

    Returns None if the input cannot be normalized.

    Examples:
    - Given a tool JSON like:
      {"base64_image": "data:image/jpeg;base64,/9j/4AAQ..."}
      → returns the same data URL string.

    - Given a public URL:
      "https://example.com/screenshot.jpg"
      → returns the same URL.

    - Given a local file URL:
      "file:///workspace/snap.png"
      → loads from disk and returns a data URL such as
        "data:image/png;base64,iVBORw0KGgoAAAANS...".

    - Given an absolute path:
      "/workspace/snap.png"
      → loads from disk and returns a data URL.

    - Given a raw base64 blob (no prefix):
      "/9j/4AAQSkZJRgABAQAAAQABAAD/..."
      → wraps as "data:image/jpeg;base64,/9j/4AAQ...".
    """
    if not isinstance(image_ref, str):
        return None
    ref = image_ref.strip()
    if not ref:
        return None
    # Already a data URL
    if ref.startswith("data:"):
        return ref
    # HTTP(S)
    if ref.startswith("http://") or ref.startswith("https://"):
        return ref
    # file:// path
    if ref.startswith("file://"):
        file_path = ref[7:]
        try:
            if os.path.exists(file_path):
                return img_path_to_data_url(file_path)
        except Exception:
            return None
        return None
    # Absolute local path
    if ref.startswith("/") and os.path.exists(ref):
        try:
            return img_path_to_data_url(ref)
        except Exception:
            return None
    # Heuristic: raw base64 (no data: or http)
    # Try a fast regex check to avoid heavy decoding for arbitrary text
    if re.fullmatch(r"[A-Za-z0-9+/=\s]+", ref[:256] or ""):
        # Best-effort wrap; do not validate entire payload for speed
        return f"data:image/jpeg;base64,{ref}"
    return None


def manage_context(messages: List[Dict[str, Any]], retain_n_turns: int = 1) -> List[Dict[str, Any]]:
    """Return a minimal message window for the model.

    Policy (simple and robust):
    - Always keep the very first user message (index 0) as context.
    - From the rest, keep only messages that are likely useful for tool flows:
      latest assistant, tool, and user image messages.
    - Limit to the last 2 * retain_n_turns of those filtered messages.

    This yields:
    - Turn 0: 1 message (the first user message).
    - Later turns: up to 3 messages total (first + latest assistant + latest image/tool) per turn retained.
    """
    if not isinstance(messages, list) or not messages:
        return []
    if retain_n_turns <= 0:
        return [messages[0]]

    def _is_user_image_message(m: Dict[str, Any]) -> bool:
        if m.get("role") != "user":
            return False
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
        return False

    base = [messages[0]]
    rest = messages[1:]
    filtered: List[Dict[str, Any]] = []
    for msg in rest:
        role = msg.get("role")
        if role == "assistant" or role == "tool" or _is_user_image_message(msg):
            filtered.append(msg)

    tail_count = max(0, 2 * int(retain_n_turns))
    window_tail = filtered[-tail_count:] if tail_count > 0 else []
    return base + window_tail


def build_user_message_from_sample(sample: Any, tools: Optional[List[Any]] = None) -> Dict[str, Any]:
    contents: List[Union[TextContent, ImageUrlContent]] = []
    if isinstance(sample.prompt, str):
        contents.append(TextContent(text=sample.prompt))
    else:
        text_parts: List[str] = []
        for part in sample.prompt:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image":
                path = part.get("path")
                if path:
                    try:
                        url = img_path_to_data_url(path)
                        contents.append(ImageUrlContent(image_url=ImageUrl(url=url)))
                    except Exception:
                        continue
        if text_parts:
            contents.insert(0, TextContent(text="".join(text_parts)))

    # Optionally append a compact tools usage block as text
    if tools:
        try:
            lines: List[str] = []
            lines.append("Tools available:")
            for t in tools:
                name = getattr(t, "name", None) or getattr(t, "id", "tool")
                desc = (getattr(t, "description", None) or "").strip()
                schema = getattr(t, "inputSchema", {}) or {}
                req = []
                try:
                    req = list((schema.get("required") or []))
                except Exception:
                    req = []
                props = []
                try:
                    props = list(((schema.get("properties") or {}).keys()))
                except Exception:
                    props = []
                line = f"- {name}: required={req} props={props}"
                if desc:
                    line += f" — {desc}"
                lines.append(line)

            # moved 'done' guidance into COMPUTER_TOOL_USAGE_GUIDE to avoid premature termination

            contents.append(TextContent(text="\n".join(lines)))
            # Append a fixed usage guide for the common computer tool
            if any(getattr(t, "name", "") == "computer" for t in tools):
                contents.append(TextContent(text=COMPUTER_TOOL_USAGE_GUIDE))
            
        except Exception:
            # best-effort; ignore tools dump errors
            pass
    # If no images, flatten to string content for router-safe payloads
    has_image = any(getattr(c, "type", "") == "image_url" for c in contents)
    if not has_image:
        text = "".join([c.text for c in contents if isinstance(c, TextContent)])
        return {"role": "user", "content": text}
    return UserMessage(content=contents).model_dump()


def save_data_url_to_image_path(
    data_url: str,
    env_key: str,
    rollout_id: int,
    turn: int,
    root_dir: str = "/workspace/images",
) -> str:
    """Save a data:image/...;base64,... URL to a deterministic local file.

    Returns absolute path to the saved file.
    """
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise ValueError("data_url must be a data: URI")

    header, b64 = data_url.split(",", 1)
    mime = "image/jpeg"
    try:
        if ";" in header:
            mime = header.split(";")[0].split(":")[1]
        else:
            mime = header.split(":")[1]
    except Exception:
        mime = "image/jpeg"

    ext = ".png" if "png" in mime else ".jpg"
    out_dir = Path(root_dir) / str(env_key) / str(rollout_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"turn_{turn}{ext}"

    out_path.write_bytes(base64.b64decode(b64))

    return str(out_path.resolve())


class ToolFunction(BaseModel):
    name: str
    description: str = ""
    parameters: Dict[str, Any] = {}
    strict: bool = True


class ToolSpec(BaseModel):
    type: str = "function"
    function: ToolFunction


def build_tools_param(tools) -> List[Dict[str, Any]]:
    # Convert MCP tools to OpenAI/SGLang tool schema, sanitizing JSON Schema features.
    def _first_non_null_anyof(schema_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        for sub in schema_list:
            if sub.get("type") != "null":
                return sub
        # fallback to first
        return schema_list[0] if schema_list else {"type": "string"}

    def _sanitize_property(prop: Dict[str, Any]) -> Dict[str, Any]:
        # Handle anyOf by picking a non-null simple schema
        if "anyOf" in prop and isinstance(prop["anyOf"], list):
            base = _first_non_null_anyof(prop["anyOf"])
            # Merge to allow enums/types coming from anyOf
            prop = {**prop, **base}

        sanitized: Dict[str, Any] = {}

        # Keep enums if present
        if "enum" in prop:
            sanitized["enum"] = prop["enum"]

        typ = prop.get("type")
        if typ:
            if typ == "array":
                sanitized["type"] = "array"
                # Prefer homogeneous array typing
                if "items" in prop and isinstance(prop["items"], dict):
                    sanitized["items"] = {"type": prop["items"].get("type", "string")}
                elif "prefixItems" in prop and isinstance(prop["prefixItems"], list):
                    # Tuple-style → coerce to homogeneous items of the first element type
                    first = prop["prefixItems"][0] if prop["prefixItems"] else {"type": "string"}
                    sanitized["items"] = {"type": first.get("type", "string")}
                if isinstance(prop.get("minItems"), int):
                    sanitized["minItems"] = prop["minItems"]
                if isinstance(prop.get("maxItems"), int):
                    sanitized["maxItems"] = prop["maxItems"]
            elif typ in ("string", "number", "integer", "boolean", "object"):
                sanitized["type"] = typ
            # Other types are dropped

        # For objects, recursively sanitize nested properties if provided
        if sanitized.get("type") == "object" and isinstance(prop.get("properties"), dict):
            nested_props: Dict[str, Any] = {}
            for k, v in prop["properties"].items():
                if isinstance(v, dict):
                    nested_props[k] = _sanitize_property(v)
            sanitized["properties"] = nested_props
            if isinstance(prop.get("required"), list):
                sanitized["required"] = [x for x in prop["required"] if isinstance(x, str)]

        return sanitized

    def _sanitize_parameters(input_schema: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure top-level object schema
        parameters: Dict[str, Any] = {"type": "object", "properties": {}}
        if not isinstance(input_schema, dict):
            return parameters
        # Properties
        raw_props = input_schema.get("properties")
        if isinstance(raw_props, dict):
            props: Dict[str, Any] = {}
            for name, prop in raw_props.items():
                if isinstance(prop, dict):
                    props[name] = _sanitize_property(prop)
            parameters["properties"] = props
        # Required
        if isinstance(input_schema.get("required"), list):
            parameters["required"] = [x for x in input_schema["required"] if isinstance(x, str)]
        return parameters

    specs: List[ToolSpec] = [
        ToolSpec(
            function=ToolFunction(
                name=t.name,
                description=t.description or "",
                parameters=_sanitize_parameters(getattr(t, "inputSchema", {}) or {}),
            )
        )
        for t in tools
    ]
    # Add a synthetic "done" tool so the model can explicitly signal completion.
    # This is not an MCP tool; it is intercepted by the caller.
    specs.append(
        ToolSpec(
            function=ToolFunction(
                name="done",
                description=(
                    "Signal that the task is complete. Provide a brief 'summary' of "
                    "what was accomplished or why no further actions are needed."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                    },
                    "required": ["summary"],
                },
            )
        )
    )
    return [s.model_dump() for s in specs]


async def list_mcp_tools(session: Any):
    listed = await session.list_tools()
    return listed.tools


