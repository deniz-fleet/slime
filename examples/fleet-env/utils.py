import base64
from pathlib import Path
import json
import mimetypes
import os
import re
from typing import Any, Dict, List, Union, Optional

from pydantic import BaseModel


COMPUTER_TOOL_USAGE_GUIDE = """## Computer Control Tool Usage Guide

### Screen Information

- Resolution: 1366x768 pixels (scaled from 1920x1080)

- Coordinate system: (0,0) is top-left, (1365,767) is bottom-right

- All coordinates must be integers within valid ranges



### Action Parameters



1. **Mouse Actions** (require coordinate):

   - `left_click`: Click at [x, y]

   - `right_click`: Right-click at [x, y]  

   - `double_click`: Double-click at [x, y]

   - `triple_click`: Triple-click at [x, y]

   - `middle_click`: Middle-click at [x, y]

   - `mouse_move`: Move cursor to [x, y]



2. **Drag Operations** (require both start_coordinate and coordinate):

   - `left_click_drag`: Drag from start_coordinate to coordinate

   

3. **Keyboard Actions**:

   - `type`: Enter text (text parameter required, no coordinate)

   - `key`: Press special key like "Return", "Tab", "Escape" (text parameter)

   - `hold_key`: Hold modifier key for duration seconds



4. **Scrolling**:

   - `scroll`: Requires scroll_direction ("up"/"down"/"left"/"right") and scroll_amount



5. **Utility Actions**:

   - `screenshot`: Take screenshot (no parameters needed)

   - `wait`: Pause for duration seconds

   - `cursor_position`: Get current cursor position



### Common Mistakes to Avoid:

❌ DO NOT use string labels for coordinates: ["search_bar", "end"]

❌ DO NOT use negative coordinates: [-1, 48]  

❌ DO NOT mix action types with wrong parameters

❌ DO NOT use coordinate parameter with type/key actions

✅ DO use integer coordinates: [683, 400]

✅ DO take screenshots first to see the screen

✅ DO verify coordinates are within bounds (x: 0-1365, y: 0-767)



### Correct Examples:

# Take screenshot first to see what's on screen

{"action": "screenshot"}



# Click on a button at specific location

{"action": "left_click", "coordinate": [500, 300]}



# Type text (no coordinate needed)

{"action": "type", "text": "Hello World"}



# Press Enter key

{"action": "key", "text": "Return"}



# Drag from one point to another

{"action": "left_click_drag", "start_coordinate": [100, 100], "coordinate": [500, 500]}



# Scroll down 5 steps

{"action": "scroll", "scroll_direction": "down", "scroll_amount": 5}
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

            # Always expose the synthetic 'done' tool to the model in the prompt.
            lines.append(
                "- done: required=['summary'] props=['summary'] — Signal completion; "
                "include a brief summary of the outcome."
            )

            # Brief instruction on how to end the session
            lines.append(
                "When you believe the task is complete, call the tool 'done' with a 'summary'. "
                "If not complete, continue using other tools."
            )

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


