import base64
import json
import mimetypes
from typing import Any, Dict, List, Union

from pydantic import BaseModel


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


def build_user_message_from_sample(sample: Any) -> Dict[str, Any]:
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
    # If no images, flatten to string content for router-safe payloads
    has_image = any(getattr(c, "type", "") == "image_url" for c in contents)
    if not has_image:
        text = "".join([c.text for c in contents if isinstance(c, TextContent)])
        return {"role": "user", "content": text}
    return UserMessage(content=contents).model_dump()


class ToolFunction(BaseModel):
    name: str
    description: str = ""
    parameters: Dict[str, Any] = {}


class ToolSpec(BaseModel):
    type: str = "function"
    function: ToolFunction


def build_tools_param(tools) -> List[Dict[str, Any]]:
    # Emit minimal parameters to avoid advanced JSON-Schema features rejected by routers
    minimal_params = {"type": "object", "properties": {}}
    specs: List[ToolSpec] = [
        ToolSpec(
            function=ToolFunction(
                name=t.name,
                description=t.description or "",
                parameters=minimal_params,
            )
        )
        for t in tools
    ]
    return [s.model_dump() for s in specs]


async def list_mcp_tools(session: Any):
    listed = await session.list_tools()
    return listed.tools


