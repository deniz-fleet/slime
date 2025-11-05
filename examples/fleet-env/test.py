import json
import os
import importlib.util


# Load utils.py from this directory (folder name has a hyphen, so use importlib)
_HERE = os.path.dirname(__file__)
_UTILS_PATH = os.path.join(_HERE, "utils.py")
_spec = importlib.util.spec_from_file_location("fleet_env_utils", _UTILS_PATH)
utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(utils)


class _Sample:
    def __init__(self, prompt):
        self.prompt = prompt


class _Tool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


def test_build_user_message_flattens_text_only():
    sample = _Sample(prompt="Say hi")
    msg = utils.build_user_message_from_sample(sample)
    assert msg["role"] == "user"
    assert isinstance(msg["content"], str)
    assert msg["content"] == "Say hi"


def test_build_tools_param_minimal_schema():
    raw_schema = {
        "type": "object",
        "title": "computerArguments",
        "properties": {
            "action": {
                "type": "string",
                "title": "Action",
                "enum": [
                    "screenshot",
                    "left_click",
                    "right_click",
                ],
            },
            "text": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Text",
            },
            "coordinate": {
                "type": "array",
                "prefixItems": [{"type": "integer"}, {"type": "integer"}],
                "minItems": 2,
                "maxItems": 2,
                "default": None,
            },
        },
        "required": ["action"],
    }

    tools = [_Tool(name="computer", description="Control display, mouse, keyboard.", input_schema=raw_schema)]

    out = utils.build_tools_param(tools)
    assert isinstance(out, list) and len(out) == 1
    fn = out[0]["function"]
    params = fn["parameters"]

    # Expect sanitized JSON Schema suitable for OpenAI/SGLang tool calling
    expected = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "screenshot",
                    "left_click",
                    "right_click",
                ],
            },
            # anyOf(string|null) -> string
            "text": {
                "type": "string",
            },
            # prefixItems -> homogeneous integer items, preserve min/maxItems
            "coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        "required": ["action"],
    }

    assert params == expected


if __name__ == "__main__":
    test_build_user_message_flattens_text_only()
    test_build_tools_param_minimal_schema()
    print("All tests passed.")


