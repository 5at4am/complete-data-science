"""Tool-calling helpers for the AI Agents phase (Phase 14).

OpenAI is imported lazily so this module stays light until actually used.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_SCHEMA_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def tool_schema(name: str, description: str, parameters: dict[str, dict]) -> dict:
    """Build a JSON-schema tool definition for an OpenAI-style client.

    ``parameters`` maps argument name -> {"type": python type, "description": str}.
    """
    props = {}
    for pname, spec in parameters.items():
        ptype = spec.get("type")
        properties = {
            "type": _SCHEMA_TYPES.get(ptype, "string"),
            "description": spec.get("description", ""),
        }
        if "enum" in spec:
            properties["enum"] = spec["enum"]
        props[pname] = properties
    required = [pname for pname, spec in parameters.items() if spec.get("required")]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def execute_tool_call(tools: dict[str, Callable], name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a single tool call. Returns a string result (or error text)."""
    if name not in tools:
        return f"ERROR: unknown tool '{name}'"
    try:
        result = tools[name](**arguments)
    except Exception as exc:  # noqa: BLE001 - reporting the error is the point
        return f"ERROR: {type(exc).__name__}: {exc}"
    return str(result)
