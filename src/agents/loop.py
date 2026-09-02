"""A minimal, dependency-light agent loop for the AI Agents phase.

Only standard-library ``json`` is needed for the loop itself; an OpenAI client
(or any client exposing ``chat.completions.create``) is passed in.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from src.agents.tool import execute_tool_call


def run_agent_loop(
    client: Any,
    system_prompt: str,
    user_input: str,
    tools: dict[str, Callable],
    tool_schemas: list[dict] | None = None,
    model: str = "gpt-4o-mini",
    max_iterations: int = 5,
) -> str:
    """Run a tool-calling agent until the model emits a final message.

    Args:
        client: object with ``chat.completions.create(...)`` (OpenAI-compatible).
        system_prompt: instructions for the assistant.
        user_input: the opening user message.
        tools: ``{name: callable}`` the agent may call.
        tool_schemas: OpenAI function schemas (see :func:`src.agents.tool.tool_schema`).
        model: model identifier string.
        max_iterations: safety cap on tool rounds.

    Returns:
        The final assistant text reply.
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    kwargs = {"model": model, "messages": messages}
    if tool_schemas:
        kwargs["tools"] = tool_schemas

    for _ in range(max_iterations):
        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        messages.append(message.model_dump() if hasattr(message, "model_dump") else message)

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return message.content or ""

        for call in tool_calls:
            name = call.function.name
            arguments = json.loads(call.function.arguments or "{}")
            result = execute_tool_call(tools, name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )

    return "Agent stopped: iteration cap reached."