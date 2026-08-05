from __future__ import annotations

import json
from typing import Any

from ...core.context import Message


class DeepSeekChatSerializer:
    def serialize(self, messages: list[Message]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.kind == "tool_call":
                calls = []
                while index < len(messages) and messages[index].kind == "tool_call":
                    call = messages[index]
                    calls.append(
                        {
                            "id": call.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": call.tool_name,
                                "arguments": json.dumps(call.tool_arguments or {}),
                            },
                        }
                    )
                    index += 1
                content = None
                if (
                    serialized
                    and serialized[-1].get("role") == "assistant"
                    and "tool_calls" not in serialized[-1]
                ):
                    content = serialized.pop()["content"] or None
                serialized.append(
                    {"role": "assistant", "content": content, "tool_calls": calls}
                )
                continue
            if message.kind == "tool_result":
                serialized.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": message.tool_call_id,
                    }
                )
            else:
                serialized.append({"role": message.role, "content": message.content})
            index += 1
        return serialized


def serialize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]
