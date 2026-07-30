from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..tools.tool import ToolCall
from .prompts import build_system_prompt


@dataclass(frozen=True)
class ReasoningConfig:
    effort: str = "low"


@dataclass(frozen=True)
class Message:
    role: str
    content: str = ""
    kind: Literal["message", "tool_call", "tool_result"] = "message"
    status: Literal["completed", "streaming", "error"] = "completed"
    token_usage: int = 0
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None

    @classmethod
    def tool_call(cls, call_id: str, name: str, arguments: dict[str, Any]) -> Message:
        return cls(
            "assistant",
            kind="tool_call",
            tool_call_id=call_id,
            tool_name=name,
            tool_arguments=arguments,
        )

    @classmethod
    def tool_result(cls, call_id: str, output: str) -> Message:
        return cls("tool", output, kind="tool_result", tool_call_id=call_id)

    def to_openai(self) -> dict[str, Any]:
        if self.kind == "tool_call":
            return {
                "type": "function_call",
                "call_id": self.tool_call_id,
                "name": self.tool_name,
                "arguments": json_dumps(self.tool_arguments or {}),
            }
        if self.kind == "tool_result":
            return {
                "type": "function_call_output",
                "call_id": self.tool_call_id,
                "output": self.content,
            }
        return {"role": self.role, "content": self.content}


class ContextManager:
    def __init__(
        self,
        system_prompt: str | None = None,
        workspace: str | Path | None = None,
        mode_instructions: str = "",
        abstraction: str = "",
        reasoning: ReasoningConfig | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> None:
        self._messages: list[Message] = []
        self.system_prompt = system_prompt or build_system_prompt(
            workspace, mode_instructions=mode_instructions
        )
        self.abstraction = abstraction
        self.reasoning = reasoning or ReasoningConfig()
        self._tool_schemas = tool_schemas or []

    def add_user(self, content: str) -> None:
        self._messages.append(Message("user", content))

    def add_assistant(self, content: str) -> None:
        self._messages.append(Message("assistant", content))

    def start_assistant_stream(self) -> int:
        self._messages.append(Message("assistant", status="streaming"))
        return len(self._messages) - 1

    def append_assistant_delta(self, index: int, delta: str) -> None:
        message = self._messages[index]
        self._messages[index] = Message(
            message.role,
            message.content + delta,
            message.kind,
            message.status,
            message.token_usage,
            message.tool_call_id,
            message.tool_name,
            message.tool_arguments,
        )

    def finish_assistant_stream(self, index: int, token_usage: int = 0) -> None:
        self._set_assistant_stream_status(index, "completed", token_usage)

    def fail_assistant_stream(self, index: int) -> None:
        self._set_assistant_stream_status(index, "error", 0)

    def _set_assistant_stream_status(
        self, index: int, status: Literal["completed", "error"], token_usage: int
    ) -> None:
        message = self._messages[index]
        self._messages[index] = Message(
            message.role,
            message.content,
            message.kind,
            status,
            token_usage,
            message.tool_call_id,
            message.tool_name,
            message.tool_arguments,
        )

    def add_tool_call(self, tool_call: ToolCall) -> None:
        self._messages.append(
            Message.tool_call(tool_call.call_id, tool_call.name, tool_call.arguments)
        )

    def add_tool_result(self, call_id: str, output: str) -> None:
        self._messages.append(Message.tool_result(call_id, output))

    def messages(self) -> list[Message]:
        return list(self._messages)

    def model_messages(self) -> list[Message]:
        messages = [Message("system", self.system_prompt)]
        if self.abstraction:
            messages.append(
                Message("system", f"Conversation summary:\n{self.abstraction}")
            )
        return messages + self.messages()

    def tool_schemas(self) -> list[dict[str, Any]]:
        return list(self._tool_schemas)

    def set_tool_schemas(self, tool_schemas: list[dict[str, Any]]) -> None:
        self._tool_schemas = list(tool_schemas)


def json_dumps(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value)
