from __future__ import annotations

from dataclasses import dataclass

from ..tools.tool import ToolCall


@dataclass(frozen=True)
class ConversationEvent:
    delta: str


@dataclass(frozen=True)
class ToolCallEvent:
    tool_call: ToolCall


@dataclass(frozen=True)
class ErrorEvent:
    message: str
    code: str | None = None


@dataclass(frozen=True)
class DoneEvent:
    token_usage: int = 0


StreamEvent = ConversationEvent | ToolCallEvent | ErrorEvent | DoneEvent
