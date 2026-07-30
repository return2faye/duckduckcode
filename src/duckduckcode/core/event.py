from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..tools.tool import ToolCall


@dataclass(frozen=True)
class ConversationEvent:
    delta: str


@dataclass(frozen=True)
class ToolCallEvent:
    tool_call: ToolCall

    @classmethod
    def create(
        cls, call_id: str, name: str, arguments: dict[str, Any]
    ) -> ToolCallEvent:
        return cls(ToolCall(call_id, name, arguments))


@dataclass(frozen=True)
class ErrorEvent:
    message: str
    code: str | None = None


@dataclass(frozen=True)
class DoneEvent:
    token_usage: int = 0


# ======== Agent Events ========


@dataclass(frozen=True)
class ToolResultEvent:
    call_id: str
    name: str
    content: str
    is_error: bool = False


PermissionChoice = Literal["allow_once", "allow_always", "deny"]


@dataclass(frozen=True)
class PermissionRequestEvent:
    call_id: str
    name: str
    content: str
    message: str


@dataclass(frozen=True)
class TurnCompleteEvent:
    iteration: int


@dataclass(frozen=True)
class LoopCompleteEvent:
    reason: Literal["completed", "max_iterations", "cancelled", "error"]
    iterations: int


@dataclass(frozen=True)
class UsageEvent:
    total_tokens: int


StreamEvent = ConversationEvent | ToolCallEvent | ErrorEvent | DoneEvent
AgentEvent = (
    ConversationEvent
    | ToolCallEvent
    | ToolResultEvent
    | PermissionRequestEvent
    | TurnCompleteEvent
    | LoopCompleteEvent
    | UsageEvent
    | ErrorEvent
)
