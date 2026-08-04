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
class PlanReviewEvent:
    plan_file: str
    content: str


@dataclass(frozen=True)
class PlanReviewResponse:
    approved: bool = False
    feedback: str = ""


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


@dataclass(frozen=True)
class SubagentEvent:
    task_id: str
    name: str
    status: Literal["started", "backgrounded", "completed", "failed", "timed_out"]
    background: bool = False


@dataclass(frozen=True)
class ContextCompactionEvent:
    status: Literal["started", "completed", "skipped"]
    automatic: bool
    before_tokens: int
    after_tokens: int = 0


@dataclass(frozen=True)
class ContextStatusEvent:
    used_tokens: int
    max_tokens: int
    auto_compact_tokens: int


@dataclass(frozen=True)
class SessionStateEvent:
    action: Literal["initialized", "new", "resumed", "deleted"]
    session_id: str
    records: tuple[dict[str, Any], ...]
    token_usage: int = 0
    cleaned: int = 0
    invalid: tuple[str, ...] = ()
    restored: bool = False


@dataclass(frozen=True)
class SessionListEvent:
    sessions: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SkillListEvent:
    skills: tuple[dict[str, Any], ...]


StreamEvent = ConversationEvent | ToolCallEvent | ErrorEvent | DoneEvent
AgentEvent = (
    ConversationEvent
    | ToolCallEvent
    | ToolResultEvent
    | PermissionRequestEvent
    | PlanReviewEvent
    | TurnCompleteEvent
    | LoopCompleteEvent
    | UsageEvent
    | SubagentEvent
    | ContextCompactionEvent
    | ContextStatusEvent
    | SessionStateEvent
    | SessionListEvent
    | SkillListEvent
    | ErrorEvent
)
