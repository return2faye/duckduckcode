from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from .event import ConversationEvent, DoneEvent, ErrorEvent, StreamEvent, ToolCallEvent
from .tool import ToolCall


class OpenAIStreamEventParser:
    def parse(self, event: Any) -> StreamEvent | None:
        event_type = getattr(event, "type", "")
        if event_type == "response.output_text.delta":
            return ConversationEvent(getattr(event, "delta", ""))
        if event_type == "response.function_call_arguments.done":
            return ToolCallEvent(_tool_call_from_event(event))
        if event_type == "response.output_item.done":
            item = getattr(event, "item", None)
            if getattr(item, "type", "") == "function_call":
                return ToolCallEvent(_tool_call_from_event(item))
        if event_type == "error":
            return ErrorEvent(
                getattr(event, "message", ""), getattr(event, "code", None)
            )
        if event_type == "response.failed":
            return ErrorEvent(
                str(
                    getattr(
                        getattr(event, "response", None), "error", "response failed"
                    )
                )
            )
        if event_type == "response.completed":
            return DoneEvent(_token_usage(getattr(event, "response", None)))
        return None


class OpenAIStreamEventHandler:
    def __init__(self, parser: OpenAIStreamEventParser | None = None) -> None:
        self.parser = parser or OpenAIStreamEventParser()

    def handle(self, events: Iterable[Any]) -> Iterator[StreamEvent]:
        try:
            for event in events:
                parsed = self.parser.parse(event)
                if parsed is not None:
                    yield parsed
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()


def _tool_call_from_event(event: Any) -> ToolCall:
    return ToolCall(
        getattr(event, "call_id", None) or getattr(event, "item_id"),
        getattr(event, "name"),
        json.loads(getattr(event, "arguments", "") or "{}"),
    )


def _token_usage(response: Any) -> int:
    usage = getattr(response, "usage", None)
    return int(getattr(usage, "total_tokens", 0) or 0)
