from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from ...core.event import ConversationEvent, DoneEvent, StreamEvent, ToolCallEvent
from ...tools.tool import ToolCall


class DeepSeekStreamEventHandler:
    def handle(self, events: Iterable[Any]) -> Iterator[StreamEvent]:
        calls: dict[int, dict[str, str]] = {}
        usage = 0
        try:
            for chunk in events:
                chunk_usage = getattr(chunk, "usage", None)
                usage = int(getattr(chunk_usage, "total_tokens", 0) or usage)
                choices = getattr(chunk, "choices", ()) or ()
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None)
                if content:
                    yield ConversationEvent(content)
                for fragment in getattr(delta, "tool_calls", None) or ():
                    index = int(getattr(fragment, "index", 0))
                    call = calls.setdefault(
                        index, {"id": "", "name": "", "arguments": ""}
                    )
                    call["id"] = getattr(fragment, "id", None) or call["id"]
                    function = getattr(fragment, "function", None)
                    call["name"] = getattr(function, "name", None) or call["name"]
                    call["arguments"] += getattr(function, "arguments", None) or ""
            for index in sorted(calls):
                call = calls[index]
                yield ToolCallEvent(
                    ToolCall(
                        call["id"],
                        call["name"],
                        json.loads(call["arguments"] or "{}"),
                    )
                )
            yield DoneEvent(usage)
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()
