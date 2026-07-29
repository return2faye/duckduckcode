from __future__ import annotations

import json
import signal
import sys
from typing import TextIO

from .agent import Agent
from .event import ConversationEvent, DoneEvent, ErrorEvent, ToolCallEvent


def run_backend(
    agent: Agent,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    active = False

    def interrupt_active_stream(_signum: int, _frame: object) -> None:
        if active:
            raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGINT, interrupt_active_stream)
    try:
        for line in input_stream:
            try:
                message = json.loads(line)["message"]
                active = True
                for event in agent.stream(message):
                    output_stream.write(json.dumps(_event_to_json(event)) + "\n")
                    output_stream.flush()
            except KeyboardInterrupt:
                output_stream.write(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "interrupted",
                            "code": "interrupted",
                        }
                    )
                    + "\n"
                )
                output_stream.flush()
            except Exception as exc:
                output_stream.write(
                    json.dumps({"type": "error", "message": str(exc)}) + "\n"
                )
                output_stream.flush()
            finally:
                active = False
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def _event_to_json(event: object) -> dict[str, object]:
    if isinstance(event, ConversationEvent):
        return {"type": "delta", "text": event.delta}
    if isinstance(event, DoneEvent):
        return {"type": "done", "token_usage": event.token_usage}
    if isinstance(event, ErrorEvent):
        return {"type": "error", "message": event.message, "code": event.code}
    if isinstance(event, ToolCallEvent):
        return {"type": "tool", "name": event.tool_call.name}
    return {"type": "unknown"}
