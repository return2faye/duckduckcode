from __future__ import annotations

import json
import signal
import sys
from typing import TextIO

from ..core.agent import Agent
from ..core.event import (
    ConversationEvent,
    ContextCompactionEvent,
    ContextStatusEvent,
    ErrorEvent,
    LoopCompleteEvent,
    PermissionChoice,
    PermissionRequestEvent,
    PlanReviewEvent,
    PlanReviewResponse,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    UsageEvent,
)


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
                data = json.loads(line)
                if data.get("type") == "set_mode":
                    mode = data.get("mode")
                    if mode == "plan":
                        agent.enter_plan_mode()
                    elif mode == "default":
                        agent.cancel_plan_mode()
                    else:
                        raise ValueError("Unsupported agent mode.")
                    continue
                if data.get("type") == "set_permission_mode":
                    agent.set_permission_mode(data.get("mode"))
                    continue
                if data.get("type") == "compact":
                    active = True
                    _run_compact(agent, output_stream)
                    continue
                if data.get("type") == "status":
                    active = True
                    _run_status(agent, output_stream)
                    continue
                message = data["message"]
                active = True
                _run_stream(agent, message, input_stream, output_stream)
            except KeyboardInterrupt:
                for event in (
                    ErrorEvent("interrupted", "interrupted"),
                    LoopCompleteEvent("cancelled", 0),
                ):
                    output_stream.write(json.dumps(_event_to_json(event)) + "\n")
                    output_stream.flush()
            except Exception as exc:
                for event in (
                    ErrorEvent(str(exc), "error"),
                    LoopCompleteEvent("error", 0),
                ):
                    output_stream.write(json.dumps(_event_to_json(event)) + "\n")
                    output_stream.flush()
            finally:
                active = False
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def _run_stream(
    agent: Agent,
    message: str,
    input_stream: TextIO,
    output_stream: TextIO,
) -> None:
    stream = agent.stream(message)
    try:
        event = next(stream)
        while True:
            output_stream.write(json.dumps(_event_to_json(event)) + "\n")
            output_stream.flush()
            if isinstance(event, PermissionRequestEvent):
                choice = _read_permission_response(input_stream, event.call_id)
                event = stream.send(choice)
            elif isinstance(event, PlanReviewEvent):
                response = _read_plan_review_response(input_stream)
                event = stream.send(response)
            else:
                event = next(stream)
    except StopIteration:
        return
    finally:
        stream.close()


def _run_compact(agent: Agent, output_stream: TextIO) -> None:
    stream = agent.compact()
    try:
        for event in stream:
            output_stream.write(json.dumps(_event_to_json(event)) + "\n")
            output_stream.flush()
    finally:
        stream.close()


def _run_status(agent: Agent, output_stream: TextIO) -> None:
    stream = agent.context_status()
    try:
        for event in stream:
            output_stream.write(json.dumps(_event_to_json(event)) + "\n")
            output_stream.flush()
    finally:
        stream.close()


def _read_permission_response(input_stream: TextIO, call_id: str) -> PermissionChoice:
    line = input_stream.readline()
    if not line:
        raise RuntimeError("Permission response stream closed.")
    data = json.loads(line)
    choices = {"allow_once", "allow_always", "deny"}
    if (
        data.get("type") != "permission_response"
        or data.get("call_id") != call_id
        or data.get("decision") not in choices
    ):
        raise RuntimeError(f"Invalid permission response for '{call_id}'.")
    return data["decision"]


def _read_plan_review_response(input_stream: TextIO) -> PlanReviewResponse:
    line = input_stream.readline()
    if not line:
        raise RuntimeError("Plan review response stream closed.")
    data = json.loads(line)
    approved = data.get("approved")
    feedback = data.get("feedback", "")
    if (
        data.get("type") != "plan_review_response"
        or not isinstance(approved, bool)
        or not isinstance(feedback, str)
    ):
        raise RuntimeError("Invalid plan review response.")
    return PlanReviewResponse(approved, feedback)


def _event_to_json(event: object) -> dict[str, object]:
    if isinstance(event, ConversationEvent):
        return {"type": "stream_text", "delta": event.delta}
    if isinstance(event, ErrorEvent):
        return {"type": "error", "message": event.message, "code": event.code}
    if isinstance(event, ToolCallEvent):
        return {
            "type": "tool_use",
            "call_id": event.tool_call.call_id,
            "name": event.tool_call.name,
            "arguments": event.tool_call.arguments,
        }
    if isinstance(event, ToolResultEvent):
        return {
            "type": "tool_result",
            "call_id": event.call_id,
            "name": event.name,
            "content": event.content,
            "is_error": event.is_error,
        }
    if isinstance(event, PermissionRequestEvent):
        return {
            "type": "permission_request",
            "call_id": event.call_id,
            "name": event.name,
            "content": event.content,
            "message": event.message,
        }
    if isinstance(event, PlanReviewEvent):
        return {
            "type": "plan_review",
            "plan_file": event.plan_file,
            "content": event.content,
        }
    if isinstance(event, UsageEvent):
        return {"type": "usage", "total_tokens": event.total_tokens}
    if isinstance(event, ContextCompactionEvent):
        return {
            "type": "context_compaction",
            "status": event.status,
            "automatic": event.automatic,
            "before_tokens": event.before_tokens,
            "after_tokens": event.after_tokens,
        }
    if isinstance(event, ContextStatusEvent):
        return {
            "type": "context_status",
            "used_tokens": event.used_tokens,
            "max_tokens": event.max_tokens,
            "auto_compact_tokens": event.auto_compact_tokens,
        }
    if isinstance(event, TurnCompleteEvent):
        return {"type": "turn_complete", "iteration": event.iteration}
    if isinstance(event, LoopCompleteEvent):
        return {
            "type": "loop_complete",
            "reason": event.reason,
            "iterations": event.iterations,
        }
    return {"type": "unknown"}
