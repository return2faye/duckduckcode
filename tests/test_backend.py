from __future__ import annotations

import io
import json
import unittest

from duckduckcode.core.event import (
    ConversationEvent,
    LoopCompleteEvent,
    PermissionRequestEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    UsageEvent,
)
from duckduckcode.interfaces.backend import run_backend
from duckduckcode.tools.tool import ToolCall


class BackendTest(unittest.TestCase):
    def test_backend_round_trips_permission_responses(self) -> None:
        choices = []

        class FakeAgent:
            def stream(self, message):
                choice = yield PermissionRequestEvent(
                    "call_1",
                    "Bash",
                    "git push origin main",
                    "approval required",
                )
                choices.append(choice)
                yield LoopCompleteEvent("completed", 1)

        output = io.StringIO()
        run_backend(
            FakeAgent(),
            input_stream=io.StringIO(
                '{"message": "push"}\n'
                '{"type": "permission_response", "call_id": "call_1", '
                '"decision": "allow_once"}\n'
            ),
            output_stream=output,
        )

        self.assertEqual(choices, ["allow_once"])
        self.assertEqual(
            [json.loads(line) for line in output.getvalue().splitlines()],
            [
                {
                    "type": "permission_request",
                    "call_id": "call_1",
                    "name": "Bash",
                    "content": "git push origin main",
                    "message": "approval required",
                },
                {
                    "type": "loop_complete",
                    "reason": "completed",
                    "iterations": 1,
                },
            ],
        )

    def test_backend_streams_json_lines_for_each_prompt(self) -> None:
        calls = []

        class FakeAgent:
            def stream(self, message):
                calls.append(message)
                yield ConversationEvent("hi")
                yield ToolCallEvent(
                    ToolCall("call_1", "ReadFile", {"path": "README.md"})
                )
                yield ToolResultEvent("call_1", "ReadFile", "contents")
                yield UsageEvent(7)
                yield TurnCompleteEvent(1)
                yield LoopCompleteEvent("completed", 1)

        output = io.StringIO()

        run_backend(
            FakeAgent(),
            input_stream=io.StringIO('{"message": "hello"}\n'),
            output_stream=output,
        )

        self.assertEqual(calls, ["hello"])
        self.assertEqual(
            [json.loads(line) for line in output.getvalue().splitlines()],
            [
                {"type": "stream_text", "delta": "hi"},
                {
                    "type": "tool_use",
                    "call_id": "call_1",
                    "name": "ReadFile",
                    "arguments": {"path": "README.md"},
                },
                {
                    "type": "tool_result",
                    "call_id": "call_1",
                    "name": "ReadFile",
                    "content": "contents",
                    "is_error": False,
                },
                {"type": "usage", "total_tokens": 7},
                {"type": "turn_complete", "iteration": 1},
                {
                    "type": "loop_complete",
                    "reason": "completed",
                    "iterations": 1,
                },
            ],
        )

    def test_backend_survives_interrupted_stream(self) -> None:
        class FakeAgent:
            def stream(self, message):
                if message == "stop":
                    yield ConversationEvent("partial")
                    raise KeyboardInterrupt
                yield ConversationEvent("next")
                yield LoopCompleteEvent("completed", 1)

        output = io.StringIO()

        run_backend(
            FakeAgent(),
            input_stream=io.StringIO('{"message": "stop"}\n{"message": "continue"}\n'),
            output_stream=output,
        )

        self.assertEqual(
            [json.loads(line) for line in output.getvalue().splitlines()],
            [
                {"type": "stream_text", "delta": "partial"},
                {"type": "error", "message": "interrupted", "code": "interrupted"},
                {
                    "type": "loop_complete",
                    "reason": "cancelled",
                    "iterations": 0,
                },
                {"type": "stream_text", "delta": "next"},
                {
                    "type": "loop_complete",
                    "reason": "completed",
                    "iterations": 1,
                },
            ],
        )

    def test_backend_closes_unexpected_failures(self) -> None:
        class FakeAgent:
            def stream(self, message):
                raise RuntimeError("broken")
                yield

        output = io.StringIO()

        run_backend(
            FakeAgent(),
            input_stream=io.StringIO('{"message": "hello"}\n'),
            output_stream=output,
        )

        self.assertEqual(
            [json.loads(line) for line in output.getvalue().splitlines()],
            [
                {"type": "error", "message": "broken", "code": "error"},
                {
                    "type": "loop_complete",
                    "reason": "error",
                    "iterations": 0,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
