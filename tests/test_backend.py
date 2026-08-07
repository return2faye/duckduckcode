from __future__ import annotations

import io
import json
import signal
import unittest

from duckduckcode.core.event import (
    ConversationEvent,
    ContextCompactionEvent,
    ContextStatusEvent,
    LoopCompleteEvent,
    PermissionRequestEvent,
    PlanReviewEvent,
    PlanReviewResponse,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    UsageEvent,
    SubagentEvent,
)
from duckduckcode.interfaces.backend import run_backend
from duckduckcode.tools.tool import ToolCall


class BackendTest(unittest.TestCase):
    def test_backend_serializes_subagent_status_and_detach_signal(self) -> None:
        class FakeAgent:
            def __init__(self):
                self.detached = 0

            def detach_subagent(self):
                self.detached += 1
                return True

            def stream(self, message):
                signal.raise_signal(signal.SIGUSR1)
                yield SubagentEvent("task", "explore", "backgrounded", True)
                yield LoopCompleteEvent("completed", 1)

        agent = FakeAgent()
        output = io.StringIO()
        run_backend(
            agent,
            input_stream=io.StringIO('{"message":"detach"}\n'),
            output_stream=output,
        )

        self.assertEqual(agent.detached, 1)
        self.assertEqual(
            json.loads(output.getvalue().splitlines()[0]),
            {
                "type": "subagent",
                "task_id": "task",
                "name": "explore",
                "status": "backgrounded",
                "background": True,
            },
        )

    def test_backend_reports_context_status(self) -> None:
        class FakeAgent:
            def context_status(self):
                yield ContextStatusEvent(84_000, 200_000, 167_000)
                yield LoopCompleteEvent("completed", 0)

        output = io.StringIO()
        run_backend(
            FakeAgent(),
            input_stream=io.StringIO('{"type": "status"}\n'),
            output_stream=output,
        )

        self.assertEqual(
            [json.loads(line) for line in output.getvalue().splitlines()],
            [
                {
                    "type": "context_status",
                    "used_tokens": 84_000,
                    "max_tokens": 200_000,
                    "auto_compact_tokens": 167_000,
                },
                {
                    "type": "loop_complete",
                    "reason": "completed",
                    "iterations": 0,
                },
            ],
        )

    def test_backend_runs_manual_context_compaction(self) -> None:
        class FakeAgent:
            def compact(self):
                yield ContextCompactionEvent("started", False, 170_000)
                yield ContextCompactionEvent("completed", False, 170_000, 40_000)
                yield LoopCompleteEvent("completed", 0)

        output = io.StringIO()
        run_backend(
            FakeAgent(),
            input_stream=io.StringIO('{"type": "compact"}\n'),
            output_stream=output,
        )

        self.assertEqual(
            [json.loads(line) for line in output.getvalue().splitlines()],
            [
                {
                    "type": "context_compaction",
                    "status": "started",
                    "automatic": False,
                    "before_tokens": 170_000,
                    "after_tokens": 0,
                },
                {
                    "type": "context_compaction",
                    "status": "completed",
                    "automatic": False,
                    "before_tokens": 170_000,
                    "after_tokens": 40_000,
                },
                {
                    "type": "loop_complete",
                    "reason": "completed",
                    "iterations": 0,
                },
            ],
        )

    def test_backend_switches_modes_without_sending_commands_to_model(self) -> None:
        calls = []

        class FakeAgent:
            def enter_plan_mode(self):
                calls.append("plan")

            def cancel_plan_mode(self):
                calls.append("default")

            def set_permission_mode(self, mode):
                calls.append(mode)

            def stream(self, message):
                calls.append(message)
                yield LoopCompleteEvent("completed", 1)

        run_backend(
            FakeAgent(),
            input_stream=io.StringIO(
                '{"type": "set_mode", "mode": "plan"}\n'
                '{"type": "set_mode", "mode": "default"}\n'
                '{"type": "set_permission_mode", "mode": "accept_edits"}\n'
                '{"message": "design the feature"}\n'
            ),
            output_stream=io.StringIO(),
        )

        self.assertEqual(
            calls,
            ["plan", "default", "accept_edits", "design the feature"],
        )

    def test_backend_round_trips_free_form_plan_review_feedback(self) -> None:
        responses = []

        class FakeAgent:
            def stream(self, message):
                response = yield PlanReviewEvent(
                    "/repo/.duckduckcode/plan.md",
                    "# Plan\n\nUse SQLite.",
                )
                responses.append(response)
                yield LoopCompleteEvent("completed", 1)

        output = io.StringIO()
        run_backend(
            FakeAgent(),
            input_stream=io.StringIO(
                '{"message": "plan"}\n'
                '{"type": "plan_review_response", "approved": false, '
                '"feedback": "Use SQLite instead"}\n'
            ),
            output_stream=output,
        )

        self.assertEqual(
            responses,
            [
                PlanReviewResponse(
                    approved=False,
                    feedback="Use SQLite instead",
                )
            ],
        )
        self.assertEqual(
            [json.loads(line) for line in output.getvalue().splitlines()],
            [
                {
                    "type": "plan_review",
                    "plan_file": "/repo/.duckduckcode/plan.md",
                    "content": "# Plan\n\nUse SQLite.",
                },
                {
                    "type": "loop_complete",
                    "reason": "completed",
                    "iterations": 1,
                },
            ],
        )

    def test_backend_accepts_an_explicit_plan_denial_without_feedback(self) -> None:
        responses = []

        class FakeAgent:
            def stream(self, message):
                response = yield PlanReviewEvent("/repo/plan.md", "# Plan")
                responses.append((message, response))
                yield LoopCompleteEvent("completed", 1)

        run_backend(
            FakeAgent(),
            input_stream=io.StringIO(
                '{"message": "plan"}\n'
                '{"type": "plan_review_response", "approved": false, '
                '"feedback": ""}\n'
            ),
            output_stream=io.StringIO(),
        )

        self.assertEqual(
            responses,
            [("plan", PlanReviewResponse(approved=False, feedback=""))],
        )

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

    def test_backend_round_trips_initialize_permission_responses(self) -> None:
        choices = []

        class FakeAgent:
            def initialize(self):
                choice = yield PermissionRequestEvent(
                    "mcp_project_config",
                    "MCP",
                    '[{"server": "files", "transport": "stdio"}]',
                    "approve project MCP configuration",
                )
                choices.append(choice)
                yield LoopCompleteEvent("completed", 0)

        output = io.StringIO()
        run_backend(
            FakeAgent(),
            input_stream=io.StringIO(
                '{"type": "initialize"}\n'
                '{"type": "permission_response", '
                '"call_id": "mcp_project_config", "decision": "allow_once"}\n'
            ),
            output_stream=output,
        )

        self.assertEqual(choices, ["allow_once"])
        self.assertEqual(
            json.loads(output.getvalue().splitlines()[-1])["reason"], "completed"
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

    def test_backend_passes_selected_skills_to_stream(self) -> None:
        calls = []

        class FakeAgent:
            def stream(self, message, selected_skills=()):
                calls.append((message, selected_skills))
                yield LoopCompleteEvent("completed", 1)

        run_backend(
            FakeAgent(),
            input_stream=io.StringIO(
                '{"message": "hello", "skills": ["demo", "other"]}\n'
            ),
            output_stream=io.StringIO(),
        )

        self.assertEqual(calls, [("hello", ["demo", "other"])])

    def test_backend_rejects_malformed_selected_skills(self) -> None:
        class FakeAgent:
            def stream(self, message, selected_skills=()):
                raise AssertionError("stream should not run")

        output = io.StringIO()
        run_backend(
            FakeAgent(),
            input_stream=io.StringIO('{"message": "hello", "skills": [""]}\n'),
            output_stream=output,
        )

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[0]["code"], "error")
        self.assertIn("non-empty exact skill names", events[0]["message"])
        self.assertEqual(events[-1]["reason"], "error")

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
