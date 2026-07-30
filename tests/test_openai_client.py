from __future__ import annotations

import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from duckduckcode.core import event as event_module
from duckduckcode.core.agent import Agent
from duckduckcode.core.client import ClientResponse
from duckduckcode.core.context import ContextManager, Message, ReasoningConfig
from duckduckcode.core.event import (
    ConversationEvent,
    DoneEvent,
    ErrorEvent,
    LoopCompleteEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    UsageEvent,
)
from duckduckcode.core.prompts import buildSystemPrompt, build_system_prompt
from duckduckcode.providers.openai.client import OpenAIClient
from duckduckcode.providers.openai.serialize import (
    OpenAIResponsesDeserializer,
    OpenAIResponsesSerializer,
)
from duckduckcode.tools.tool import ToolCall, ToolManager, ToolResult


class FakeResponses:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return [
            type("Event", (), {"type": "response.output_text.delta", "delta": "ok"})()
        ]


class AgentEventTest(unittest.TestCase):
    def test_exposes_react_loop_event_types(self) -> None:
        expected = {
            "ToolResultEvent",
            "TurnCompleteEvent",
            "LoopCompleteEvent",
            "UsageEvent",
            "AgentEvent",
        }

        self.assertEqual(expected - set(dir(event_module)), set())


class OpenAIClientTest(unittest.TestCase):
    def test_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                OpenAIClient()

    def test_stream_uses_injected_serializer(self) -> None:
        fake_responses = FakeResponses()
        serializer = type(
            "Serializer",
            (),
            {
                "serialize": lambda self, messages, reasoning: {
                    "reasoning": {"effort": reasoning.effort},
                    "input": [
                        {"role": item.role, "content": item.content}
                        for item in messages
                    ],
                }
            },
        )()
        client = OpenAIClient(
            api_key="test-key",
            model="test-model",
            serializer=serializer,
        )
        client._client = type("Client", (), {"responses": fake_responses})()

        messages = [Message("user", "hello")]
        self.assertEqual(list(client.stream(messages)), [ConversationEvent("ok")])
        self.assertEqual(fake_responses.kwargs["model"], "test-model")
        self.assertTrue(fake_responses.kwargs["stream"])
        self.assertEqual(
            fake_responses.kwargs["input"], [{"role": "user", "content": "hello"}]
        )
        self.assertEqual(fake_responses.kwargs["reasoning"], {"effort": "low"})
        self.assertFalse(hasattr(client, "chat_once"))

    def test_stream_uses_injected_event_handler(self) -> None:
        fake_responses = FakeResponses()
        handler = type(
            "Handler",
            (),
            {
                "handle": lambda self, events: (
                    ConversationEvent(f"custom:{event.delta}") for event in events
                )
            },
        )()
        client = OpenAIClient(api_key="test-key", event_handler=handler)
        client._client = type("Client", (), {"responses": fake_responses})()

        self.assertEqual(
            list(client.stream([Message("user", "hello")])),
            [ConversationEvent("custom:ok")],
        )

    def test_defaults_to_o4_mini_with_low_reasoning(self) -> None:
        fake_responses = FakeResponses()
        client = OpenAIClient(api_key="test-key")
        client._client = type("Client", (), {"responses": fake_responses})()

        list(client.stream([Message("user", "hello")]))

        self.assertEqual(fake_responses.kwargs["model"], "o4-mini")
        self.assertEqual(fake_responses.kwargs["reasoning"], {"effort": "low"})

    def test_stream_passes_tools_and_returns_tool_calls(self) -> None:
        fake_responses = FakeResponses()
        fake_responses.create = lambda **kwargs: setattr(
            fake_responses, "kwargs", kwargs
        ) or [
            type(
                "Event",
                (),
                {
                    "type": "response.output_item.done",
                    "item": type(
                        "Item",
                        (),
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "read_file",
                            "arguments": '{"path": "README.md"}',
                        },
                    )(),
                },
            )()
        ]
        client = OpenAIClient(api_key="test-key", model="test-model")
        client._client = type("Client", (), {"responses": fake_responses})()
        tools = [
            {"type": "function", "name": "read_file", "parameters": {}, "strict": True}
        ]

        events = list(client.stream([Message("user", "read README")], tools=tools))

        self.assertEqual(fake_responses.kwargs["tools"], tools)
        self.assertEqual(
            events,
            [ToolCallEvent(ToolCall("call_1", "read_file", {"path": "README.md"}))],
        )

    def test_openai_serializer_puts_system_prompt_first(self) -> None:
        payload = OpenAIResponsesSerializer().serialize(
            [
                Message("system", "system prompt"),
                Message("user", "hello"),
                Message("assistant", "hi"),
            ],
            ReasoningConfig("low"),
        )

        self.assertEqual(
            payload["input"],
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )
        self.assertNotIn("instructions", payload)
        self.assertEqual(payload["reasoning"], {"effort": "low"})

    def test_openai_serializer_handles_tool_messages(self) -> None:
        payload = OpenAIResponsesSerializer().serialize(
            [
                Message("system", "system prompt"),
                Message.tool_call("call_1", "read_file", {"path": "README.md"}),
                Message.tool_result("call_1", "file contents"),
            ],
            ReasoningConfig("low"),
        )

        self.assertEqual(
            payload["input"],
            [
                {"role": "system", "content": "system prompt"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path": "README.md"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "file contents",
                },
            ],
        )

    def test_openai_deserializer_returns_text_and_tool_calls(self) -> None:
        response = type(
            "Response",
            (),
            {
                "output_text": "done",
                "output": [
                    type(
                        "Call",
                        (),
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "read_file",
                            "arguments": '{"path": "README.md"}',
                        },
                    )()
                ],
            },
        )()

        self.assertEqual(
            OpenAIResponsesDeserializer().deserialize(response),
            ClientResponse(
                "done", [ToolCall("call_1", "read_file", {"path": "README.md"})]
            ),
        )


class ContextManagerTest(unittest.TestCase):
    def test_build_system_prompt_includes_identity_environment_and_mode_slot(
        self,
    ) -> None:
        prompt = build_system_prompt(
            workspace="/repo",
            os_name="TestOS",
            mode_instructions="Mode: test-only",
            model="test-model",
        )

        self.assertIn("You are DuckDuckCode", prompt)
        self.assertIn("Role constraints:", prompt)
        self.assertIn("Behavior guidelines:", prompt)
        self.assertIn("Tool use:", prompt)
        self.assertIn("Bug fixes:", prompt)
        self.assertIn("Security and safety:", prompt)
        self.assertIn("Use absolute file paths when calling file tools.", prompt)
        self.assertIn(
            "Do not refuse to start a long-running service solely because Bash",
            prompt,
        )
        self.assertIn("detached background process", prompt)
        self.assertIn("log file", prompt)
        self.assertIn("verify that it started", prompt)
        self.assertIn("OS: TestOS", prompt)
        self.assertIn("Model: test-model", prompt)
        self.assertIn("Working directory: /repo", prompt)
        self.assertIn("Mode: test-only", prompt)
        self.assertEqual(
            buildSystemPrompt("/repo", "TestOS", "Mode: test-only", "test-model"),
            prompt,
        )

    def test_context_builds_default_system_prompt_for_workspace(self) -> None:
        context = ContextManager(workspace="/repo")

        self.assertIn("Working directory: /repo", context.model_messages()[0].content)

    def test_model_messages_include_system_prompt_and_abstraction_first(self) -> None:
        context = ContextManager(system_prompt="system prompt", abstraction="summary")
        context.add_user("hello")

        self.assertEqual(
            context.model_messages(),
            [
                Message("system", "system prompt"),
                Message("system", "Conversation summary:\nsummary"),
                Message("user", "hello"),
            ],
        )
        self.assertEqual(context.messages(), [Message("user", "hello")])

    def test_adds_user_and_assistant_messages(self) -> None:
        context = ContextManager(system_prompt="system prompt")

        context.add_user("hello")
        context.add_assistant("hi")
        context.add_tool_call(ToolCall("call_1", "read_file", {"path": "README.md"}))
        context.add_tool_result("call_1", "file contents")

        self.assertEqual(
            context.messages(),
            [
                Message("user", "hello"),
                Message("assistant", "hi"),
                Message.tool_call("call_1", "read_file", {"path": "README.md"}),
                Message.tool_result("call_1", "file contents"),
            ],
        )

    def test_defaults_to_low_reasoning(self) -> None:
        self.assertEqual(ContextManager().reasoning, ReasoningConfig("low"))

    def test_context_stores_tool_schemas_without_executing_tools(self) -> None:
        tools = ToolManager()
        tools.register(
            "echo",
            "Echo text",
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            lambda text: text,
        )
        context = ContextManager()
        context.set_tool_schemas(tools.schemas())

        self.assertEqual(context.tool_schemas(), tools.schemas())
        self.assertFalse(hasattr(context, "execute_tool"))

    def test_streaming_assistant_message_can_be_appended_and_completed(self) -> None:
        context = ContextManager()

        index = context.start_assistant_stream()
        context.append_assistant_delta(index, "hel")
        context.append_assistant_delta(index, "lo")
        context.finish_assistant_stream(index)

        self.assertEqual(
            context.messages(),
            [Message("assistant", "hello", status="completed", token_usage=0)],
        )


class ToolManagerTest(unittest.TestCase):
    def test_exposes_schema_and_executes_tool(self) -> None:
        manager = ToolManager()
        manager.register(
            "echo",
            "Echo text",
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            lambda text: text,
        )

        self.assertEqual(
            manager.schemas(),
            [
                {
                    "type": "function",
                    "name": "echo",
                    "description": "Echo text",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                    "strict": True,
                }
            ],
        )
        self.assertEqual(
            manager.execute(ToolCall("call_1", "echo", {"text": "hello"})),
            ToolResult("hello"),
        )


class AgentTest(unittest.TestCase):
    def test_stream_uses_context_manager(self) -> None:
        calls = []

        class FakeClient:
            def stream(self, messages, tools=None, reasoning=None):
                calls.append((list(messages), reasoning))
                yield ConversationEvent(f"answer {len(calls)}")
                yield DoneEvent()

        context = ContextManager(system_prompt="system prompt")
        agent = Agent(FakeClient(), context)

        self.assertEqual(
            list(agent.stream("hello")),
            [
                ConversationEvent("answer 1"),
                UsageEvent(0),
                TurnCompleteEvent(1),
                LoopCompleteEvent("completed", 1),
            ],
        )
        self.assertEqual(
            list(agent.stream("continue")),
            [
                ConversationEvent("answer 2"),
                UsageEvent(0),
                TurnCompleteEvent(1),
                LoopCompleteEvent("completed", 1),
            ],
        )
        self.assertEqual(
            calls,
            [
                (
                    [Message("system", "system prompt"), Message("user", "hello")],
                    ReasoningConfig("low"),
                ),
                (
                    [
                        Message("system", "system prompt"),
                        Message("user", "hello"),
                        Message("assistant", "answer 1"),
                        Message("user", "continue"),
                    ],
                    ReasoningConfig("low"),
                ),
            ],
        )
        self.assertFalse(hasattr(agent, "messages"))

    def test_stream_executes_tool_calls_until_final_text(self) -> None:
        calls = []
        tool_manager = ToolManager()
        tool_manager.register(
            "echo",
            "Echo text",
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            lambda text: text,
        )

        class FakeClient:
            def stream(self, messages, tools=None, reasoning=None):
                calls.append((list(messages), tools))
                if len(calls) == 1:
                    yield ToolCallEvent(ToolCall("call_1", "echo", {"text": "hello"}))
                    yield DoneEvent(2)
                    return
                yield ConversationEvent("done")
                yield DoneEvent(3)

        context = ContextManager()
        agent = Agent(FakeClient(), context, tool_manager)

        self.assertEqual(
            list(agent.stream("use tool")),
            [
                ToolCallEvent(ToolCall("call_1", "echo", {"text": "hello"})),
                ToolResultEvent("call_1", "echo", "hello"),
                UsageEvent(2),
                TurnCompleteEvent(1),
                ConversationEvent("done"),
                UsageEvent(3),
                TurnCompleteEvent(2),
                LoopCompleteEvent("completed", 2),
            ],
        )
        self.assertEqual(
            context.messages(),
            [
                Message("user", "use tool"),
                Message("assistant", "", status="completed", token_usage=2),
                Message.tool_call("call_1", "echo", {"text": "hello"}),
                Message.tool_result("call_1", '{"content": "hello", "isError": false}'),
                Message("assistant", "done", status="completed", token_usage=3),
            ],
        )
        self.assertEqual(calls[0][1], tool_manager.schemas())

    def test_stream_batches_tool_calls_and_emits_results_as_completed(self) -> None:
        first = ToolCall("call_1", "first", {})
        second = ToolCall("call_2", "second", {})
        batches = []

        class CompletingOutOfOrder:
            def schemas(self):
                return []

            def execute_many(self, tool_calls):
                batches.append(list(tool_calls))
                yield second, ToolResult("second result")
                yield first, ToolResult("first result")

        class FakeClient:
            calls = 0

            def stream(self, messages, tools=None, reasoning=None):
                self.calls += 1
                if self.calls == 1:
                    yield ToolCallEvent(first)
                    yield ToolCallEvent(second)
                    yield DoneEvent(4)
                    return
                yield ConversationEvent("done")
                yield DoneEvent(1)

        context = ContextManager()
        events = list(
            Agent(FakeClient(), context, CompletingOutOfOrder()).stream("use tools")
        )

        self.assertEqual(batches, [[first, second]])
        self.assertEqual(
            events,
            [
                ToolCallEvent(first),
                ToolCallEvent(second),
                ToolResultEvent("call_2", "second", "second result"),
                ToolResultEvent("call_1", "first", "first result"),
                UsageEvent(4),
                TurnCompleteEvent(1),
                ConversationEvent("done"),
                UsageEvent(1),
                TurnCompleteEvent(2),
                LoopCompleteEvent("completed", 2),
            ],
        )
        self.assertEqual(
            context.messages(),
            [
                Message("user", "use tools"),
                Message("assistant", "", status="completed", token_usage=4),
                Message.tool_call("call_1", "first", {}),
                Message.tool_call("call_2", "second", {}),
                Message.tool_result(
                    "call_2",
                    '{"content": "second result", "isError": false}',
                ),
                Message.tool_result(
                    "call_1",
                    '{"content": "first result", "isError": false}',
                ),
                Message("assistant", "done", status="completed", token_usage=1),
            ],
        )

    def test_stream_returns_unknown_tool_errors_to_the_model(self) -> None:
        calls = []
        tool_manager = ToolManager()

        class FakeClient:
            def stream(self, messages, tools=None, reasoning=None):
                calls.append(list(messages))
                if len(calls) == 1:
                    yield ToolCallEvent(ToolCall("call_1", "missing", {}))
                    yield DoneEvent()
                    return
                yield ConversationEvent("recovered")
                yield DoneEvent()

        context = ContextManager()

        self.assertEqual(
            list(Agent(FakeClient(), context, tool_manager).stream("use tool")),
            [
                ToolCallEvent(ToolCall("call_1", "missing", {})),
                ToolResultEvent(
                    "call_1", "missing", "Unknown tool: missing", is_error=True
                ),
                UsageEvent(0),
                TurnCompleteEvent(1),
                ConversationEvent("recovered"),
                UsageEvent(0),
                TurnCompleteEvent(2),
                LoopCompleteEvent("completed", 2),
            ],
        )
        self.assertIn(
            Message.tool_result(
                "call_1",
                '{"content": "Unknown tool: missing", "isError": true}',
            ),
            calls[1],
        )

    def test_stream_adds_placeholder_and_appends_conversation_events(self) -> None:
        class FakeClient:
            def stream(self, messages, tools=None, reasoning=None):
                yield ConversationEvent("hel")
                yield ConversationEvent("lo")
                yield DoneEvent(token_usage=3)

        context = ContextManager()
        agent = Agent(FakeClient(), context)

        self.assertEqual(
            list(agent.stream("hello")),
            [
                ConversationEvent("hel"),
                ConversationEvent("lo"),
                UsageEvent(3),
                TurnCompleteEvent(1),
                LoopCompleteEvent("completed", 1),
            ],
        )
        self.assertEqual(
            context.messages(),
            [
                Message("user", "hello"),
                Message("assistant", "hello", status="completed", token_usage=3),
            ],
        )

    def test_stream_marks_assistant_message_error_on_error_event(self) -> None:
        class FakeClient:
            def stream(self, messages, tools=None, reasoning=None):
                yield ConversationEvent("hel")
                yield ErrorEvent("bad")

        context = ContextManager()
        agent = Agent(FakeClient(), context)

        self.assertEqual(
            list(agent.stream("hello")),
            [
                ConversationEvent("hel"),
                ErrorEvent("bad"),
                LoopCompleteEvent("error", 1),
            ],
        )
        self.assertEqual(
            context.messages(),
            [
                Message("user", "hello"),
                Message("assistant", "hel", status="error", token_usage=0),
            ],
        )

    def test_stream_stops_after_maximum_iterations(self) -> None:
        calls = 0
        tool_manager = ToolManager()
        tool_manager.register(
            "echo",
            "Echo text",
            {"type": "object", "properties": {}},
            lambda: "ok",
        )

        class FakeClient:
            def stream(self, messages, tools=None, reasoning=None):
                nonlocal calls
                calls += 1
                yield ToolCallEvent(ToolCall(f"call_{calls}", "echo", {}))
                yield DoneEvent(calls)

        events = list(
            Agent(
                FakeClient(),
                None,
                tool_manager,
                2,
            ).stream("keep going")
        )

        self.assertEqual(calls, 2)
        self.assertEqual(events[-3], TurnCompleteEvent(2))
        self.assertIsInstance(events[-2], ErrorEvent)
        self.assertEqual(events[-2].code, "max_iterations")
        self.assertEqual(events[-1], LoopCompleteEvent("max_iterations", 2))

    def test_maximum_iterations_must_be_between_one_and_fifty(self) -> None:
        class FakeClient:
            def stream(self, messages, tools=None, reasoning=None):
                return iter(())

        self.assertEqual(getattr(Agent(FakeClient()), "max_iterations", None), 50)
        for maximum in (0, 51):
            with self.subTest(maximum=maximum):
                with self.assertRaisesRegex(ValueError, "between 1 and 50"):
                    Agent(FakeClient(), None, None, maximum)

    def test_keyboard_interrupt_cancels_the_loop(self) -> None:
        class FakeClient:
            def stream(self, messages, tools=None, reasoning=None):
                yield ConversationEvent("partial")
                raise KeyboardInterrupt

        context = ContextManager()

        try:
            events = list(Agent(FakeClient(), context).stream("hello"))
        except KeyboardInterrupt:
            events = None

        self.assertEqual(
            events,
            [
                ConversationEvent("partial"),
                ErrorEvent("interrupted", "interrupted"),
                LoopCompleteEvent("cancelled", 1),
            ],
        )
        self.assertEqual(context.messages()[-1].status, "error")

    def test_keyboard_interrupt_during_tool_does_not_leave_an_orphan_call(
        self,
    ) -> None:
        tool_manager = ToolManager()
        tool_manager.register(
            "interrupt",
            "Interrupt",
            {"type": "object", "properties": {}},
            lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
        )

        class FakeClient:
            def stream(self, messages, tools=None, reasoning=None):
                yield ToolCallEvent(ToolCall("call_1", "interrupt", {}))
                yield DoneEvent()

        context = ContextManager()
        events = list(Agent(FakeClient(), context, tool_manager).stream("stop"))

        self.assertEqual(
            events,
            [
                ToolCallEvent(ToolCall("call_1", "interrupt", {})),
                ErrorEvent("interrupted", "interrupted"),
                LoopCompleteEvent("cancelled", 1),
            ],
        )
        self.assertEqual(
            context.messages(),
            [
                Message("user", "stop"),
                Message("assistant", "", status="error"),
            ],
        )

    def test_keyboard_interrupt_cancels_pending_safe_tools_and_discards_results(
        self,
    ) -> None:
        tool_manager = ToolManager()
        slow_started = threading.Event()
        release_slow = threading.Event()
        slow_finished = threading.Event()
        release_pending = threading.Event()
        cancelled_started = threading.Event()

        def slow():
            slow_started.set()
            release_slow.wait(5)
            slow_finished.set()
            return "slow"

        def interrupt():
            slow_started.wait(1)
            raise KeyboardInterrupt

        def pending_running():
            release_pending.wait(5)
            return "pending"

        def pending_cancelled():
            cancelled_started.set()
            return "cancelled"

        for name, handler in (
            ("slow", slow),
            ("interrupt", interrupt),
            ("pending_running", pending_running),
            ("pending_cancelled", pending_cancelled),
        ):
            tool_manager.register(
                name,
                name,
                {"type": "object", "properties": {}},
                handler,
                is_concurrency_safe=True,
            )

        class FakeClient:
            def stream(self, messages, tools=None, reasoning=None):
                for name in (
                    "slow",
                    "interrupt",
                    "pending_running",
                    "pending_cancelled",
                ):
                    yield ToolCallEvent(ToolCall(f"call_{name}", name, {}))
                yield DoneEvent()

        context = ContextManager()
        try:
            with patch(
                "duckduckcode.tools.tool.ThreadPoolExecutor",
                side_effect=lambda: ThreadPoolExecutor(max_workers=2),
            ):
                events = list(
                    Agent(FakeClient(), context, tool_manager).stream("cancel")
                )
        finally:
            release_slow.set()
            release_pending.set()

        self.assertEqual(
            events,
            [
                ToolCallEvent(ToolCall("call_slow", "slow", {})),
                ToolCallEvent(ToolCall("call_interrupt", "interrupt", {})),
                ToolCallEvent(ToolCall("call_pending_running", "pending_running", {})),
                ToolCallEvent(
                    ToolCall("call_pending_cancelled", "pending_cancelled", {})
                ),
                ErrorEvent("interrupted", "interrupted"),
                LoopCompleteEvent("cancelled", 1),
            ],
        )
        self.assertFalse(cancelled_started.is_set())
        self.assertTrue(slow_finished.wait(1))
        self.assertEqual(
            context.messages(),
            [
                Message("user", "cancel"),
                Message("assistant", "", status="error"),
            ],
        )

    def test_closing_stream_preserves_partial_assistant_message(self) -> None:
        class FakeClient:
            def stream(self, messages, tools=None, reasoning=None):
                yield ConversationEvent("partial")
                while True:
                    yield ConversationEvent("more")

        context = ContextManager()
        stream = Agent(FakeClient(), context).stream("hello")

        self.assertEqual(next(stream), ConversationEvent("partial"))
        stream.close()

        self.assertEqual(
            context.messages(),
            [
                Message("user", "hello"),
                Message("assistant", "partial", status="error", token_usage=0),
            ],
        )


if __name__ == "__main__":
    unittest.main()
