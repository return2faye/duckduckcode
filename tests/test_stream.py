from __future__ import annotations

import unittest

from duckduckcode.client import ClientResponse
from duckduckcode.context import Message, ReasoningConfig
from duckduckcode.event import ConversationEvent, DoneEvent, ErrorEvent, ToolCallEvent
from duckduckcode.openai_client import OpenAIClient
from duckduckcode.stream import OpenAIStreamEventHandler, OpenAIStreamEventParser
from duckduckcode.tool import ToolCall


class OpenAIStreamTest(unittest.TestCase):
    def test_parser_maps_text_tool_error_and_done_events(self) -> None:
        parser = OpenAIStreamEventParser()

        self.assertEqual(
            parser.parse(
                type(
                    "Event", (), {"type": "response.output_text.delta", "delta": "hi"}
                )()
            ),
            ConversationEvent("hi"),
        )
        self.assertEqual(
            parser.parse(
                type(
                    "Event",
                    (),
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": "call_1",
                        "name": "read_file",
                        "arguments": '{"path": "README.md"}',
                    },
                )()
            ),
            ToolCallEvent(ToolCall("call_1", "read_file", {"path": "README.md"})),
        )
        self.assertEqual(
            parser.parse(
                type(
                    "Event",
                    (),
                    {"type": "error", "message": "bad", "code": "rate_limit"},
                )()
            ),
            ErrorEvent("bad", "rate_limit"),
        )
        self.assertEqual(
            parser.parse(
                type(
                    "Event",
                    (),
                    {
                        "type": "response.completed",
                        "response": type(
                            "Response",
                            (),
                            {"usage": type("Usage", (), {"total_tokens": 12})()},
                        )(),
                    },
                )()
            ),
            DoneEvent(token_usage=12),
        )

    def test_handler_filters_unknown_events(self) -> None:
        events = [
            type("Event", (), {"type": "response.created"})(),
            type("Event", (), {"type": "response.output_text.delta", "delta": "hi"})(),
        ]

        self.assertEqual(
            list(OpenAIStreamEventHandler().handle(events)), [ConversationEvent("hi")]
        )

    def test_openai_client_stream_uses_handler_and_stream_true(self) -> None:
        class FakeResponses:
            def __init__(self) -> None:
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return [
                    type(
                        "Event",
                        (),
                        {"type": "response.output_text.delta", "delta": "hi"},
                    )()
                ]

        class FakeHandler:
            def handle(self, events):
                for event in events:
                    yield ConversationEvent(event.delta)

        fake_responses = FakeResponses()
        client = OpenAIClient(api_key="test-key", event_handler=FakeHandler())
        client._client = type("Client", (), {"responses": fake_responses})()

        self.assertEqual(
            list(
                client.stream(
                    [Message("user", "hello")], reasoning=ReasoningConfig("low")
                )
            ),
            [ConversationEvent("hi")],
        )
        self.assertTrue(fake_responses.kwargs["stream"])
        self.assertEqual(fake_responses.kwargs["reasoning"], {"effort": "low"})


if __name__ == "__main__":
    unittest.main()
