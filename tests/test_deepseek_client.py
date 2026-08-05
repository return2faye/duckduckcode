from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from duckduckcode.config import Config, ModelConfig
from duckduckcode.core.context import Message, ReasoningConfig
from duckduckcode.core.event import ConversationEvent, DoneEvent, ToolCallEvent
from duckduckcode.providers import create_client
from duckduckcode.providers.deepseek.client import DeepSeekClient
from duckduckcode.providers.deepseek.serialize import (
    DeepSeekChatSerializer,
    serialize_tools,
)
from duckduckcode.providers.deepseek.stream import DeepSeekStreamEventHandler
from duckduckcode.tools.tool import ToolCall


def chunk(*, content=None, reasoning_content=None, tool_calls=None, usage=None):
    choices = []
    if content is not None or reasoning_content is not None or tool_calls is not None:
        choices = [
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                )
            )
        ]
    return SimpleNamespace(choices=choices, usage=usage)


def tool_fragment(index, *, call_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class Events(list):
    closed = False

    def close(self):
        self.closed = True


class DeepSeekSerializerTest(unittest.TestCase):
    def test_serializes_messages_and_groups_one_assistant_tool_response(self) -> None:
        messages = [
            Message("system", "system prompt"),
            Message("user", "inspect"),
            Message("assistant", "I will inspect."),
            Message.tool_call("call_1", "ReadFile", {"path": "README.md"}),
            Message.tool_call("call_2", "Glob", {"pattern": "*.py"}),
            Message.tool_result("call_1", "read output"),
            Message.tool_result("call_2", "glob output"),
            Message("assistant", "done"),
        ]

        self.assertEqual(
            DeepSeekChatSerializer().serialize(messages),
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "inspect"},
                {
                    "role": "assistant",
                    "content": "I will inspect.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "ReadFile",
                                "arguments": '{"path": "README.md"}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "Glob",
                                "arguments": '{"pattern": "*.py"}',
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "content": "read output",
                    "tool_call_id": "call_1",
                },
                {
                    "role": "tool",
                    "content": "glob output",
                    "tool_call_id": "call_2",
                },
                {"role": "assistant", "content": "done"},
            ],
        )

    def test_converts_responses_tools_to_chat_functions(self) -> None:
        parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        }

        self.assertEqual(
            serialize_tools(
                [
                    {
                        "type": "function",
                        "name": "ReadFile",
                        "description": "Read a file.",
                        "strict": True,
                        "parameters": parameters,
                    }
                ]
            ),
            [
                {
                    "type": "function",
                    "function": {
                        "name": "ReadFile",
                        "description": "Read a file.",
                        "parameters": parameters,
                    },
                }
            ],
        )


class DeepSeekStreamTest(unittest.TestCase):
    def test_emits_content_fragmented_tool_calls_and_usage_without_reasoning(
        self,
    ) -> None:
        events = Events(
            [
                chunk(reasoning_content="private reasoning"),
                chunk(content="hel"),
                chunk(content="lo"),
                chunk(
                    tool_calls=[
                        tool_fragment(
                            0,
                            call_id="call_1",
                            name="ReadFile",
                            arguments='{"path":"README.md",',
                        )
                    ]
                ),
                chunk(
                    tool_calls=[
                        tool_fragment(
                            0,
                            arguments='"offset":1,"limit":20}',
                        )
                    ]
                ),
                chunk(usage=SimpleNamespace(total_tokens=14)),
            ]
        )

        self.assertEqual(
            list(DeepSeekStreamEventHandler().handle(events)),
            [
                ConversationEvent("hel"),
                ConversationEvent("lo"),
                ToolCallEvent(
                    ToolCall(
                        "call_1",
                        "ReadFile",
                        {"path": "README.md", "offset": 1, "limit": 20},
                    )
                ),
                DoneEvent(14),
            ],
        )
        self.assertTrue(events.closed)


class DeepSeekClientTest(unittest.TestCase):
    def test_factory_selects_the_configured_provider(self) -> None:
        config = Config(
            "openai-key",
            deepseek_api_key="deepseek-key",
            deepseek_base_url="https://deepseek.example/v1",
        )
        openai = object()
        deepseek = object()
        with (
            patch(
                "duckduckcode.providers.OpenAIClient", return_value=openai
            ) as openai_client,
            patch(
                "duckduckcode.providers.DeepSeekClient", return_value=deepseek
            ) as deepseek_client,
        ):
            self.assertIs(
                create_client(config, ModelConfig("openai", "openai-model")),
                openai,
            )
            self.assertIs(
                create_client(
                    config,
                    ModelConfig("deepseek", "configured-model"),
                    model="override-model",
                ),
                deepseek,
            )

        self.assertEqual(openai_client.call_args.kwargs["api_key"], "openai-key")
        self.assertEqual(openai_client.call_args.kwargs["model"], "openai-model")
        self.assertEqual(
            deepseek_client.call_args.kwargs,
            {
                "api_key": "deepseek-key",
                "model": "override-model",
                "base_url": "https://deepseek.example/v1",
                "langsmith_tracing": False,
                "langsmith_api_key": None,
                "langsmith_project": "duckduckcode",
            },
        )

    def test_stream_sends_chat_completion_payload(self) -> None:
        completions = SimpleNamespace()
        completions.create = lambda **kwargs: setattr(
            completions, "kwargs", kwargs
        ) or Events([chunk(content="ok"), chunk(usage=SimpleNamespace(total_tokens=3))])
        raw = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
            close=lambda: setattr(raw, "closed", True),
        )
        tool = {
            "type": "function",
            "name": "ReadFile",
            "description": "Read a file.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }
        with patch(
            "duckduckcode.providers.deepseek.client.OpenAI", return_value=raw
        ) as constructor:
            client = DeepSeekClient(
                api_key="deepseek-key",
                model="deepseek-model",
                base_url="https://deepseek.example/v1",
            )
            result = list(
                client.stream(
                    [Message("user", "hello")],
                    tools=[tool],
                    reasoning=ReasoningConfig("high"),
                    max_output_tokens=123,
                )
            )
            client.close()

        self.assertEqual(result, [ConversationEvent("ok"), DoneEvent(3)])
        constructor.assert_called_once_with(
            api_key="deepseek-key", base_url="https://deepseek.example/v1"
        )
        self.assertEqual(completions.kwargs["model"], "deepseek-model")
        self.assertEqual(
            completions.kwargs["messages"], [{"role": "user", "content": "hello"}]
        )
        self.assertEqual(completions.kwargs["reasoning_effort"], "high")
        self.assertEqual(completions.kwargs["max_tokens"], 123)
        self.assertEqual(completions.kwargs["stream_options"], {"include_usage": True})
        self.assertNotIn("strict", completions.kwargs["tools"][0]["function"])
        self.assertTrue(raw.closed)

    def test_requires_api_key(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
            DeepSeekClient()


if __name__ == "__main__":
    unittest.main()
