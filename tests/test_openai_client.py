from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from duckduckcode.agent import Agent
from duckduckcode.client import ClientResponse
from duckduckcode.context import ContextManager, Message, ReasoningConfig
from duckduckcode.openai_client import OpenAIClient
from duckduckcode.serialize import OpenAIResponsesDeserializer, OpenAIResponsesSerializer
from duckduckcode.tool import ToolCall, ToolManager


class FakeResponses:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return type("Response", (), {"output_text": "ok", "output": []})()


class OpenAIClientTest(unittest.TestCase):
    def test_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                OpenAIClient()

    def test_generate_uses_injected_serializer(self) -> None:
        fake_responses = FakeResponses()
        serializer = type(
            "Serializer",
            (),
            {
                "serialize": lambda self, messages, reasoning: {
                    "reasoning": {"effort": reasoning.effort},
                    "input": [{"role": item.role, "content": item.content} for item in messages],
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
        self.assertEqual(client.generate(messages).text, "ok")
        self.assertEqual(fake_responses.kwargs["model"], "test-model")
        self.assertEqual(fake_responses.kwargs["input"], [{"role": "user", "content": "hello"}])
        self.assertEqual(fake_responses.kwargs["reasoning"], {"effort": "low"})
        self.assertFalse(hasattr(client, "chat_once"))

    def test_generate_uses_injected_deserializer(self) -> None:
        fake_responses = FakeResponses()
        deserializer = type(
            "Deserializer",
            (),
            {"deserialize": lambda self, response: ClientResponse(f"custom:{response.output_text}")},
        )()
        client = OpenAIClient(api_key="test-key", deserializer=deserializer)
        client._client = type("Client", (), {"responses": fake_responses})()

        self.assertEqual(client.generate([Message("user", "hello")]).text, "custom:ok")

    def test_defaults_to_o4_mini_with_low_reasoning(self) -> None:
        fake_responses = FakeResponses()
        client = OpenAIClient(api_key="test-key")
        client._client = type("Client", (), {"responses": fake_responses})()

        client.generate([Message("user", "hello")])

        self.assertEqual(fake_responses.kwargs["model"], "o4-mini")
        self.assertEqual(fake_responses.kwargs["reasoning"], {"effort": "low"})

    def test_generate_passes_tools_and_returns_tool_calls(self) -> None:
        fake_responses = FakeResponses()
        fake_responses.create = lambda **kwargs: (
            setattr(fake_responses, "kwargs", kwargs)
            or type(
                "Response",
                (),
                {
                    "output_text": "",
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
        )
        client = OpenAIClient(api_key="test-key", model="test-model")
        client._client = type("Client", (), {"responses": fake_responses})()
        tools = [{"type": "function", "name": "read_file", "parameters": {}, "strict": True}]

        response = client.generate([Message("user", "read README")], tools=tools)

        self.assertEqual(fake_responses.kwargs["tools"], tools)
        self.assertEqual(response.tool_calls, [ToolCall("call_1", "read_file", {"path": "README.md"})])

    def test_openai_serializer_puts_system_prompt_first(self) -> None:
        payload = OpenAIResponsesSerializer().serialize(
            [Message("system", "system prompt"), Message("user", "hello"), Message("assistant", "hi")],
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
                {"type": "function_call_output", "call_id": "call_1", "output": "file contents"},
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
            ClientResponse("done", [ToolCall("call_1", "read_file", {"path": "README.md"})]),
        )


class ContextManagerTest(unittest.TestCase):
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
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            lambda text: text,
        )
        context = ContextManager()
        context.set_tool_schemas(tools.schemas())

        self.assertEqual(context.tool_schemas(), tools.schemas())
        self.assertFalse(hasattr(context, "execute_tool"))


class ToolManagerTest(unittest.TestCase):
    def test_exposes_schema_and_executes_tool(self) -> None:
        manager = ToolManager()
        manager.register(
            "echo",
            "Echo text",
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
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
        self.assertEqual(manager.execute(ToolCall("call_1", "echo", {"text": "hello"})), "hello")


class AgentTest(unittest.TestCase):
    def test_ask_uses_context_manager(self) -> None:
        calls = []

        class FakeClient:
            def generate(self, messages, tools=None, reasoning=None):
                calls.append((list(messages), reasoning))
                return ClientResponse(f"answer {len(calls)}")

        context = ContextManager(system_prompt="system prompt")
        agent = Agent(FakeClient(), context)

        self.assertEqual(agent.ask("hello"), "answer 1")
        self.assertEqual(agent.ask("continue"), "answer 2")
        self.assertEqual(
            calls,
            [
                ([Message("system", "system prompt"), Message("user", "hello")], ReasoningConfig("low")),
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

    def test_ask_executes_tool_calls_until_final_text(self) -> None:
        calls = []
        tool_manager = ToolManager()
        tool_manager.register(
            "echo",
            "Echo text",
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            lambda text: text,
        )

        class FakeClient:
            def generate(self, messages, tools=None, reasoning=None):
                calls.append((list(messages), tools))
                if len(calls) == 1:
                    return ClientResponse(tool_calls=[ToolCall("call_1", "echo", {"text": "hello"})])
                return ClientResponse("done")

        context = ContextManager()
        agent = Agent(FakeClient(), context, tool_manager)

        self.assertEqual(agent.ask("use tool"), "done")
        self.assertEqual(
            context.messages(),
            [
                Message("user", "use tool"),
                Message.tool_call("call_1", "echo", {"text": "hello"}),
                Message.tool_result("call_1", "hello"),
                Message("assistant", "done"),
            ],
        )
        self.assertEqual(calls[0][1], tool_manager.schemas())


if __name__ == "__main__":
    unittest.main()
