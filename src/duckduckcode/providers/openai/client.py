from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from ...core.client import Client, ClientResponse
from ...core.context import Message, ReasoningConfig
from ...core.event import StreamEvent
from .serialize import (
    MessageDeserializer,
    MessageSerializer,
    OpenAIResponsesDeserializer,
    OpenAIResponsesSerializer,
)
from .stream import OpenAIStreamEventHandler


class OpenAIClient(Client):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        serializer: MessageSerializer | None = None,
        deserializer: MessageDeserializer | None = None,
        event_handler: OpenAIStreamEventHandler | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        self.model = model or "o4-mini"
        self.serializer = serializer or OpenAIResponsesSerializer()
        self.deserializer = deserializer or OpenAIResponsesDeserializer()
        self.event_handler = event_handler or OpenAIStreamEventHandler()
        self._client = OpenAI(api_key=api_key)

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        reasoning: ReasoningConfig | None = None,
    ) -> Iterator[StreamEvent]:
        payload = self.serializer.serialize(
            messages,
            reasoning or ReasoningConfig(),
        )
        if tools:
            payload["tools"] = tools
        events = self._client.responses.create(
            model=self.model,
            stream=True,
            **payload,
        )
        return self.event_handler.handle(events)
