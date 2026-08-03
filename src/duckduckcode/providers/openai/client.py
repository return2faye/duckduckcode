from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai import OpenAI
from langsmith import Client as LangSmithClient
from langsmith.wrappers import wrap_openai

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
        langsmith_tracing: bool = False,
        langsmith_api_key: str | None = None,
        langsmith_project: str = "duckduckcode",
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        self.model = model or "o4-mini"
        self.serializer = serializer or OpenAIResponsesSerializer()
        self.deserializer = deserializer or OpenAIResponsesDeserializer()
        self.event_handler = event_handler or OpenAIStreamEventHandler()
        self._langsmith_client = None
        self._client = OpenAI(api_key=api_key)
        if langsmith_tracing:
            if not langsmith_api_key:
                raise RuntimeError(
                    "LANGSMITH_API_KEY is required when LANGSMITH_TRACING is true"
                )
            self._langsmith_client = LangSmithClient(api_key=langsmith_api_key)
            self._client = wrap_openai(
                self._client,
                tracing_extra={
                    "client": self._langsmith_client,
                    "project_name": langsmith_project,
                    "enabled": True,
                },
            )

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        reasoning: ReasoningConfig | None = None,
        max_output_tokens: int | None = None,
    ) -> Iterator[StreamEvent]:
        payload = self.serializer.serialize(
            messages,
            reasoning or ReasoningConfig(),
        )
        if tools:
            payload["tools"] = tools
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        events = self._client.responses.create(
            model=self.model,
            stream=True,
            **payload,
        )
        return self.event_handler.handle(events)

    def close(self) -> None:
        try:
            self._client.close()
        finally:
            if self._langsmith_client is not None:
                self._langsmith_client.flush()
