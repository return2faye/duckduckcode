from __future__ import annotations

from openai import OpenAI

from .client import Client, ClientResponse
from .context import Message, ReasoningConfig
from .serialize import MessageDeserializer, MessageSerializer, OpenAIResponsesDeserializer, OpenAIResponsesSerializer


class OpenAIClient(Client):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        serializer: MessageSerializer | None = None,
        deserializer: MessageDeserializer | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        self.model = model or "o4-mini"
        self.serializer = serializer or OpenAIResponsesSerializer()
        self.deserializer = deserializer or OpenAIResponsesDeserializer()
        self._client = OpenAI(api_key=api_key)

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        reasoning: ReasoningConfig | None = None,
    ) -> ClientResponse:
        payload = self.serializer.serialize(
            messages,
            reasoning or ReasoningConfig(),
        )
        if tools:
            payload["tools"] = tools
        response = self._client.responses.create(
            model=self.model,
            **payload,
        )
        return self.deserializer.deserialize(response)
