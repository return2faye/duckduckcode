from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langsmith import Client as LangSmithClient
from langsmith.wrappers import wrap_openai
from openai import OpenAI

from ...core.client import Client
from ...core.context import Message, ReasoningConfig
from ...core.event import StreamEvent
from .serialize import DeepSeekChatSerializer, serialize_tools
from .stream import DeepSeekStreamEventHandler


class DeepSeekClient(Client):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://api.deepseek.com",
        serializer: DeepSeekChatSerializer | None = None,
        event_handler: DeepSeekStreamEventHandler | None = None,
        langsmith_tracing: bool = False,
        langsmith_api_key: str | None = None,
        langsmith_project: str = "duckduckcode",
    ) -> None:
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required")
        self.model = model or "deepseek-v4-pro"
        self.serializer = serializer or DeepSeekChatSerializer()
        self.event_handler = event_handler or DeepSeekStreamEventHandler()
        self._langsmith_client = None
        self._client = OpenAI(api_key=api_key, base_url=base_url)
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
        effort = (reasoning or ReasoningConfig()).effort
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.serializer.serialize(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if effort == "none":
            payload["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            payload["reasoning_effort"] = effort
        if tools:
            payload["tools"] = serialize_tools(tools)
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        events = self._client.chat.completions.create(**payload)
        return self.event_handler.handle(events)

    def close(self) -> None:
        try:
            self._client.close()
        finally:
            if self._langsmith_client is not None:
                self._langsmith_client.flush()
