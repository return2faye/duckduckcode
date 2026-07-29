from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import Any

from ...core.client import ClientResponse
from ...core.context import Message, ReasoningConfig
from ...tools.tool import ToolCall


class MessageSerializer(ABC):
    @abstractmethod
    def serialize(
        self,
        messages: list[Message],
        reasoning: ReasoningConfig,
    ) -> dict[str, Any]:
        raise NotImplementedError


class MessageDeserializer(ABC):
    @abstractmethod
    def deserialize(self, response: Any) -> ClientResponse:
        raise NotImplementedError


class OpenAIResponsesSerializer(MessageSerializer):
    def serialize(
        self,
        messages: list[Message],
        reasoning: ReasoningConfig,
    ) -> dict[str, Any]:
        return {
            "input": [message.to_openai() for message in messages],
            "reasoning": {"effort": reasoning.effort},
        }


class OpenAIResponsesDeserializer(MessageDeserializer):
    def deserialize(self, response: Any) -> ClientResponse:
        return ClientResponse(
            response.output_text,
            [_to_tool_call(item) for item in response.output if _is_tool_call(item)],
        )


def _is_tool_call(item: Any) -> bool:
    return getattr(item, "type", None) == "function_call"


def _to_tool_call(item: Any) -> ToolCall:
    return ToolCall(item.call_id, item.name, json.loads(item.arguments or "{}"))
