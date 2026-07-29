from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .context import Message, ReasoningConfig
from .tool import ToolCall


@dataclass(frozen=True)
class ClientResponse:
    text: str = ""
    tool_calls: list[ToolCall] | None = None

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            object.__setattr__(self, "tool_calls", [])


class Client(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        reasoning: ReasoningConfig | None = None,
    ) -> ClientResponse:
        raise NotImplementedError
