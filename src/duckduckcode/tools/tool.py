from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

Validator = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params: dict[str, Any]
    handler: Callable[..., Any]
    validator: Validator
    is_read_only: bool = False
    is_dangerous: bool = False
    is_concurrency_safe: bool = False
    category: str = "general"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.params,
            "strict": True,
        }


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False

    def to_model_output(self) -> str:
        return json.dumps({"content": self.content, "isError": self.is_error})


def create_tool(
    name: str,
    description: str,
    params: dict[str, Any],
    handler: Callable[..., Any],
    validator: Validator,
    *,
    is_read_only: bool = False,
    is_dangerous: bool = False,
    is_concurrency_safe: bool = False,
    category: str = "general",
) -> Tool:
    return Tool(
        name,
        description,
        params,
        handler,
        validator,
        is_read_only,
        is_dangerous,
        is_concurrency_safe,
        category,
    )


class ToolManager:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: Tool | str,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        handler: Callable[..., Any] | None = None,
        *,
        validator: Validator | None = None,
        is_read_only: bool = False,
        is_dangerous: bool = False,
        is_concurrency_safe: bool = False,
        category: str = "general",
    ) -> None:
        if isinstance(name, Tool):
            if (
                description is not None
                or parameters is not None
                or handler is not None
                or validator is not None
                or is_read_only
                or is_dangerous
                or is_concurrency_safe
                or category != "general"
            ):
                raise TypeError("A Tool cannot be combined with registration arguments")
            registered = name
        else:
            if description is None or parameters is None or handler is None:
                raise TypeError(
                    "name, description, parameters, and handler are required"
                )
            registered = create_tool(
                name,
                description,
                parameters,
                handler,
                validator if validator is not None else _identity,
                is_read_only=is_read_only,
                is_dangerous=is_dangerous,
                is_concurrency_safe=is_concurrency_safe,
                category=category,
            )
        self._tools[registered.name] = registered

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        if tool_call.name not in self._tools:
            return ToolResult(f"Unknown tool: {tool_call.name}", is_error=True)

        tool = self._tools[tool_call.name]
        try:
            arguments = tool.validator(dict(tool_call.arguments))
            if not isinstance(arguments, dict):
                raise TypeError("Tool validator must return a dictionary")
            result = tool.handler(**arguments)
            if isinstance(result, ToolResult):
                return result
            return ToolResult(result if isinstance(result, str) else json.dumps(result))
        except Exception as exc:
            return ToolResult(str(exc), is_error=True)


def _identity(arguments: dict[str, Any]) -> dict[str, Any]:
    return arguments
