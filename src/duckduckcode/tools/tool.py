from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from queue import Queue
import re
from typing import Any, Callable, Protocol

Validator = Callable[[dict[str, Any]], dict[str, Any]]
SUBAGENT_SLUG_RE = re.compile(r"^[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*$")
MAX_SUBAGENT_SLUG_LENGTH = 48


class QuerySource(str, Enum):
    USER = "user"
    SUBAGENT = "subagent"


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False

    def to_model_output(self) -> str:
        return json.dumps({"content": self.content, "isError": self.is_error})


class Tool(Protocol):
    name: str
    description: str
    params: dict[str, Any]
    is_read_only: bool
    is_dangerous: bool
    is_concurrency_safe: bool
    category: str
    strict: bool

    def schema(self) -> dict[str, Any]: ...

    def execute(self, arguments: dict[str, Any]) -> ToolResult: ...

    def permission_content(self, arguments: dict[str, Any]) -> str | None: ...


@dataclass(frozen=True)
class BuiltinTool:
    name: str
    description: str
    params: dict[str, Any]
    handler: Callable[..., Any]
    validator: Validator
    is_read_only: bool = False
    is_dangerous: bool = False
    is_concurrency_safe: bool = False
    category: str = "general"
    strict: bool = True

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.params,
            "strict": self.strict,
        }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        normalized = self.validator(arguments)
        if not isinstance(normalized, dict):
            raise TypeError("Tool validator must return a dictionary")
        result = self.handler(**normalized)
        if isinstance(result, ToolResult):
            return result
        return ToolResult(result if isinstance(result, str) else json.dumps(result))

    def permission_content(self, arguments: dict[str, Any]) -> str | None:
        return None


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
    strict: bool = True,
) -> BuiltinTool:
    return BuiltinTool(
        name,
        description,
        params,
        handler,
        validator,
        is_read_only,
        is_dangerous,
        is_concurrency_safe,
        category,
        strict,
    )


def create_exit_plan_mode_tool() -> Tool:
    return create_tool(
        "ExitPlanMode",
        "Call ExitPlanMode only when Plan Mode is active and the final plan has "
        "been written to the Plan file. It asks the user to approve execution or "
        "provide feedback. Do not call it for ordinary questions or before the "
        "plan is ready.",
        {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        lambda: ToolResult(
            "ExitPlanMode is handled by the agent while Plan Mode is active.",
            is_error=True,
        ),
        _identity,
        category="mode",
    )


LOAD_SKILL_PARAMS = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Exact discovered skill name.",
        },
        "task": {
            "type": "string",
            "description": "Current goal or concrete task for the skill.",
        },
    },
    "required": ["name", "task"],
    "additionalProperties": False,
}


def create_load_skill_tool(handler: Callable[..., Any]) -> Tool:
    return create_tool(
        "LoadSkill",
        "Load one discovered Skill by exact name for the current user turn. "
        "Call this before solving when the skill catalog matches the user's intent.",
        LOAD_SKILL_PARAMS,
        handler,
        validate_load_skill_arguments,
        is_read_only=True,
        category="search",
    )


def create_agent_tool(
    definition_types: list[str] | tuple[str, ...] | Mapping[str, str],
    handler: Callable[..., Any],
) -> Tool:
    descriptions = (
        dict(definition_types) if isinstance(definition_types, Mapping) else {}
    )
    types = list(dict.fromkeys(definition_types))
    params = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Complete subtask."},
            "description": {
                "type": "string",
                "description": "Short task description.",
            },
            "subagent_type": {
                "type": ["string", "null"],
                "enum": [*types, None],
            },
            "model": {"type": ["string", "null"]},
            "run_in_background": {"type": "boolean"},
            "name": {
                "type": ["string", "null"],
                "description": (
                    "Optional ASCII slug up to 48 characters; separators '.', '_', "
                    "and '-' must appear singly between letters or digits."
                ),
            },
            "isolation": {
                "type": "boolean",
                "description": (
                    "For a writable fork, use a persistent Git worktree and return "
                    "its cumulative patch without applying it. Definitions use a "
                    "read-only snapshot."
                ),
            },
        },
        "required": [
            "prompt",
            "description",
            "subagent_type",
            "model",
            "run_in_background",
            "name",
            "isolation",
        ],
        "additionalProperties": False,
    }

    guidance = "".join(
        f"\n- {name}: {descriptions[name]}" for name in types if name in descriptions
    )
    return create_tool(
        "Agent",
        "Run one non-interactive Definition-based subagent or fork this conversation. "
        "Fork tasks always run in the background. Available Definition types:"
        + guidance,
        params,
        handler,
        lambda arguments: validate_agent_arguments(arguments, types),
        is_read_only=True,
        category="agent",
    )


class ToolManager:
    def __init__(self, source: QuerySource = QuerySource.USER) -> None:
        self._tools: dict[str, Tool] = {}
        self.source = source
        self._dirty = True

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
        strict: bool = True,
    ) -> None:
        if not isinstance(name, str):
            if (
                description is not None
                or parameters is not None
                or handler is not None
                or validator is not None
                or is_read_only
                or is_dangerous
                or is_concurrency_safe
                or category != "general"
                or not strict
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
                strict=strict,
            )
        self._tools[registered.name] = registered
        self._dirty = True

    def unregister(self, name: str) -> Tool | None:
        removed = self._tools.pop(name, None)
        if removed is not None:
            self._dirty = True
        return removed

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_clean(self) -> None:
        self._dirty = False

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, tool_call: ToolCall) -> ToolResult:
        if self.source == QuerySource.SUBAGENT and tool_call.name == "Agent":
            return ToolResult("Subagents cannot invoke the Agent tool.", is_error=True)
        if tool_call.name not in self._tools:
            return ToolResult(f"Unknown tool: {tool_call.name}", is_error=True)

        tool = self._tools[tool_call.name]
        try:
            result = tool.execute(dict(tool_call.arguments))
            if not isinstance(result, ToolResult):
                raise TypeError("Tool.execute() must return ToolResult")
            return result
        except Exception as exc:
            return ToolResult(str(exc), is_error=True)

    def execute_many(
        self, tool_calls: list[ToolCall]
    ) -> Iterator[tuple[ToolCall, ToolResult]]:
        safe_batch: list[ToolCall] = []
        for tool_call in tool_calls:
            tool = self._tools.get(tool_call.name)
            if tool is not None and tool.is_concurrency_safe:
                safe_batch.append(tool_call)
                continue
            yield from self._execute_safe_batch(safe_batch)
            safe_batch.clear()
            yield tool_call, self.execute(tool_call)
        yield from self._execute_safe_batch(safe_batch)

    def _execute_safe_batch(
        self, tool_calls: list[ToolCall]
    ) -> Iterator[tuple[ToolCall, ToolResult]]:
        if len(tool_calls) == 1:
            yield tool_calls[0], self.execute(tool_calls[0])
            return
        if not tool_calls:
            return

        completed: Queue[tuple[ToolCall, ToolResult | BaseException]] = Queue()

        def execute(tool_call: ToolCall) -> None:
            try:
                result: ToolResult | BaseException = self.execute(tool_call)
            except BaseException as exc:
                result = exc
            completed.put((tool_call, result))

        executor = ThreadPoolExecutor()
        futures = [executor.submit(execute, tool_call) for tool_call in tool_calls]
        try:
            for _ in futures:
                tool_call, result = completed.get()
                if isinstance(result, BaseException):
                    raise result
                yield tool_call, result
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


def _identity(arguments: dict[str, Any]) -> dict[str, Any]:
    return arguments


def validate_agent_arguments(
    arguments: dict[str, Any], definition_types: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    expected = {
        "prompt",
        "description",
        "subagent_type",
        "model",
        "run_in_background",
        "name",
        "isolation",
    }
    missing = sorted(expected - arguments.keys())
    unsupported = sorted(arguments.keys() - expected)
    if missing or unsupported:
        detail = (
            "missing: " + ", ".join(missing)
            if missing
            else "unsupported: " + ", ".join(unsupported)
        )
        raise ValueError(f"Agent failed: {detail}.")
    for field in ("prompt", "description"):
        if not isinstance(arguments[field], str) or not arguments[field].strip():
            raise ValueError(f"Agent failed: '{field}' must be a non-empty string.")
    for field in ("model", "name"):
        value = arguments[field]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(
                f"Agent failed: '{field}' must be null or a non-empty string."
            )
    subagent_type = arguments["subagent_type"]
    if subagent_type is not None and subagent_type not in definition_types:
        raise ValueError(f"Agent failed: unknown subagent_type '{subagent_type}'.")
    for field in ("run_in_background", "isolation"):
        if not isinstance(arguments[field], bool):
            raise ValueError(f"Agent failed: '{field}' must be a boolean.")
    normalized = dict(arguments)
    if subagent_type is None:
        normalized["run_in_background"] = True
    for field in ("prompt", "description", "model", "name"):
        if normalized[field] is not None:
            normalized[field] = normalized[field].strip()
    name = normalized["name"]
    if name is not None and (
        len(name) > MAX_SUBAGENT_SLUG_LENGTH or SUBAGENT_SLUG_RE.fullmatch(name) is None
    ):
        raise ValueError(
            "Agent failed: 'name' must be an ASCII slug up to 48 characters with "
            "single '.', '_', or '-' separators between letters or digits."
        )
    return normalized


def validate_load_skill_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    unsupported = sorted(arguments.keys() - {"name", "task"})
    if unsupported:
        raise ValueError(
            "LoadSkill failed: unsupported parameter(s): " + ", ".join(unsupported)
        )
    name = arguments.get("name")
    task = arguments.get("task")
    if not isinstance(name, str) or not name:
        raise ValueError("LoadSkill failed: 'name' must be a non-empty string.")
    if not isinstance(task, str) or not task:
        raise ValueError("LoadSkill failed: 'task' must be a non-empty string.")
    return {"name": name, "task": task}
