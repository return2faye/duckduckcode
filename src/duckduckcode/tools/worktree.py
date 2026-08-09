from __future__ import annotations

from typing import Any, Callable

from .tool import Tool, create_tool


def create_list_worktrees_tool(
    handler: Callable[[], object],
) -> Tool:
    return create_tool(
        "ListWorktrees",
        "List persistent isolated-fork worktrees, including their exact IDs, "
        "branches, base commits, activity, dirty files, and parent-change state.",
        {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler,
        _empty_arguments,
        is_read_only=True,
        is_concurrency_safe=True,
        category="worktree",
    )


def create_remove_worktree_tool(
    handler: Callable[[str], object],
) -> Tool:
    return create_tool(
        "RemoveWorktree",
        "Remove one inactive persistent worktree by the exact ID returned by "
        "ListWorktrees or an isolated Agent result. A final binary patch is "
        "returned before the worktree and temporary branch are deleted.",
        {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Exact managed worktree ID.",
                }
            },
            "required": ["id"],
            "additionalProperties": False,
        },
        lambda id: handler(id),
        _remove_arguments,
        category="worktree",
    )


def _empty_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if arguments:
        raise ValueError("ListWorktrees does not accept arguments.")
    return {}


def _remove_arguments(arguments: dict[str, Any]) -> dict[str, str]:
    if set(arguments) != {"id"}:
        raise ValueError("RemoveWorktree requires exactly 'id'.")
    worktree_id = arguments["id"]
    if not isinstance(worktree_id, str) or not worktree_id.strip():
        raise ValueError("RemoveWorktree 'id' must be a non-empty string.")
    return {"id": worktree_id.strip()}
