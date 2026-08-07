from __future__ import annotations

import json
from typing import Any

from .tool import BuiltinTool, ToolResult, create_tool

_OPERATIONS = (
    "definition",
    "references",
    "hover",
    "document_symbols",
    "workspace_symbols",
)
_FIELDS = {
    "operation",
    "server",
    "path",
    "line",
    "column",
    "query",
    "include_declaration",
}


def create_lsp_tool(manager: Any) -> BuiltinTool:
    nullable_string = {"type": ["string", "null"]}
    nullable_integer = {"type": ["integer", "null"], "minimum": 1}
    params = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": list(_OPERATIONS)},
            "server": {"type": "string", "enum": list(manager.server_names)},
            "path": nullable_string,
            "line": nullable_integer,
            "column": nullable_integer,
            "query": nullable_string,
            "include_declaration": {"type": ["boolean", "null"]},
        },
        "required": sorted(_FIELDS),
        "additionalProperties": False,
    }
    return create_tool(
        "LSP",
        "Navigate code with a configured language server. Coordinates are 1-based "
        "Unicode code points; returned LSP positions remain 0-based. Servers:\n- "
        + "\n- ".join(manager.server_descriptions),
        params,
        lambda **arguments: ToolResult(
            json.dumps(manager.execute(**arguments), ensure_ascii=False)
        ),
        lambda arguments: _validate(arguments, manager.server_names),
        is_read_only=True,
        is_concurrency_safe=True,
        category="search",
    )


def _validate(arguments: dict[str, Any], servers: tuple[str, ...]) -> dict[str, Any]:
    if set(arguments) != _FIELDS:
        raise ValueError("LSP requires exactly: " + ", ".join(sorted(_FIELDS)))
    operation = arguments["operation"]
    server = arguments["server"]
    if operation not in _OPERATIONS:
        raise ValueError("LSP operation is invalid")
    if server not in servers:
        raise ValueError("LSP server must be a configured server")
    path = arguments["path"]
    line = arguments["line"]
    column = arguments["column"]
    query = arguments["query"]
    include = arguments["include_declaration"]
    if operation in {"definition", "references", "hover"}:
        if not isinstance(path, str) or not path:
            raise ValueError(f"LSP {operation} requires path")
        if isinstance(line, bool) or not isinstance(line, int) or line < 1:
            raise ValueError(f"LSP {operation} requires a positive line")
        if isinstance(column, bool) or not isinstance(column, int) or column < 1:
            raise ValueError(f"LSP {operation} requires a positive column")
        if query is not None or (operation != "references" and include is not None):
            raise ValueError(f"LSP {operation} received irrelevant arguments")
    elif operation == "document_symbols":
        if not isinstance(path, str) or not path:
            raise ValueError("LSP document_symbols requires path")
        if any(value is not None for value in (line, column, query, include)):
            raise ValueError("LSP document_symbols only requires path")
    else:
        if not isinstance(query, str) or not query:
            raise ValueError("LSP workspace_symbols requires a non-empty query")
        if any(value is not None for value in (path, line, column, include)):
            raise ValueError("LSP workspace_symbols only requires query")
    if include is not None and not isinstance(include, bool):
        raise ValueError("LSP include_declaration must be boolean or null")
    return dict(arguments)
