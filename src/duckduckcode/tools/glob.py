from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

from .tool import Tool, ToolResult, create_tool

EXCLUDED_DIRECTORIES = {
    ".git",
    "node_modules",
    "vendor",
    ".idea",
    "__pycache__",
}
MAX_RESULTS = 200

GLOB_PARAMS = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Glob pattern such as **/*.py.",
        },
        "path": {
            "type": ["string", "null"],
            "description": "Absolute search root. Use null for the working directory; relative paths are accepted only for compatibility.",
        },
    },
    "required": ["pattern", "path"],
    "additionalProperties": False,
}


def create_glob_tool(working_directory: Path | None = None) -> Tool:
    base_directory = (working_directory or Path.cwd()).resolve()
    return create_tool(
        "Glob",
        "Find files by glob pattern, including recursive ** patterns. Returns up to 200 newest files first.",
        GLOB_PARAMS,
        lambda pattern, path: _glob(base_directory, pattern, path),
        _validate_arguments,
        is_read_only=True,
        is_concurrency_safe=True,
        category="search",
    )


def _validate_arguments(arguments: dict[str, Any]) -> dict[str, str | None]:
    unsupported = sorted(arguments.keys() - {"pattern", "path"})
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            f"Glob failed: unsupported parameter(s): {names}. "
            "Remove them and use only pattern and path."
        )

    if "pattern" not in arguments:
        raise ValueError(
            "Glob failed: 'pattern' is required. Provide a glob such as **/*.py."
        )
    pattern = arguments["pattern"]
    if not isinstance(pattern, str):
        raise ValueError(
            "Glob failed: 'pattern' must be a string. "
            "Provide a glob such as **/*.py."
        )
    if not pattern:
        raise ValueError(
            "Glob failed: 'pattern' cannot be empty. " "Provide a glob such as **/*.py."
        )
    if Path(pattern).is_absolute():
        raise ValueError(
            "Glob failed: 'pattern' must be relative. "
            "Use 'path' to choose the search root."
        )

    path = arguments.get("path")
    if path is not None and not isinstance(path, str):
        raise ValueError(
            "Glob failed: 'path' must be a string or null. "
            "Use null for the working directory."
        )
    if isinstance(path, str) and not path.strip():
        raise ValueError(
            "Glob failed: 'path' cannot be empty. "
            "Use null for the working directory."
        )
    return {"pattern": pattern, "path": path}


def _glob(working_directory: Path, pattern: str, path: str | None) -> ToolResult:
    candidate = Path(path) if path is not None else working_directory
    unresolved = candidate if candidate.is_absolute() else working_directory / candidate

    try:
        root = unresolved.resolve()
        if not root.exists():
            return ToolResult(
                f"Glob failed: search root '{unresolved}' does not exist. "
                "Check the path and try again.",
                is_error=True,
            )
        if not root.is_dir():
            return ToolResult(
                f"Glob failed: search root '{root}' is not a directory. "
                "Provide a directory path.",
                is_error=True,
            )

        matches: dict[Path, int] = {}
        for matched in glob.iglob(
            pattern,
            root_dir=root,
            recursive=True,
            include_hidden=True,
        ):
            relative = Path(matched)
            if EXCLUDED_DIRECTORIES.intersection(relative.parts):
                continue
            result = (root / relative).resolve()
            if result.is_file():
                matches[result] = result.stat().st_mtime_ns
    except PermissionError:
        return ToolResult(
            f"Glob failed: Permission denied while searching '{unresolved}'. "
            "Check directory permissions.",
            is_error=True,
        )
    except OSError as exc:
        return ToolResult(
            f"Glob failed: could not search '{unresolved}': {exc}. "
            "Check the path and try again.",
            is_error=True,
        )

    paths = []
    for result, _mtime in sorted(
        matches.items(), key=lambda item: item[1], reverse=True
    )[:MAX_RESULTS]:
        try:
            display = result.relative_to(working_directory)
        except ValueError:
            display = result
        paths.append(display.as_posix())
    return ToolResult("\n".join(paths) if paths else "(no matches)")
