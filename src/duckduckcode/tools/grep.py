from __future__ import annotations

import io
import os
from pathlib import Path
import re
from typing import Any

from .glob import _matches_glob
from .tool import Tool, ToolResult, create_tool

EXCLUDED_DIRECTORIES = {
    ".git",
    "node_modules",
    "vendor",
    ".idea",
    "__pycache__",
}
MAX_MATCHES = 100

GREP_PARAMS = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Regular expression to search for.",
        },
        "path": {
            "type": ["string", "null"],
            "description": "Absolute search root inside the working directory. Use null for the working directory; relative paths are accepted only for compatibility.",
        },
        "glob": {
            "type": ["string", "null"],
            "description": "Optional glob applied to relative file paths.",
        },
        "context": {
            "type": ["integer", "null"],
            "minimum": 0,
            "description": "Context lines before and after each match. Use null for 0.",
        },
    },
    "required": ["pattern", "path", "glob", "context"],
    "additionalProperties": False,
}


def create_grep_tool(
    working_directory: Path | None = None,
    allowed_directories: tuple[Path, ...] | None = None,
) -> Tool:
    base_directory = (working_directory or Path.cwd()).resolve()
    allowed = tuple(
        path.resolve() for path in (allowed_directories or (base_directory,))
    )
    return create_tool(
        "Grep",
        "Use Grep to search file contents with a regular expression when looking for definitions, references, or error text. Provide an absolute search root inside an allowed directory, or null for the working directory; relative roots are accepted only for compatibility. Use glob to filter relative file paths, context for nearby lines, then ReadFile before EditFile. Returns up to 100 matching lines.",
        GREP_PARAMS,
        lambda pattern, path, glob, context: _grep(
            base_directory, allowed, pattern, path, glob, context
        ),
        _validate_arguments,
        is_read_only=True,
        is_concurrency_safe=True,
        category="search",
    )


def _validate_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    unsupported = sorted(arguments.keys() - {"pattern", "path", "glob", "context"})
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            f"Grep failed: unsupported parameter(s): {names}. "
            "Remove them and use only pattern, path, glob, and context."
        )

    if "pattern" not in arguments:
        raise ValueError(
            "Grep failed: 'pattern' is required. Provide a regular expression."
        )
    pattern = arguments["pattern"]
    if not isinstance(pattern, str):
        raise ValueError(
            "Grep failed: 'pattern' must be a string. " "Provide a regular expression."
        )
    if not pattern:
        raise ValueError(
            "Grep failed: 'pattern' cannot be empty. " "Provide a regular expression."
        )
    try:
        compiled_pattern = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Grep failed: invalid regular expression: {exc}.") from exc

    path = arguments.get("path")
    if path is not None and not isinstance(path, str):
        raise ValueError(
            "Grep failed: 'path' must be a string or null. "
            "Use null for the working directory."
        )
    if isinstance(path, str) and not path.strip():
        raise ValueError(
            "Grep failed: 'path' cannot be empty. "
            "Use null for the working directory."
        )

    file_glob = arguments.get("glob")
    if file_glob is not None and not isinstance(file_glob, str):
        raise ValueError(
            "Grep failed: 'glob' must be a string or null. "
            "Use null to search every file."
        )
    if isinstance(file_glob, str) and not file_glob:
        raise ValueError(
            "Grep failed: 'glob' cannot be empty. " "Use null to search every file."
        )
    if isinstance(file_glob, str) and Path(file_glob).is_absolute():
        raise ValueError(
            "Grep failed: 'glob' must be relative because it is applied to paths "
            "inside the search root."
        )

    context = arguments.get("context")
    if context is None:
        context = 0
    if isinstance(context, bool) or not isinstance(context, int) or context < 0:
        raise ValueError(
            "Grep failed: 'context' must be a non-negative integer or null."
        )

    return {
        "pattern": compiled_pattern,
        "path": path,
        "glob": file_glob,
        "context": context,
    }


def _grep(
    working_directory: Path,
    allowed_directories: tuple[Path, ...],
    pattern: re.Pattern[str],
    path: str | None,
    file_glob: str | None,
    context: int,
) -> ToolResult:
    candidate = Path(path) if path is not None else working_directory
    unresolved = candidate if candidate.is_absolute() else working_directory / candidate
    root = unresolved.resolve()

    allowed_root = next(
        (allowed for allowed in allowed_directories if root.is_relative_to(allowed)),
        None,
    )
    if allowed_root is None:
        return ToolResult(
            f"Grep failed: search root '{root}' must be inside an allowed directory.",
            is_error=True,
        )
    relative_root = root.relative_to(allowed_root)
    if not root.exists():
        return ToolResult(
            f"Grep failed: search root '{unresolved}' does not exist. "
            "Check the path and try again.",
            is_error=True,
        )
    if not root.is_dir():
        return ToolResult(
            f"Grep failed: search root '{root}' is not a directory. "
            "Provide a directory path.",
            is_error=True,
        )
    if EXCLUDED_DIRECTORIES.intersection(relative_root.parts):
        return ToolResult("(no matches)")

    output: list[str] = []
    match_count = 0
    truncated = False
    glob_parts = Path(file_glob).parts if file_glob is not None else None

    for directory, directory_names, file_names in os.walk(root):
        directory_names[:] = sorted(
            name for name in directory_names if name not in EXCLUDED_DIRECTORIES
        )
        for filename in sorted(file_names):
            file_path = Path(directory) / filename
            relative_path = file_path.relative_to(root)
            if glob_parts is not None and not _matches_glob(
                relative_path.parts, glob_parts
            ):
                continue
            try:
                resolved_file = file_path.resolve()
                if not any(
                    resolved_file.is_relative_to(allowed)
                    for allowed in allowed_directories
                ):
                    continue
                with file_path.open("rb") as binary_stream:
                    if b"\x00" in binary_stream.read(512):
                        continue
                    binary_stream.seek(0)
                    with io.TextIOWrapper(
                        binary_stream, encoding="utf-8", errors="replace"
                    ) as text_stream:
                        lines = [line.rstrip("\r\n") for line in text_stream]
            except (OSError, ValueError):
                continue

            intervals: list[tuple[int, int]] = []
            matching_lines: set[int] = set()
            for line_index, line in enumerate(lines):
                if not pattern.search(line):
                    continue
                matching_lines.add(line_index)
                match_count += 1
                start = max(0, line_index - context)
                end = min(len(lines), line_index + context + 1)
                if intervals and start <= intervals[-1][1]:
                    intervals[-1] = (
                        intervals[-1][0],
                        max(intervals[-1][1], end),
                    )
                else:
                    intervals.append((start, end))
                if match_count == MAX_MATCHES:
                    truncated = True
                    break

            try:
                display_path = file_path.relative_to(working_directory).as_posix()
            except ValueError:
                display_path = file_path.as_posix()
            for start, end in intervals:
                output.extend(
                    f"{display_path}{':' if line_number in matching_lines else '-'}"
                    f"{line_number + 1}"
                    f"{':' if line_number in matching_lines else '-'}"
                    f"{lines[line_number]}"
                    for line_number in range(start, end)
                )
            if truncated:
                break
        if truncated:
            break

    if truncated:
        output.append(f"[Results truncated after {MAX_MATCHES} matches.]")
    return ToolResult("\n".join(output) if output else "(no matches)")
