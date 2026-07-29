from __future__ import annotations

import io
import os
from pathlib import Path
import stat
from typing import Any

from ..tool import Tool, ToolResult, create_tool

MAX_READ_LINES = 2000
# ponytail: fixed context guards; make configurable if model contexts diverge.
MAX_LINE_CHARS = 100_000
MAX_OUTPUT_CHARS = 200_000

READ_FILE_PARAMS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Absolute path or path relative to the working directory.",
        },
        "offset": {
            "type": ["integer", "null"],
            "minimum": 1,
            "description": "1-based first line. Use null for 1.",
        },
        "limit": {
            "type": ["integer", "null"],
            "minimum": 1,
            "maximum": MAX_READ_LINES,
            "description": "Number of lines to read. Use null for 2000.",
        },
    },
    "required": ["path", "offset", "limit"],
    "additionalProperties": False,
}


def create_read_file_tool(working_directory: Path | None = None) -> Tool:
    base_directory = (working_directory or Path.cwd()).resolve()
    return create_tool(
        "ReadFile",
        "Read a UTF-8 text file with line numbers. Use offset and limit to read large files in chunks.",
        READ_FILE_PARAMS,
        lambda path, offset, limit: _read_file(base_directory, path, offset, limit),
        _validate_arguments,
        is_read_only=True,
        is_concurrency_safe=True,
        category="file",
    )


def _validate_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    unsupported = sorted(arguments.keys() - {"path", "offset", "limit"})
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            f"ReadFile failed: unsupported parameter(s): {names}. "
            "Remove them and use only path, offset, and limit."
        )

    if "path" not in arguments:
        raise ValueError(
            "ReadFile failed: 'path' is required. "
            "Provide an absolute path or a path relative to the working directory."
        )
    path = arguments["path"]
    if not isinstance(path, str):
        raise ValueError(
            "ReadFile failed: 'path' must be a string. Provide a filesystem path."
        )
    if not path.strip():
        raise ValueError(
            "ReadFile failed: 'path' cannot be empty. Provide a filesystem path."
        )

    offset = _positive_integer(arguments.get("offset"), "offset", 1)
    limit = _positive_integer(arguments.get("limit"), "limit", MAX_READ_LINES)
    if limit > MAX_READ_LINES:
        raise ValueError(
            f"ReadFile failed: 'limit' must be between 1 and {MAX_READ_LINES}. "
            "Use multiple calls to read more lines."
        )
    return {"path": path, "offset": offset, "limit": limit}


def _positive_integer(value: Any, name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"ReadFile failed: '{name}' must be a positive integer. "
            f"Use {name}=1 or a larger whole number."
        )
    return value


def _read_file(
    working_directory: Path, path: str, offset: int, limit: int
) -> ToolResult:
    candidate = Path(path)
    unresolved = candidate if candidate.is_absolute() else working_directory / candidate

    try:
        resolved = unresolved.resolve()
        descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as binary_stream:
            mode = os.fstat(binary_stream.fileno()).st_mode
            if stat.S_ISDIR(mode):
                return ToolResult(
                    f"ReadFile failed: '{unresolved}' is a directory. "
                    "Provide a file path inside that directory.",
                    is_error=True,
                )
            if not stat.S_ISREG(mode):
                return ToolResult(
                    f"ReadFile failed: '{resolved}' is not a regular file. "
                    "Use Bash with an appropriate command-line tool to inspect it.",
                    is_error=True,
                )
            if b"\x00" in binary_stream.read(512):
                return ToolResult(
                    f"ReadFile failed: '{resolved}' appears to be binary because "
                    "the first 512 bytes contain a NUL byte. Use Bash with `file` "
                    "or `xxd` to inspect it.",
                    is_error=True,
                )
            binary_stream.seek(0)

            lines: list[tuple[int, str]] = []
            last_line = 0
            output_characters = 0
            has_more = False
            with io.TextIOWrapper(binary_stream, encoding="utf-8") as stream:
                line_number = 0
                while True:
                    line = stream.readline(MAX_LINE_CHARS + 1)
                    if not line:
                        break
                    line_number += 1
                    last_line = line_number
                    if line_number >= offset and len(lines) == limit:
                        has_more = True
                        break
                    if len(line) > MAX_LINE_CHARS:
                        return ToolResult(
                            f"ReadFile failed: line {line_number} exceeds "
                            f"{MAX_LINE_CHARS} characters. Use Bash with `cut`, "
                            "`head`, or another command that can bound bytes.",
                            is_error=True,
                        )
                    if line_number < offset:
                        continue
                    line = line.rstrip("\r\n")
                    rendered_size = len(f"{line_number}: {line}") + (1 if lines else 0)
                    if output_characters + rendered_size > MAX_OUTPUT_CHARS:
                        return ToolResult(
                            f"ReadFile failed: requested output exceeds "
                            f"{MAX_OUTPUT_CHARS} characters. Retry with a smaller "
                            "limit or use Bash to select narrower content.",
                            is_error=True,
                        )
                    output_characters += rendered_size
                    lines.append((line_number, line))
    except FileNotFoundError:
        return ToolResult(
            f"ReadFile failed: '{unresolved}' does not exist. "
            "Check the path and try again.",
            is_error=True,
        )
    except IsADirectoryError:
        return ToolResult(
            f"ReadFile failed: '{unresolved}' is a directory. "
            "Provide a file path inside that directory.",
            is_error=True,
        )
    except PermissionError:
        return ToolResult(
            f"ReadFile failed: Permission denied for '{unresolved}'. "
            "Check file permissions or choose a readable file.",
            is_error=True,
        )
    except UnicodeDecodeError:
        return ToolResult(
            f"ReadFile failed: '{unresolved}' is not valid UTF-8 text. "
            "Use Bash with `file`, `iconv`, or `xxd` to inspect or convert it.",
            is_error=True,
        )
    except OSError as exc:
        return ToolResult(
            f"ReadFile failed: could not read '{unresolved}': {exc}. "
            "Try again or inspect the path with a command-line tool.",
            is_error=True,
        )

    if not lines:
        if last_line == 0 and offset == 1:
            return ToolResult("(empty file)")
        if last_line == 0:
            return ToolResult(
                f"ReadFile failed: offset {offset} is beyond the end of "
                f"'{resolved}' because the file is empty. Retry with offset=1.",
                is_error=True,
            )
        return ToolResult(
            f"ReadFile failed: offset {offset} is beyond the end of '{resolved}'; "
            f"the last line is {last_line}. Retry with offset between 1 and "
            f"{last_line}.",
            is_error=True,
        )

    content = "\n".join(f"{line_number}: {line}" for line_number, line in lines[:limit])
    if has_more:
        content += (
            f"\n[More lines available. Continue with offset={offset + limit}, "
            f"limit={limit}.]"
        )
    return ToolResult(content)
