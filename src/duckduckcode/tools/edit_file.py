from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from .tool import Tool, ToolResult, create_tool

EDIT_FILE_PARAMS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Absolute file path. Relative paths are accepted only for compatibility.",
        },
        "old_string": {
            "type": "string",
            "description": "Exact text to replace. It must occur exactly once in the file.",
        },
        "new_string": {
            "type": "string",
            "description": "Replacement text.",
        },
    },
    "required": ["path", "old_string", "new_string"],
    "additionalProperties": False,
}


def create_edit_file_tool(working_directory: Path | None = None) -> Tool:
    base_directory = (working_directory or Path.cwd()).resolve()
    return create_tool(
        "EditFile",
        "Use EditFile for targeted edits to an existing UTF-8 text file after using ReadFile first. Pass an absolute file path; relative paths are accepted only for compatibility. Provide old_string with enough exact context to match once, then new_string; the tool returns numbered context near the change.",
        EDIT_FILE_PARAMS,
        lambda path, old_string, new_string: _edit_file(
            base_directory, path, old_string, new_string
        ),
        _validate_arguments,
        category="file",
    )


def _validate_arguments(arguments: dict[str, Any]) -> dict[str, str]:
    unsupported = sorted(arguments.keys() - {"path", "old_string", "new_string"})
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            f"EditFile failed: unsupported parameter(s): {names}. "
            "Remove them and use only path, old_string, and new_string."
        )

    if "path" not in arguments:
        raise ValueError(
            "EditFile failed: 'path' is required. Provide a filesystem path."
        )
    path = arguments["path"]
    if not isinstance(path, str):
        raise ValueError(
            "EditFile failed: 'path' must be a string. Provide a filesystem path."
        )
    if not path.strip():
        raise ValueError(
            "EditFile failed: 'path' cannot be empty. Provide a filesystem path."
        )

    if "old_string" not in arguments:
        raise ValueError(
            "EditFile failed: 'old_string' is required. "
            "Provide the exact text to replace."
        )
    old_string = arguments["old_string"]
    if not isinstance(old_string, str):
        raise ValueError(
            "EditFile failed: 'old_string' must be a string. "
            "Provide exact UTF-8 text."
        )
    if not old_string:
        raise ValueError(
            "EditFile failed: 'old_string' cannot be empty. "
            "Provide exact text that occurs once."
        )

    if "new_string" not in arguments:
        raise ValueError(
            "EditFile failed: 'new_string' is required. " "Provide replacement text."
        )
    new_string = arguments["new_string"]
    if not isinstance(new_string, str):
        raise ValueError(
            "EditFile failed: 'new_string' must be a string. "
            "Provide replacement UTF-8 text."
        )
    return {"path": path, "old_string": old_string, "new_string": new_string}


def _edit_file(
    working_directory: Path, path: str, old_string: str, new_string: str
) -> ToolResult:
    candidate = Path(path)
    unresolved = candidate if candidate.is_absolute() else working_directory / candidate
    temporary_path: str | None = None

    try:
        resolved = unresolved.parent.resolve() / unresolved.name
        if resolved.is_symlink():
            return ToolResult(
                f"EditFile failed: '{resolved}' is a symbolic link. "
                "Provide its explicit target path.",
                is_error=True,
            )

        descriptor = os.open(
            resolved,
            os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(descriptor, "rb") as stream:
            file_stat = os.fstat(stream.fileno())
            if stat.S_ISDIR(file_stat.st_mode):
                return ToolResult(
                    f"EditFile failed: '{resolved}' is a directory. "
                    "Provide a file path instead.",
                    is_error=True,
                )
            if not stat.S_ISREG(file_stat.st_mode):
                return ToolResult(
                    f"EditFile failed: '{resolved}' is not a regular file. "
                    "Choose a regular UTF-8 text file.",
                    is_error=True,
                )
            mode = stat.S_IMODE(file_stat.st_mode)
            content = stream.read().decode("utf-8")

        matches = content.count(old_string)
        if matches == 0:
            return ToolResult(
                f"EditFile failed: 'old_string' was not found in '{resolved}'. "
                "Read the file again and provide exact text.",
                is_error=True,
            )
        if matches > 1:
            return ToolResult(
                f"EditFile failed: 'old_string' appears {matches} times in "
                f"'{resolved}'. Provide more context so it matches exactly once.",
                is_error=True,
            )

        match_index = content.index(old_string)
        edited = content.replace(old_string, new_string, 1)
        payload = edited.encode("utf-8")
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(temporary_path, resolved)
        temporary_path = None
    except FileNotFoundError:
        return ToolResult(
            f"EditFile failed: '{unresolved}' does not exist. "
            "Check the path and try again.",
            is_error=True,
        )
    except IsADirectoryError:
        return ToolResult(
            f"EditFile failed: '{unresolved}' is a directory. "
            "Provide a file path instead.",
            is_error=True,
        )
    except PermissionError:
        return ToolResult(
            f"EditFile failed: Permission denied for '{unresolved}'. "
            "Check file and directory permissions.",
            is_error=True,
        )
    except UnicodeDecodeError:
        return ToolResult(
            f"EditFile failed: '{unresolved}' is not valid UTF-8 text. "
            "Convert it to UTF-8 before editing.",
            is_error=True,
        )
    except UnicodeEncodeError:
        return ToolResult(
            "EditFile failed: 'new_string' cannot be encoded as valid UTF-8 text. "
            "Remove invalid Unicode surrogate characters and try again.",
            is_error=True,
        )
    except OSError as exc:
        return ToolResult(
            f"EditFile failed: could not edit '{unresolved}': {exc}. "
            "Check the path, available space, and permissions, then try again.",
            is_error=True,
        )
    finally:
        if temporary_path is not None:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass

    lines = edited.splitlines()
    if not lines:
        context = "(empty file)"
    else:
        changed_start = content.count("\n", 0, match_index) + 1
        changed_end = changed_start + new_string.count("\n")
        first = max(1, changed_start - 3)
        last = min(len(lines), changed_end + 3)
        context = "\n".join(
            f"{line_number}: {lines[line_number - 1]}"
            for line_number in range(first, last + 1)
        )
    return ToolResult(f"Successfully edited '{resolved}'.\n{context}")
