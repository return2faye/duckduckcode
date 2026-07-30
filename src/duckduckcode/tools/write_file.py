from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from .tool import Tool, ToolResult, create_tool

WRITE_FILE_PARAMS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Absolute file path. Relative paths are accepted only for compatibility.",
        },
        "content": {
            "type": "string",
            "description": "Complete UTF-8 file content to write.",
        },
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}


def create_write_file_tool(working_directory: Path | None = None) -> Tool:
    base_directory = (working_directory or Path.cwd()).resolve()
    return create_tool(
        "WriteFile",
        "Use WriteFile only to create a new UTF-8 text file or replace a whole file when that is intended. Pass an absolute file path inside the working directory or private temporary directory; relative paths are accepted only for compatibility. Prefer EditFile for targeted changes to existing files, and ReadFile first when replacing existing content.",
        WRITE_FILE_PARAMS,
        lambda path, content: _write_file(base_directory, path, content),
        _validate_arguments,
        category="file",
    )


def _validate_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    unsupported = sorted(arguments.keys() - {"path", "content"})
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            f"WriteFile failed: unsupported parameter(s): {names}. "
            "Remove them and use only path and content."
        )

    if "path" not in arguments:
        raise ValueError(
            "WriteFile failed: 'path' is required. "
            "Provide an absolute path or a path relative to the working directory."
        )
    path = arguments["path"]
    if not isinstance(path, str):
        raise ValueError(
            "WriteFile failed: 'path' must be a string. Provide a filesystem path."
        )
    if not path.strip():
        raise ValueError(
            "WriteFile failed: 'path' cannot be empty. Provide a filesystem path."
        )

    if "content" not in arguments:
        raise ValueError(
            "WriteFile failed: 'content' is required. "
            "Provide the complete file content, not a patch or appended fragment."
        )
    content = arguments["content"]
    if not isinstance(content, str):
        raise ValueError(
            "WriteFile failed: 'content' must be a string. "
            "Provide the complete file as UTF-8 text."
        )
    return {"path": path, "content": content}


def _write_file(working_directory: Path, path: str, content: str) -> ToolResult:
    candidate = Path(path)
    unresolved = candidate if candidate.is_absolute() else working_directory / candidate
    temporary_path: str | None = None

    try:
        parent = unresolved.parent.resolve()
        resolved = parent / unresolved.name
        missing_parents: list[Path] = []
        current = parent
        while not current.exists():
            missing_parents.append(current)
            current = current.parent

        for directory in reversed(missing_parents):
            try:
                directory.mkdir(mode=0o755)
            except FileExistsError:
                if not directory.is_dir():
                    raise NotADirectoryError(
                        f"Parent path '{directory}' is not a directory"
                    )
            else:
                directory.chmod(0o755)

        if resolved.is_symlink():
            return ToolResult(
                f"WriteFile failed: '{resolved}' is a symbolic link. "
                "Provide its explicit target path or use Bash if replacing the "
                "link itself is intended.",
                is_error=True,
            )
        if resolved.exists():
            mode = resolved.stat().st_mode
            if stat.S_ISDIR(mode):
                return ToolResult(
                    f"WriteFile failed: '{resolved}' is a directory. "
                    "Provide a file path instead.",
                    is_error=True,
                )
            if not stat.S_ISREG(mode):
                return ToolResult(
                    f"WriteFile failed: '{resolved}' is not a regular file. "
                    "Choose a regular file path or use Bash for special files.",
                    is_error=True,
                )

        payload = content.encode("utf-8")
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{resolved.name}.", suffix=".tmp", dir=parent
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.replace(temporary_path, resolved)
        temporary_path = None
    except UnicodeEncodeError:
        return ToolResult(
            "WriteFile failed: 'content' cannot be encoded as valid UTF-8 text. "
            "Remove invalid Unicode surrogate characters and try again.",
            is_error=True,
        )
    except PermissionError:
        return ToolResult(
            f"WriteFile failed: Permission denied for '{unresolved}'. "
            "Check the parent directory and file permissions, then try again.",
            is_error=True,
        )
    except IsADirectoryError:
        return ToolResult(
            f"WriteFile failed: '{unresolved}' is a directory. "
            "Provide a file path instead.",
            is_error=True,
        )
    except OSError as exc:
        return ToolResult(
            f"WriteFile failed: could not write '{unresolved}': {exc}. "
            "Try again after checking the path, available space, and permissions.",
            is_error=True,
        )
    finally:
        if temporary_path is not None:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass

    return ToolResult(f"Successfully wrote {len(payload)} bytes to '{resolved}'.")
