from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from ..tools.tool import ToolCall

_PATH_TOOLS = frozenset({"ReadFile", "WriteFile", "EditFile", "Glob", "Grep"})


class PathSandbox:
    def __init__(self, workspace: Path, temporary_parent: Path | None = None) -> None:
        self.workspace = workspace.resolve()
        parent = (temporary_parent or Path(tempfile.gettempdir())).resolve()
        self._temporary = tempfile.TemporaryDirectory(
            prefix="duckduckcode-", dir=parent
        )
        self._tool_results = tempfile.TemporaryDirectory(
            prefix="duckduckcode-results-", dir=parent
        )
        self.temporary_directory = Path(self._temporary.name).resolve()
        self.tool_result_directory = Path(self._tool_results.name).resolve()
        self.temporary_directory.chmod(0o700)
        self.tool_result_directory.chmod(0o700)
        self.allowed_directories = (
            self.workspace,
            self.temporary_directory,
            self.tool_result_directory,
        )
        self._active = True

    def __call__(self, tool_call: ToolCall) -> str | None:
        if tool_call.name not in _PATH_TOOLS:
            return None

        path = tool_call.arguments.get("path")
        if path is None:
            candidate = self.workspace
        elif isinstance(path, str):
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = self.workspace / candidate
        else:
            return None

        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            return (
                f"Permission denied: {tool_call.name} path '{candidate}' "
                "could not be resolved safely."
            )

        if any(resolved.is_relative_to(root) for root in self.allowed_directories):
            return None
        return (
            f"Permission denied: {tool_call.name} path '{candidate}' resolves "
            f"outside the allowed directories."
        )

    def close(self) -> None:
        self._temporary.cleanup()
        self._tool_results.cleanup()
        self._active = False

    def start_task(self) -> None:
        if self._active:
            return
        self.temporary_directory.mkdir(mode=0o700)
        self.temporary_directory.chmod(0o700)
        self._active = True

    def finish_task(self) -> None:
        if not self._active:
            return
        if self.temporary_directory.is_symlink():
            self.temporary_directory.unlink()
        elif self.temporary_directory.exists():
            shutil.rmtree(self.temporary_directory)
        self._active = False
