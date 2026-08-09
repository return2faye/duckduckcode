from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from ..tools.tool import ToolCall

_PATH_TOOLS = frozenset({"ReadFile", "WriteFile", "EditFile", "Glob", "Grep"})
_SKILL_READ_TOOLS = frozenset({"ReadFile", "Glob", "Grep"})


class PathSandbox:
    def __init__(
        self,
        workspace: Path,
        temporary_parent: Path | None = None,
        *,
        protect_git_metadata: bool = False,
        read_only_paths: tuple[Path, ...] = (),
    ) -> None:
        self.workspace = workspace.resolve()
        self.protect_git_metadata = protect_git_metadata
        self._read_only_paths = tuple(path.resolve() for path in read_only_paths)
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
        self._skill_directories: tuple[Path, ...] = ()
        self._active = True

    def set_skill_directories(self, directories: tuple[Path, ...]) -> None:
        self._skill_directories = tuple(path.resolve() for path in directories)

    def current_allowed_directories(self) -> tuple[Path, ...]:
        return self.allowed_directories + self._skill_directories

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

        git_metadata = self.workspace / ".git"
        read_only = any(
            resolved == root or resolved.is_relative_to(root)
            for root in self._read_only_paths
        )
        if tool_call.name in {"WriteFile", "EditFile"} and read_only:
            return "Permission denied: injected worktree dependencies are read-only."
        if (
            self.protect_git_metadata
            and tool_call.name in {"WriteFile", "EditFile"}
            and (resolved == git_metadata or resolved.is_relative_to(git_metadata))
        ):
            return "Permission denied: isolated subagents cannot modify Git metadata."

        roots = (
            self.current_allowed_directories()
            if tool_call.name in _SKILL_READ_TOOLS
            else self.allowed_directories
        )
        if any(resolved.is_relative_to(root) for root in roots) or (
            tool_call.name in _SKILL_READ_TOOLS and read_only
        ):
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
