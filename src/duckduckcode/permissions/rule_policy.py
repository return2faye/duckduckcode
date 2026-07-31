from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import glob
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Literal

import yaml

from ..tools.glob import _matches_glob
from ..tools.tool import ToolCall

PermissionAction = Literal["deny", "ask", "allow", "unspecified"]
RuleAction = Literal["deny", "ask", "allow"]
PermissionMode = Literal["full_access", "ask_for_approval", "accept_edits"]
DEFAULT_PERMISSION_MODE: PermissionMode = "ask_for_approval"
PERMISSION_MODES = frozenset({"full_access", "ask_for_approval", "accept_edits"})
_ACTION_BY_NAME: dict[str, RuleAction] = {
    "deny": "deny",
    "ask": "ask",
    "allow": "allow",
}
_PATH_TOOLS = frozenset({"ReadFile", "WriteFile", "EditFile", "Glob", "Grep"})


@dataclass(frozen=True)
class PermissionDecision:
    action: PermissionAction
    message: str = ""
    content: str = ""


@dataclass(frozen=True)
class _Rule:
    action: RuleAction
    tool_name: str
    pattern: str
    text: str
    local: bool


class RulePolicy:
    def __init__(
        self,
        workspace: Path,
        temporary_directory: Path,
        rules: list[_Rule],
        local_file: Path,
        local_data: dict[str, list[dict[str, str]]],
        permission_mode: PermissionMode,
    ) -> None:
        self.workspace = workspace.resolve()
        self.temporary_directory = temporary_directory.resolve()
        self.rules = rules
        self.local_file = local_file
        self._local_data = local_data
        self.permission_mode = permission_mode

    @classmethod
    def load(
        cls,
        workspace: Path,
        temporary_directory: Path,
        known_tools: set[str],
        *,
        home: Path | None = None,
    ) -> RulePolicy:
        workspace = workspace.resolve()
        temporary_directory = temporary_directory.resolve()
        home = (home or Path.home()).resolve()
        global_directory = _permission_directory(
            home / ".duckduckcode", home, "home directory"
        )
        project_directory = _permission_directory(
            workspace / ".duckduckcode", workspace, "workspace"
        )
        local_file = project_directory / "permissions.local.yaml"
        sources = (
            (global_directory / "permissions.yaml", False),
            (project_directory / "permissions.yaml", False),
            (local_file, True),
        )
        for path, _local in sources:
            _initialize_file(path, known_tools)
        rules: list[_Rule] = []
        local_data: dict[str, list[dict[str, str]]] = {}
        permission_mode = DEFAULT_PERMISSION_MODE
        for path, local in sources:
            source_rules, data, source_mode = _load_file(
                path,
                local,
                workspace,
                temporary_directory,
                known_tools,
            )
            rules.extend(source_rules)
            if local:
                local_data = data
                permission_mode = source_mode
        return cls(
            workspace,
            temporary_directory,
            rules,
            local_file,
            local_data,
            permission_mode,
        )

    @staticmethod
    def read_permission_mode(workspace: Path) -> PermissionMode:
        path = workspace.resolve() / ".duckduckcode" / "permissions.local.yaml"
        if not path.exists():
            return DEFAULT_PERMISSION_MODE
        if not path.parent.resolve().is_relative_to(workspace.resolve()):
            raise RuntimeError(
                f"Permission file '{path}' resolves outside the workspace."
            )
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise RuntimeError(
                f"Permission file '{path}' has invalid YAML: {exc}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not read permission file '{path}': {exc}"
            ) from exc
        if loaded is None:
            return DEFAULT_PERMISSION_MODE
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Permission file '{path}' must contain a YAML mapping.")
        mode = loaded.get("permission_mode", DEFAULT_PERMISSION_MODE)
        if not isinstance(mode, str) or mode not in PERMISSION_MODES:
            raise RuntimeError(
                f"Permission file '{path}' has invalid permission_mode '{mode}'."
            )
        return mode

    def check(self, tool_call: ToolCall) -> PermissionDecision:
        content = self._content(tool_call)
        if content is None:
            return PermissionDecision("unspecified")
        matches = [
            rule
            for rule in self.rules
            if rule.tool_name == tool_call.name and _matches(rule, content)
        ]
        denied = next((rule for rule in matches if rule.action == "deny"), None)
        if denied is not None:
            return PermissionDecision(
                "deny",
                f"Permission denied by rule '{denied.text}'.",
                content,
            )
        exact_local_allow = next(
            (
                rule
                for rule in matches
                if rule.action == "allow"
                and rule.local
                and rule.pattern == glob.escape(content)
            ),
            None,
        )
        if exact_local_allow is not None:
            return PermissionDecision("allow", content=content)
        asked = next((rule for rule in matches if rule.action == "ask"), None)
        if asked is not None:
            return PermissionDecision(
                "ask",
                f"Permission approval required by rule '{asked.text}'.",
                content,
            )
        if any(rule.action == "allow" for rule in matches):
            return PermissionDecision("allow", content=content)
        return PermissionDecision(
            "ask",
            "Permission approval required: no permission rule matched.",
            content,
        )

    def remember_allow(self, tool_call: ToolCall) -> None:
        content = self._content(tool_call)
        if content is None:
            raise ValueError(
                f"Cannot remember permission for malformed {tool_call.name} input."
            )
        pattern = self._stored_pattern(tool_call.name, content)
        local_data = {
            name: [dict(entry) for entry in entries]
            for name, entries in self._local_data.items()
        }
        entry = {"content": pattern, "action": "allow"}
        tool_rules = local_data.setdefault(tool_call.name, [])
        if entry in tool_rules:
            return
        tool_rules.append(entry)
        self._write_local(local_data, self.permission_mode)
        self._local_data = local_data
        self.rules.append(
            _build_rule(
                "allow",
                tool_call.name,
                pattern,
                self.local_file,
                True,
                self.workspace,
                self.temporary_directory,
            )
        )

    def set_permission_mode(self, mode: PermissionMode) -> None:
        if mode not in PERMISSION_MODES:
            raise ValueError(f"Unsupported permission mode '{mode}'.")
        self._write_local(self._local_data, mode)
        self.permission_mode = mode

    def _write_local(
        self,
        local_data: dict[str, list[dict[str, str]]],
        permission_mode: PermissionMode,
    ) -> None:
        self.local_file.parent.mkdir(parents=True, exist_ok=True)
        permission_directory = self.local_file.parent.resolve()
        if not permission_directory.is_relative_to(self.workspace):
            raise RuntimeError(
                f"Permission file '{self.local_file}' resolves outside the workspace."
            )
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".permissions.", suffix=".yaml", dir=permission_directory
        )
        target = permission_directory / self.local_file.name
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                yaml.safe_dump(
                    {"permission_mode": permission_mode, **local_data},
                    stream,
                    allow_unicode=True,
                    sort_keys=False,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, target)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _content(self, tool_call: ToolCall) -> str | None:
        if tool_call.name == "Bash":
            command = tool_call.arguments.get("command")
            return command if isinstance(command, str) else None
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
            return candidate.resolve().as_posix()
        except (OSError, RuntimeError):
            return None

    def _stored_pattern(self, tool_name: str, content: str) -> str:
        if tool_name == "Bash":
            return glob.escape(content)
        path = Path(content)
        if path.is_relative_to(self.workspace):
            relative = path.relative_to(self.workspace).as_posix()
            return "./" + glob.escape(relative)
        if path.is_relative_to(self.temporary_directory):
            relative = path.relative_to(self.temporary_directory).as_posix()
            return "${TEMP}/" + glob.escape(relative)
        return glob.escape(path.as_posix())


def _load_file(
    path: Path,
    local: bool,
    workspace: Path,
    temporary_directory: Path,
    known_tools: set[str],
) -> tuple[
    list[_Rule],
    dict[str, list[dict[str, str]]],
    PermissionMode,
]:
    if not path.exists():
        return [], {}, DEFAULT_PERMISSION_MODE
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Permission file '{path}' has invalid YAML: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not read permission file '{path}': {exc}") from exc
    if loaded is None:
        return [], {}, DEFAULT_PERMISSION_MODE
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Permission file '{path}' must contain a YAML mapping.")
    return _load_tool_rules(
        loaded,
        path,
        local,
        workspace,
        temporary_directory,
        known_tools,
    )


def _permission_directory(directory: Path, root: Path, name: str) -> Path:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        resolved = directory.resolve()
    except OSError as exc:
        raise RuntimeError(
            f"Could not initialize permission directory '{directory}': {exc}"
        ) from exc
    if not resolved.is_relative_to(root):
        raise RuntimeError(
            f"Permission directory '{directory}' resolves outside the {name}."
        )
    return resolved


def _initialize_file(path: Path, known_tools: set[str]) -> None:
    if path.exists():
        return

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".permissions.", suffix=".yaml", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(
                {tool_name: [] for tool_name in sorted(known_tools)},
                stream,
                allow_unicode=True,
                sort_keys=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Could not initialize permission file '{path}': {exc}"
        ) from exc
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _load_tool_rules(
    loaded: dict[object, object],
    path: Path,
    local: bool,
    workspace: Path,
    temporary_directory: Path,
    known_tools: set[str],
) -> tuple[
    list[_Rule],
    dict[str, list[dict[str, str]]],
    PermissionMode,
]:
    rules: list[_Rule] = []
    data = {tool_name: [] for tool_name in sorted(known_tools)}
    permission_mode = DEFAULT_PERMISSION_MODE
    for tool_name, entries in loaded.items():
        if tool_name == "permission_mode":
            if not local:
                raise RuntimeError(
                    f"Permission file '{path}' may only set permission_mode locally."
                )
            if not isinstance(entries, str) or entries not in PERMISSION_MODES:
                raise RuntimeError(
                    f"Permission file '{path}' has invalid permission_mode '{entries}'."
                )
            permission_mode = entries
            continue
        if not isinstance(tool_name, str) or tool_name not in known_tools:
            raise RuntimeError(
                f"Permission file '{path}' references unknown tool '{tool_name}'."
            )
        if not isinstance(entries, list):
            raise RuntimeError(
                f"Permission file '{path}' field '{tool_name}' must be a list."
            )
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"content", "action"}:
                raise RuntimeError(
                    f"Permission file '{path}' rules must contain exactly "
                    "'content' and 'action'."
                )
            content = entry["content"]
            action = entry["action"]
            if not isinstance(content, str) or not content:
                raise RuntimeError(
                    f"Permission file '{path}' has a rule with invalid content."
                )
            if not isinstance(action, str) or action not in _ACTION_BY_NAME:
                raise RuntimeError(
                    f"Permission file '{path}' has invalid action '{action}'."
                )
            normalized = {"content": content, "action": action}
            if normalized in data[tool_name]:
                continue
            data[tool_name].append(normalized)
            rules.append(
                _build_rule(
                    _ACTION_BY_NAME[action],
                    tool_name,
                    content,
                    path,
                    local,
                    workspace,
                    temporary_directory,
                )
            )
    return rules, data, permission_mode


def _build_rule(
    action: RuleAction,
    tool_name: str,
    pattern: str,
    path: Path,
    local: bool,
    workspace: Path,
    temporary_directory: Path,
) -> _Rule:
    text = f"{tool_name}({pattern})"
    if tool_name in _PATH_TOOLS:
        if pattern.startswith("./"):
            pattern = (workspace / pattern[2:]).as_posix()
        elif pattern == "${TEMP}":
            pattern = temporary_directory.as_posix()
        elif pattern.startswith("${TEMP}/"):
            pattern = (
                temporary_directory / pattern.removeprefix("${TEMP}/")
            ).as_posix()
        else:
            pattern = pattern.replace("\\", "/")
    return _Rule(action, tool_name, pattern, text, local)


def _matches(rule: _Rule, content: str) -> bool:
    if rule.tool_name == "Bash":
        return fnmatchcase(content, rule.pattern)
    return _matches_glob(
        PurePosixPath(content).parts,
        PurePosixPath(rule.pattern).parts,
    )
