from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import glob
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Literal

import yaml

from ..tools.glob import _matches_glob
from ..tools.tool import ToolCall

PermissionAction = Literal["deny", "ask", "allow", "unspecified"]
RuleAction = Literal["deny", "ask", "allow"]
_ACTIONS: tuple[RuleAction, ...] = ("deny", "ask", "allow")
_PATH_TOOLS = frozenset({"ReadFile", "WriteFile", "EditFile", "Glob", "Grep"})
_RULE_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\((.*)\)", re.DOTALL)
_EMPTY_PERMISSIONS = "deny: []\nask: []\nallow: []\n"


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
        known_tools: set[str],
        rules: list[_Rule],
        local_file: Path,
        local_data: dict[str, list[str]],
    ) -> None:
        self.workspace = workspace.resolve()
        self.temporary_directory = temporary_directory.resolve()
        self.known_tools = known_tools
        self.rules = rules
        self.local_file = local_file
        self._local_data = local_data

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
            _initialize_file(path)
        rules: list[_Rule] = []
        local_data: dict[str, list[str]] = {}
        for path, local in sources:
            data = _load_file(path)
            if local:
                local_data = data
            for action in _ACTIONS:
                for text in data.get(action, []):
                    rules.append(
                        _parse_rule(
                            action,
                            text,
                            path,
                            local,
                            workspace,
                            temporary_directory,
                            known_tools,
                        )
                    )
        return cls(
            workspace,
            temporary_directory,
            known_tools,
            rules,
            local_file,
            local_data,
        )

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
        text = f"{tool_call.name}({pattern})"
        local_data = {
            action: list(values) for action, values in self._local_data.items()
        }
        allowed = local_data.setdefault("allow", [])
        if text in allowed:
            return
        self.local_file.parent.mkdir(parents=True, exist_ok=True)
        permission_directory = self.local_file.parent.resolve()
        if not permission_directory.is_relative_to(self.workspace):
            raise RuntimeError(
                f"Permission file '{self.local_file}' resolves outside the workspace."
            )
        allowed.append(text)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".permissions.", suffix=".yaml", dir=permission_directory
        )
        target = permission_directory / self.local_file.name
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                yaml.safe_dump(
                    local_data,
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
        self._local_data = local_data
        self.rules.append(
            _parse_rule(
                "allow",
                text,
                self.local_file,
                True,
                self.workspace,
                self.temporary_directory,
                self.known_tools,
            )
        )

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


def _load_file(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Permission file '{path}' has invalid YAML: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not read permission file '{path}': {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Permission file '{path}' must contain a YAML mapping.")
    unknown = set(loaded) - set(_ACTIONS)
    if unknown:
        raise RuntimeError(
            f"Permission file '{path}' has unknown field(s): "
            + ", ".join(sorted(map(str, unknown)))
        )
    data: dict[str, list[str]] = {}
    for action, values in loaded.items():
        if not isinstance(action, str):
            raise RuntimeError(f"Permission file '{path}' has a non-string field name.")
        if not isinstance(values, list):
            raise RuntimeError(
                f"Permission file '{path}' field '{action}' must be a list."
            )
        if not all(isinstance(value, str) for value in values):
            raise RuntimeError(
                f"Permission file '{path}' field '{action}' must contain strings."
            )
        data[action] = list(dict.fromkeys(values))
    return data


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


def _initialize_file(path: Path) -> None:
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".permissions.", suffix=".yaml", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_EMPTY_PERMISSIONS)
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


def _parse_rule(
    action: RuleAction,
    text: str,
    path: Path,
    local: bool,
    workspace: Path,
    temporary_directory: Path,
    known_tools: set[str],
) -> _Rule:
    match = _RULE_PATTERN.fullmatch(text)
    if match is None or not match.group(2):
        raise RuntimeError(f"Permission file '{path}' has invalid rule '{text}'.")
    tool_name, pattern = match.groups()
    if tool_name not in known_tools:
        raise RuntimeError(
            f"Permission file '{path}' references unknown tool '{tool_name}'."
        )
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
