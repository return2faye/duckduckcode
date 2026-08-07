from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
import shlex
from typing import Protocol

from ..tools.tool import Tool, ToolCall
from .rule_policy import (
    DEFAULT_PERMISSION_MODE,
    PermissionDecision,
    PermissionMode,
)

PermissionRule = Callable[[ToolCall], str | None]
_READ_ONLY_GIT_COMMANDS = {
    "status",
    "diff",
    "log",
    "show",
    "rev-parse",
    "ls-files",
    "grep",
}
_UNSAFE_GIT_OPTIONS = (
    "--ext-diff",
    "--open-files-in-pager",
    "--output",
    "--textconv",
)


class PermissionPolicy(Protocol):
    permission_mode: PermissionMode

    def check(
        self, tool_call: ToolCall, *, tool: Tool | None = None
    ) -> PermissionDecision: ...

    def remember_allow(
        self, tool_call: ToolCall, *, tool: Tool | None = None
    ) -> None: ...

    def set_permission_mode(self, mode: PermissionMode) -> None: ...


class PermissionChecker:
    def __init__(
        self,
        rules: Iterable[PermissionRule] = (),
        policy: PermissionPolicy | None = None,
    ) -> None:
        self._rules = tuple(rules)
        self._policy = policy

    def check(
        self,
        tool_call: ToolCall,
        *,
        tool: Tool | None = None,
        plan_file: Path | None = None,
    ) -> PermissionDecision:
        for rule in self._rules:
            denial = rule(tool_call)
            if denial is not None:
                return PermissionDecision("deny", denial)
        if self._policy is not None:
            decision = self._policy.check(tool_call, tool=tool)
        else:
            decision = PermissionDecision("unspecified")
        if decision.action == "deny":
            return decision
        if plan_file is not None:
            if tool is not None and _is_plan_safe(tool_call, tool, plan_file):
                return PermissionDecision("allow", content=decision.content)
            return PermissionDecision(
                "deny",
                "Plan Mode is active. Write or update the plan file, then call "
                "ExitPlanMode. Business files cannot be modified before plan approval.",
                decision.content or _content(tool_call),
            )
        if tool is None or self._policy is None:
            return decision
        if (
            self.permission_mode == "full_access"
            or tool.is_read_only
            or (self.permission_mode == "accept_edits" and tool.category == "file")
        ):
            return PermissionDecision("allow", content=decision.content)
        if decision.action == "allow":
            return decision
        return PermissionDecision(
            "ask",
            decision.message or "Permission approval required by permission mode.",
            decision.content or _content(tool_call),
        )

    @property
    def permission_mode(self) -> PermissionMode:
        if self._policy is None:
            return DEFAULT_PERMISSION_MODE
        return getattr(self._policy, "permission_mode", DEFAULT_PERMISSION_MODE)

    def set_permission_mode(self, mode: PermissionMode) -> None:
        if self._policy is None:
            raise RuntimeError("No permission policy is configured.")
        self._policy.set_permission_mode(mode)

    def remember_allow(self, tool_call: ToolCall, *, tool: Tool | None = None) -> None:
        if self._policy is None:
            raise RuntimeError("No permission policy is configured.")
        self._policy.remember_allow(tool_call, tool=tool)

    def start_task(self) -> None:
        for rule in self._rules:
            start_task = getattr(rule, "start_task", None)
            if callable(start_task):
                start_task()

    def finish_task(self) -> None:
        for rule in reversed(self._rules):
            finish_task = getattr(rule, "finish_task", None)
            if callable(finish_task):
                finish_task()

    def close(self) -> None:
        for rule in reversed(self._rules):
            close = getattr(rule, "close", None)
            if callable(close):
                close()


def _is_plan_safe(tool_call: ToolCall, tool: Tool, plan_file: Path) -> bool:
    if tool.is_read_only:
        return True
    if tool_call.name in {"WriteFile", "EditFile"}:
        path = tool_call.arguments.get("path")
        if not isinstance(path, str):
            return False
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = plan_file.parent.parent / candidate
        return candidate.resolve() == plan_file.resolve()
    if tool_call.name != "Bash":
        return False
    if tool_call.arguments.get("network_access", False) is not False:
        return False
    command = tool_call.arguments.get("command")
    if not isinstance(command, str) or any(
        token in command for token in ("\n", ";", "&", "|", ">", "<", "`", "$(")
    ):
        return False
    try:
        arguments = shlex.split(command)
    except ValueError:
        return False
    if arguments == ["pwd"]:
        return True
    command_index = (
        2
        if len(arguments) >= 3
        and arguments[0] == "git"
        and arguments[1] == "--no-pager"
        else 1
    )
    return (
        len(arguments) > command_index
        and arguments[0] == "git"
        and arguments[command_index] in _READ_ONLY_GIT_COMMANDS
        and not any(
            argument.startswith(_UNSAFE_GIT_OPTIONS)
            for argument in arguments[command_index + 1 :]
        )
    )


def _content(tool_call: ToolCall) -> str:
    for name in ("command", "path"):
        value = tool_call.arguments.get(name)
        if isinstance(value, str):
            return value
    return tool_call.name
