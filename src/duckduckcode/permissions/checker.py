from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

from ..tools.tool import ToolCall
from .rule_policy import PermissionDecision

PermissionRule = Callable[[ToolCall], str | None]


class PermissionPolicy(Protocol):
    def check(self, tool_call: ToolCall) -> PermissionDecision: ...

    def remember_allow(self, tool_call: ToolCall) -> None: ...


class PermissionChecker:
    def __init__(
        self,
        rules: Iterable[PermissionRule] = (),
        policy: PermissionPolicy | None = None,
    ) -> None:
        self._rules = tuple(rules)
        self._policy = policy

    def check(self, tool_call: ToolCall) -> PermissionDecision:
        for rule in self._rules:
            denial = rule(tool_call)
            if denial is not None:
                return PermissionDecision("deny", denial)
        if self._policy is not None:
            return self._policy.check(tool_call)
        return PermissionDecision("unspecified")

    def remember_allow(self, tool_call: ToolCall) -> None:
        if self._policy is None:
            raise RuntimeError("No permission policy is configured.")
        self._policy.remember_allow(tool_call)

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
