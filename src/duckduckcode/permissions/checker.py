from __future__ import annotations

from collections.abc import Callable, Iterable

from ..tools.tool import ToolCall

PermissionRule = Callable[[ToolCall], str | None]


class PermissionChecker:
    def __init__(self, rules: Iterable[PermissionRule] = ()) -> None:
        self._rules = tuple(rules)

    def check(self, tool_call: ToolCall) -> str | None:
        for rule in self._rules:
            denial = rule(tool_call)
            if denial is not None:
                return denial
        return None

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
