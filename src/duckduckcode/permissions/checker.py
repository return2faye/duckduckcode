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
