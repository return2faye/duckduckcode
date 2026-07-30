from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from .client import Client
from .context import ContextManager
from .event import (
    AgentEvent,
    ConversationEvent,
    DoneEvent,
    ErrorEvent,
    LoopCompleteEvent,
    PermissionChoice,
    PermissionRequestEvent,
    PlanReviewEvent,
    PlanReviewResponse,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    UsageEvent,
)
from ..permissions import PermissionChecker
from ..tools.tool import ToolCall, ToolManager, ToolResult

AgentResponse = PermissionChoice | PlanReviewResponse | None


class Agent:
    def __init__(
        self,
        client: Client,
        context: ContextManager | None = None,
        tools: ToolManager | None = None,
        max_iterations: int = 50,
        permission_checker: PermissionChecker | None = None,
        plan_file: str | Path | None = None,
    ) -> None:
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or not 1 <= max_iterations <= 50
        ):
            raise ValueError("max_iterations must be between 1 and 50")
        self.client = client
        self.context = context or ContextManager()
        self.tools = tools or ToolManager()
        self.max_iterations = max_iterations
        self.permission_checker = permission_checker or PermissionChecker()
        self.plan_file = Path(plan_file).resolve() if plan_file is not None else None

    def enter_plan_mode(self, plan_file: str | Path | None = None) -> None:
        if plan_file is not None:
            self.plan_file = Path(plan_file).resolve()
        if self.plan_file is None:
            raise RuntimeError("Plan file is not configured.")
        self._remove_plan_file()
        self.context.set_mode("plan")

    def cancel_plan_mode(self) -> None:
        self._remove_plan_file()
        self.context.set_mode("default")
        self.context.add_user(
            "Plan Mode was cancelled. Do not execute the pending plan "
            "unless the user asks again."
        )

    def stream(self, user_message: str) -> Generator[AgentEvent, AgentResponse, None]:
        self.permission_checker.start_task()
        try:
            yield from self._stream(user_message)
        finally:
            self.permission_checker.finish_task()

    def _stream(self, user_message: str) -> Generator[AgentEvent, AgentResponse, None]:
        self.context.add_user(user_message)
        self.context.set_tool_schemas(self.tools.schemas())
        plan_execution_pending = False
        approved_plan_active = False
        plan_execution_failed = False

        for iteration in range(1, self.max_iterations + 1):
            messages = self.context.model_messages()
            assistant_index = self.context.start_assistant_stream()
            tool_calls = []
            stream_finished = False
            token_usage = 0

            try:
                try:
                    for event in self.client.stream(
                        messages,
                        tools=self.context.tool_schemas(),
                        reasoning=self.context.reasoning,
                    ):
                        if isinstance(event, ConversationEvent):
                            self.context.append_assistant_delta(
                                assistant_index, event.delta
                            )
                            yield event
                        elif isinstance(event, ToolCallEvent):
                            tool_calls.append(event.tool_call)
                            yield event
                        elif isinstance(event, DoneEvent):
                            self.context.finish_assistant_stream(
                                assistant_index, event.token_usage
                            )
                            stream_finished = True
                            token_usage = event.token_usage
                            break
                        elif isinstance(event, ErrorEvent):
                            self.context.fail_assistant_stream(assistant_index)
                            stream_finished = True
                            yield event
                            yield LoopCompleteEvent("error", iteration)
                            return

                    completed_results = []
                    if tool_calls:
                        completed_results = yield from self._execute_tools(tool_calls)
                        for tool_call, result in completed_results:
                            yield ToolResultEvent(
                                tool_call.call_id,
                                tool_call.name,
                                result.content,
                                result.is_error,
                            )
                    for tool_call in tool_calls:
                        self.context.add_tool_call(tool_call)
                    for tool_call, result in completed_results:
                        self.context.add_tool_result(
                            tool_call.call_id,
                            result.to_model_output(),
                        )
                    plan_approved = self.context.mode == "default" and any(
                        tool_call.name == "ExitPlanMode" and not result.is_error
                        for tool_call, result in completed_results
                    )
                    if plan_approved:
                        plan_execution_pending = True
                        approved_plan_active = True
                    if approved_plan_active and any(
                        tool_call.name != "ExitPlanMode" and result.is_error
                        for tool_call, result in completed_results
                    ):
                        plan_execution_failed = True
                    if plan_execution_pending and any(
                        tool_call.name != "ExitPlanMode" for tool_call in tool_calls
                    ):
                        plan_execution_pending = False
                    yield UsageEvent(token_usage)
                    yield TurnCompleteEvent(iteration)
                except KeyboardInterrupt:
                    self.context.fail_assistant_stream(assistant_index)
                    stream_finished = True
                    yield ErrorEvent("interrupted", "interrupted")
                    yield LoopCompleteEvent("cancelled", iteration)
                    return
            finally:
                if not stream_finished:
                    self.context.fail_assistant_stream(assistant_index)

            if not tool_calls:
                if plan_execution_pending:
                    if iteration == self.max_iterations:
                        yield ErrorEvent(
                            f"Agent stopped after {self.max_iterations} iterations.",
                            "max_iterations",
                        )
                        yield LoopCompleteEvent("max_iterations", iteration)
                        return
                    self.context.add_user(
                        "The plan is approved, but execution has not started. "
                        "Begin executing it now with the required tools. "
                        "Do not only describe what you will do."
                    )
                    continue
                if approved_plan_active and not plan_execution_failed:
                    self._remove_plan_file()
                yield LoopCompleteEvent("completed", iteration)
                return
            if iteration == self.max_iterations:
                yield ErrorEvent(
                    f"Agent stopped after {self.max_iterations} iterations.",
                    "max_iterations",
                )
                yield LoopCompleteEvent("max_iterations", iteration)
                return

    def _execute_tools(self, tool_calls: list[ToolCall]) -> Generator[
        PermissionRequestEvent | PlanReviewEvent,
        AgentResponse,
        list[tuple[ToolCall, ToolResult]],
    ]:
        allowed: list[ToolCall] = []
        completed: list[tuple[ToolCall, ToolResult]] = []
        for tool_call in tool_calls:
            if tool_call.name == "ExitPlanMode":
                completed.append((tool_call, (yield from self._review_plan())))
                continue
            if self.context.mode == "plan":
                tool = self.tools.get(tool_call.name)
                if tool is None:
                    allowed.append(tool_call)
                    continue
                decision = self.permission_checker.check(
                    tool_call,
                    tool=tool,
                    plan_file=self.plan_file,
                )
            else:
                decision = self.permission_checker.check(tool_call)
            if decision.action in {"allow", "unspecified"}:
                allowed.append(tool_call)
                continue
            if decision.action == "ask":
                choice = yield PermissionRequestEvent(
                    tool_call.call_id,
                    tool_call.name,
                    decision.content,
                    decision.message,
                )
                if choice == "allow_always":
                    self.permission_checker.remember_allow(tool_call)
                    allowed.append(tool_call)
                elif choice == "allow_once":
                    allowed.append(tool_call)
                else:
                    completed.append(
                        (
                            tool_call,
                            ToolResult(
                                "Permission denied by user.",
                                is_error=True,
                            ),
                        )
                    )
                continue
            completed.append((tool_call, ToolResult(decision.message, is_error=True)))
        completed.extend(self.tools.execute_many(allowed))
        return completed

    def _review_plan(
        self,
    ) -> Generator[PlanReviewEvent, PlanReviewResponse | None, ToolResult]:
        if self.context.mode != "plan" or self.plan_file is None:
            return ToolResult("Plan Mode is not active.", is_error=True)
        if not self.plan_file.is_file():
            return ToolResult(
                f"Plan file '{self.plan_file}' does not exist. Write the plan first.",
                is_error=True,
            )
        try:
            content = self.plan_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(f"Could not read plan file: {exc}", is_error=True)
        response = yield PlanReviewEvent(str(self.plan_file), content)
        if response is not None and response.approved:
            self.context.set_mode("default")
            return ToolResult(
                "Plan approved. Plan Mode is now inactive; execute the plan."
            )
        feedback = response.feedback.strip() if response is not None else ""
        if feedback:
            return ToolResult(f"User feedback: {feedback}")
        return ToolResult("Plan review cancelled; continue planning.", is_error=True)

    def _remove_plan_file(self) -> None:
        if self.plan_file is not None:
            self.plan_file.unlink(missing_ok=True)

    def close(self) -> None:
        self.permission_checker.close()
