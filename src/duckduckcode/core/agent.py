from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Literal

from .client import Client
from .context import (
    COMPACTION_OUTPUT_TOKENS,
    CONTEXT_SAFETY_TOKENS,
    ContextManager,
    Message,
)
from .event import (
    AgentEvent,
    ConversationEvent,
    ContextCompactionEvent,
    ContextStatusEvent,
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
from .prompts import COMPACTION_SYSTEM_PROMPT
from ..permissions import PermissionChecker, PermissionMode
from ..tools.tool import ToolCall, ToolManager, ToolResult

AgentResponse = PermissionChoice | PlanReviewResponse | None
COMPACTION_MAX_ATTEMPTS = 3
MAX_CONTEXT_LENGTH_RECOVERIES = 1


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
        self._compaction_circuit_open = False

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

    def set_permission_mode(self, mode: PermissionMode) -> None:
        self.permission_checker.set_permission_mode(mode)

    def compact(self) -> Generator[AgentEvent, None, None]:
        self._compaction_circuit_open = False
        self.context.set_tool_schemas(self.tools.schemas())
        status = yield from self._compact(automatic=False)
        yield LoopCompleteEvent("error" if status == "error" else "completed", 0)

    def context_status(self) -> Generator[AgentEvent, None, None]:
        self.context.set_tool_schemas(self.tools.schemas())
        yield ContextStatusEvent(
            self.context.estimated_tokens(),
            self.context.context_window_tokens,
            self.context.auto_compact_tokens,
        )
        yield LoopCompleteEvent("completed", 0)

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
        context_length_recoveries = 0

        for iteration in range(1, self.max_iterations + 1):
            if self.context.should_compact():
                yield from self._compact(automatic=True)
            if self.context.estimated_tokens() >= self.context.context_window_tokens:
                yield ErrorEvent(
                    "Context is too large after compaction. Start a new conversation "
                    "or shorten the current request.",
                    "context_length",
                )
                yield LoopCompleteEvent("error", iteration)
                return
            messages = self.context.model_messages()
            assistant_index = self.context.start_assistant_stream()
            tool_calls = []
            stream_finished = False
            token_usage = 0
            api_error: ErrorEvent | None = None

            try:
                try:
                    try:
                        events = self.client.stream(
                            messages,
                            tools=self.context.tool_schemas(),
                            reasoning=self.context.reasoning,
                        )
                        try:
                            for event in events:
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
                                    api_error = event
                                    break
                        finally:
                            close = getattr(events, "close", None)
                            if close is not None:
                                close()
                    except Exception as exc:
                        api_error = _error_from_exception(exc)

                    if api_error is not None:
                        assistant = self.context.messages()[assistant_index]
                        can_recover = (
                            _is_context_length_error(api_error)
                            and context_length_recoveries
                            < MAX_CONTEXT_LENGTH_RECOVERIES
                            and not assistant.content
                            and not tool_calls
                        )
                        if can_recover:
                            self.context.discard_assistant_stream(assistant_index)
                            stream_finished = True
                            status = yield from self._compact(automatic=True)
                            if status == "completed":
                                context_length_recoveries += 1
                                continue
                            if status == "error":
                                yield LoopCompleteEvent("error", iteration)
                                return
                        else:
                            self.context.fail_assistant_stream(assistant_index)
                            stream_finished = True
                        yield api_error
                        yield LoopCompleteEvent("error", iteration)
                        return

                    completed_results = []
                    if tool_calls:
                        completed_results = yield from self._execute_tools(tool_calls)
                        results_by_id = {
                            tool_call.call_id: (tool_call, result)
                            for tool_call, result in completed_results
                        }
                        completed_results = [
                            results_by_id[tool_call.call_id]
                            for tool_call in tool_calls
                            if tool_call.call_id in results_by_id
                        ]
                        completed_results = self.context.compact_tool_results(
                            completed_results
                        )
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

    def _compact(
        self, automatic: bool
    ) -> Generator[AgentEvent, None, Literal["completed", "skipped", "error"]]:
        before_tokens = self.context.estimated_tokens()
        if automatic and self._compaction_circuit_open:
            return "skipped"
        candidate = self.context.compaction_input()
        if candidate is None:
            if not automatic:
                yield ContextCompactionEvent(
                    "skipped",
                    automatic,
                    before_tokens,
                    before_tokens,
                )
            return "skipped"

        transcript, cutoff = candidate
        messages = [
            Message("system", COMPACTION_SYSTEM_PROMPT),
            Message("user", transcript),
        ]
        tools = self.context.tool_schemas()
        input_tokens = self.context.estimate_request_tokens(messages, tools)
        max_output_tokens = min(
            COMPACTION_OUTPUT_TOKENS,
            self.context.context_window_tokens - input_tokens - CONTEXT_SAFETY_TOKENS,
        )
        if max_output_tokens < 1:
            self._compaction_circuit_open = True
            yield ErrorEvent(
                "Context compaction cannot fit inside the configured context "
                "window. Start a new conversation or increase "
                "CONTEXT_WINDOW_TOKENS.",
                "context_compaction",
            )
            return "error"

        yield ContextCompactionEvent("started", automatic, before_tokens)
        usage = 0
        last_error: Exception | None = None
        for _ in range(COMPACTION_MAX_ATTEMPTS):
            input_tokens = self.context.estimate_request_tokens(messages, tools)
            max_output_tokens = min(
                COMPACTION_OUTPUT_TOKENS,
                self.context.context_window_tokens
                - input_tokens
                - CONTEXT_SAFETY_TOKENS,
            )
            if max_output_tokens < 1:
                last_error = RuntimeError(
                    "tool-call feedback exhausted the context window"
                )
                continue
            output = ""
            completed = False
            attempt_usage = 0
            tool_calls: list[ToolCall] = []
            try:
                for event in self.client.stream(
                    messages,
                    tools=tools,
                    reasoning=self.context.reasoning,
                    max_output_tokens=max_output_tokens,
                ):
                    if isinstance(event, ConversationEvent):
                        output += event.delta
                    elif isinstance(event, ToolCallEvent):
                        tool_calls.append(event.tool_call)
                    elif isinstance(event, ErrorEvent):
                        raise RuntimeError(event.message)
                    elif isinstance(event, DoneEvent):
                        attempt_usage = event.token_usage
                        completed = True
                        break
                if not completed:
                    raise RuntimeError("the model response did not complete")
                if tool_calls:
                    for tool_call in tool_calls:
                        messages.append(
                            Message.tool_call(
                                tool_call.call_id,
                                tool_call.name,
                                tool_call.arguments,
                            )
                        )
                    for tool_call in tool_calls:
                        messages.append(
                            Message.tool_result(
                                tool_call.call_id,
                                ToolResult(
                                    "Tools are unavailable during context compaction. "
                                    "Return only the required <analysis> and <summary> "
                                    "sections.",
                                    is_error=True,
                                ).to_model_output(),
                            )
                        )
                    raise RuntimeError("the model attempted to call a tool")
                self.context.apply_compaction(_extract_summary(output), cutoff)
                usage += attempt_usage
                self._compaction_circuit_open = False
                break
            except Exception as exc:
                usage += attempt_usage
                last_error = exc
        else:
            self._compaction_circuit_open = True
            if usage:
                yield UsageEvent(usage)
            yield ErrorEvent(
                "Context compaction failed after 3 attempts: "
                f"{last_error}. Automatic compaction is disabled until "
                "/compact is run.",
                "context_compaction",
            )
            return "error"

        if usage:
            yield UsageEvent(usage)
        yield ContextCompactionEvent(
            "completed",
            automatic,
            before_tokens,
            self.context.estimated_tokens(),
        )
        return "completed"

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
            get_tool = getattr(self.tools, "get", None)
            tool = get_tool(tool_call.name) if callable(get_tool) else None
            if self.context.mode == "plan" and tool is not None:
                decision = self.permission_checker.check(
                    tool_call,
                    tool=tool,
                    plan_file=self.plan_file,
                )
            else:
                decision = self.permission_checker.check(tool_call, tool=tool)
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
        try:
            self.context.close()
        finally:
            self.permission_checker.close()


def _extract_summary(output: str) -> str:
    analysis_start = output.find("<analysis>")
    analysis_end = output.find("</analysis>", analysis_start + len("<analysis>"))
    summary_start = output.find("<summary>", analysis_end + len("</analysis>"))
    summary_end = output.find("</summary>", summary_start + len("<summary>"))
    if min(analysis_start, analysis_end, summary_start, summary_end) < 0:
        raise RuntimeError("the model did not return analysis and summary sections")
    summary = output[summary_start + len("<summary>") : summary_end].strip()
    if not summary:
        raise RuntimeError("the model returned an empty summary")
    return summary


def _error_from_exception(exc: Exception) -> ErrorEvent:
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if code is None and isinstance(body, dict):
        nested = body.get("error")
        code = body.get("code") or (
            nested.get("code") if isinstance(nested, dict) else None
        )
    return ErrorEvent(str(exc), str(code) if code is not None else None)


def _is_context_length_error(error: ErrorEvent) -> bool:
    text = f"{error.code or ''} {error.message}".lower().replace("-", "_")
    return any(
        marker in text
        for marker in (
            "context_length_exceeded",
            "context_length",
            "context window",
            "maximum context length",
            "prompt_too_long",
            "prompt too long",
            "prompt is too long",
            "too many tokens",
            "input too long",
            "exceeds the maximum number of tokens",
        )
    )
