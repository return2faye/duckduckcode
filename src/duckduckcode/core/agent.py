from __future__ import annotations

from collections.abc import Generator
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from .client import Client
from .context import (
    COMPACTION_OUTPUT_TOKENS,
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
    SessionListEvent,
    SessionStateEvent,
    SkillListEvent,
    SubagentEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    UsageEvent,
)
from .prompts import COMPACTION_SYSTEM_PROMPT
from .skill import SkillManager
from .lsp import LSPManager
from .mcp import MCPManager
from .subagent import DefinitionManager, SubagentManager
from ..permissions import PermissionChecker, PermissionMode
from ..tools.tool import (
    QuerySource,
    ToolCall,
    ToolManager,
    ToolResult,
    create_agent_tool,
    validate_agent_arguments,
    validate_load_skill_arguments,
)

AgentResponse = PermissionChoice | PlanReviewResponse | None
COMPACTION_MAX_ATTEMPTS = 3
MAX_CONTEXT_LENGTH_RECOVERIES = 1
MAX_IDENTICAL_TOOL_FAILURES = 2


class Agent:
    def __init__(
        self,
        client: Client,
        context: ContextManager | None = None,
        tools: ToolManager | None = None,
        max_iterations: int = 50,
        permission_checker: PermissionChecker | None = None,
        plan_file: str | Path | None = None,
        session_manager: object | None = None,
        memory_manager: object | None = None,
        skill_manager: SkillManager | None = None,
        skill_root_callback: Any | None = None,
        fork_agent_factory: Callable[[], Agent] | None = None,
        definition_manager: DefinitionManager | None = None,
        subagent_manager: SubagentManager | None = None,
        query_source: QuerySource = QuerySource.USER,
        mcp_manager: MCPManager | None = None,
        lsp_manager: LSPManager | None = None,
        owns_lsp_manager: bool = False,
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
        self.session_manager = session_manager
        self.memory_manager = memory_manager
        self.skill_manager = skill_manager
        self._skill_root_callback = skill_root_callback
        self._fork_agent_factory = fork_agent_factory
        self.definition_manager = definition_manager
        self.subagent_manager = subagent_manager
        self.query_source = query_source
        self.mcp_manager = mcp_manager
        self._mcp_initialized = False
        self.lsp_manager = lsp_manager
        self._owns_lsp_manager = owns_lsp_manager
        self._lsp_initialized = False
        self._runtime_conversation_key = uuid4().hex
        self._skill_list_snapshot: tuple[dict[str, Any], ...] | None = None
        self._session_snapshot = (
            session_manager.start() if session_manager is not None else None
        )
        self._startup_checked = False
        self._compaction_circuit_open = False
        self._tool_failures: dict[tuple[str, str], int] = {}

    def enter_plan_mode(self, plan_file: str | Path | None = None) -> None:
        if plan_file is not None:
            self.plan_file = Path(plan_file).resolve()
        if self.plan_file is None:
            raise RuntimeError("Plan file is not configured.")
        self._remove_plan_file()
        self.context.set_mode("plan")

    def cancel_plan_mode(self) -> None:
        self._remove_plan_file()
        self._add_user(
            "Plan Mode was cancelled. Do not execute the pending plan "
            "unless the user asks again.",
            visible=False,
        )
        self.context.set_mode("default")

    def set_permission_mode(self, mode: PermissionMode) -> None:
        self.permission_checker.set_permission_mode(mode)

    def initialize(self) -> Generator[AgentEvent, AgentResponse, None]:
        if self._session_snapshot is not None:
            yield _session_state("initialized", self._session_snapshot)
        yield from self._refresh_skills(force=True)
        yield from self._refresh_definitions()
        yield from self._initialize_mcp()
        yield from self._initialize_lsp()
        yield from self._startup_compaction()
        yield LoopCompleteEvent("completed", 0)

    def list_sessions(self) -> Generator[AgentEvent, None, None]:
        manager = self._require_sessions()
        yield SessionListEvent(
            tuple(
                {
                    "id": info.id,
                    "created_at": info.created_at,
                    "last_activity": info.last_activity,
                    "status": info.status,
                    "active": info.active,
                }
                for info in manager.list()
            )
        )
        yield LoopCompleteEvent("completed", 0)

    def new_session(self) -> Generator[AgentEvent, None, None]:
        self._ensure_session_switch_allowed()
        snapshot = self._require_sessions().create()
        self._session_snapshot = snapshot
        yield from self._restore_mcp_session(snapshot)
        self._startup_checked = False
        yield _session_state("new", snapshot)
        yield from self._startup_compaction()
        yield LoopCompleteEvent("completed", 0)

    def resume_session(self, session_id: str) -> Generator[AgentEvent, None, None]:
        self._ensure_session_switch_allowed()
        snapshot = self._require_sessions().resume(session_id)
        self._session_snapshot = snapshot
        yield from self._restore_mcp_session(snapshot)
        self._startup_checked = False
        yield _session_state("resumed", snapshot)
        yield from self._startup_compaction()
        yield LoopCompleteEvent("completed", 0)

    def delete_session(
        self, session_id: str | None = None
    ) -> Generator[AgentEvent, None, None]:
        self._ensure_session_switch_allowed()
        if self.subagent_manager is not None:
            target = session_id or self._session_key()
            self.subagent_manager.terminate_session(target)
        snapshot = self._require_sessions().delete(session_id)
        self._session_snapshot = snapshot
        yield from self._restore_mcp_session(snapshot)
        self._startup_checked = False
        yield _session_state("deleted", snapshot)
        yield LoopCompleteEvent("completed", 0)

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

    def stream(
        self, user_message: str, selected_skills: list[str] | tuple[str, ...] = ()
    ) -> Generator[AgentEvent, AgentResponse, None]:
        self._tool_failures.clear()
        self.permission_checker.start_task()
        try:
            yield from self._stream(user_message, selected_skills)
        finally:
            self._clear_active_skills()
            self.permission_checker.finish_task()

    def _stream(
        self, user_message: str, selected_skills: list[str] | tuple[str, ...]
    ) -> Generator[AgentEvent, AgentResponse, None]:
        yield from self._initialize_mcp()
        yield from self._initialize_lsp()
        memory_start = (
            len(self.session_manager.snapshot().records)
            if self.memory_manager is not None and self.session_manager is not None
            else None
        )
        if self.memory_manager is not None:
            memory, warning = self.memory_manager.refresh()
            self.context.set_long_term_memory(memory)
            if warning:
                yield ErrorEvent(warning, "memory")
        yield from self._refresh_skills()
        yield from self._refresh_definitions()
        yield from self._drain_subagents()
        try:
            self._add_user(user_message)
        except Exception as exc:
            yield _persistence_error(exc)
            yield LoopCompleteEvent("error", 0)
            return
        self.context.set_tool_schemas(self.tools.schemas())
        if selected_skills:
            calls = []
            results_by_id: dict[str, tuple[ToolCall, ToolResult]] = {}
            for name in sorted(set(selected_skills)):
                calls.append(
                    ToolCall(
                        f"selected_skill_{uuid4().hex}",
                        "LoadSkill",
                        {"name": name, "task": user_message},
                    )
                )
            try:
                for tool_call in calls:
                    self._add_tool_call(tool_call)
                    yield ToolCallEvent(tool_call)
                completed = yield from self._execute_tools(calls)
                results_by_id = {
                    tool_call.call_id: (tool_call, result)
                    for tool_call, result in completed
                }
                for tool_call, result in completed:
                    self._add_tool_result(tool_call, result)
                    yield ToolResultEvent(
                        tool_call.call_id,
                        tool_call.name,
                        result.content,
                        result.is_error,
                    )
            except KeyboardInterrupt:
                for tool_call in calls:
                    if tool_call.call_id not in results_by_id:
                        self._add_tool_result(
                            tool_call,
                            ToolResult("Tool execution was interrupted.", True),
                        )
                yield ErrorEvent("interrupted", "interrupted")
                yield LoopCompleteEvent("cancelled", 0)
                return
            except Exception as exc:
                yield _persistence_error(exc)
                yield LoopCompleteEvent("error", 0)
                return
        plan_execution_pending = False
        approved_plan_active = False
        plan_execution_failed = False
        context_length_recoveries = 0

        for iteration in range(1, self.max_iterations + 1):
            yield from self._drain_subagents()
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
                                    try:
                                        self._finish_assistant(
                                            assistant_index,
                                            "completed",
                                            event.token_usage,
                                        )
                                    except Exception as exc:
                                        api_error = _persistence_error(exc)
                                        stream_finished = True
                                        break
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
                        if api_error.code == "session_persistence":
                            yield api_error
                            yield LoopCompleteEvent("error", iteration)
                            return
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
                            try:
                                self._finish_assistant(assistant_index, "error")
                            except Exception as exc:
                                stream_finished = True
                                yield _persistence_error(exc)
                                yield LoopCompleteEvent("error", iteration)
                                return
                            stream_finished = True
                        yield api_error
                        yield LoopCompleteEvent("error", iteration)
                        return

                    try:
                        if self.session_manager is not None:
                            for tool_call in tool_calls:
                                self._add_tool_call(tool_call)
                    except Exception as exc:
                        yield _persistence_error(exc)
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
                        try:
                            if self.session_manager is None:
                                for tool_call in tool_calls:
                                    self._add_tool_call(tool_call)
                            for tool_call, result in completed_results:
                                self._add_tool_result(tool_call, result)
                                yield ToolResultEvent(
                                    tool_call.call_id,
                                    tool_call.name,
                                    result.content,
                                    result.is_error,
                                )
                        except Exception as exc:
                            yield _persistence_error(exc)
                            yield LoopCompleteEvent("error", iteration)
                            return
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
                    try:
                        if not stream_finished:
                            self._finish_assistant(assistant_index, "error")
                        elif self.session_manager is None:
                            self.context.fail_assistant_stream(assistant_index)
                        elif self.session_manager is not None:
                            for tool_call in tool_calls:
                                self._add_tool_result(
                                    tool_call,
                                    ToolResult("Tool execution was interrupted.", True),
                                )
                    except Exception as exc:
                        stream_finished = True
                        yield _persistence_error(exc)
                        yield LoopCompleteEvent("error", iteration)
                        return
                    stream_finished = True
                    yield ErrorEvent("interrupted", "interrupted")
                    yield LoopCompleteEvent("cancelled", iteration)
                    return
            finally:
                if not stream_finished:
                    try:
                        self._finish_assistant(assistant_index, "error")
                    except Exception:
                        pass

            if not tool_calls:
                if plan_execution_pending:
                    if iteration == self.max_iterations:
                        yield ErrorEvent(
                            f"Agent stopped after {self.max_iterations} iterations.",
                            "max_iterations",
                        )
                        yield LoopCompleteEvent("max_iterations", iteration)
                        return
                    self._add_user(
                        "The plan is approved, but execution has not started. "
                        "Begin executing it now with the required tools. "
                        "Do not only describe what you will do.",
                        visible=False,
                    )
                    continue
                if approved_plan_active and not plan_execution_failed:
                    self._remove_plan_file()
                self._start_memory_worker(memory_start)
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
            self.context.context_window_tokens
            - input_tokens
            - self.context.compaction_safety_tokens,
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
                - self.context.compaction_safety_tokens,
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
                summary = _extract_summary(output)
                try:
                    self._apply_compaction(summary, cutoff, attempt_usage)
                except Exception as exc:
                    self._compaction_circuit_open = True
                    usage += attempt_usage
                    if usage:
                        yield UsageEvent(usage)
                    yield _persistence_error(exc)
                    return "error"
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
        AgentEvent,
        AgentResponse,
        list[tuple[ToolCall, ToolResult]],
    ]:
        allowed: list[ToolCall] = []
        completed: list[tuple[ToolCall, ToolResult]] = []
        for tool_call in tool_calls:
            guard_result = self._tool_loop_guard_result(tool_call)
            if guard_result is not None:
                completed.append((tool_call, guard_result))
                continue
            if tool_call.name == "ExitPlanMode":
                completed.append((tool_call, (yield from self._review_plan())))
                continue
            if (
                tool_call.name == "RemoveWorktree"
                and self.subagent_manager is not None
                and self.context.mode != "plan"
            ):
                try:
                    preflight = self.subagent_manager.worktree_manager.preflight_remove(
                        str(tool_call.arguments.get("id", ""))
                    )
                except Exception as exc:
                    completed.append((tool_call, ToolResult(str(exc), True)))
                    continue
                if preflight["dirty"]:
                    if self.query_source == QuerySource.SUBAGENT:
                        completed.append(
                            (
                                tool_call,
                                ToolResult(
                                    "Permission denied: subagents cannot remove dirty "
                                    "worktrees.",
                                    True,
                                ),
                            )
                        )
                        continue
                    choice = yield PermissionRequestEvent(
                        tool_call.call_id,
                        tool_call.name,
                        json.dumps(preflight, ensure_ascii=False),
                        "This worktree has uncommitted changes. Delete it after "
                        "capturing a final patch?",
                    )
                    if choice in {"allow_once", "allow_always"}:
                        allowed.append(tool_call)
                    else:
                        completed.append(
                            (
                                tool_call,
                                ToolResult("Worktree removal cancelled by user.", True),
                            )
                        )
                    continue
            get_tool = getattr(self.tools, "get", None)
            tool = get_tool(tool_call.name) if callable(get_tool) else None
            if (
                self.subagent_manager is not None
                and self.subagent_manager.workspace_busy
                and tool_call.name in {"WriteFile", "EditFile", "Bash"}
            ):
                completed.append(
                    (
                        tool_call,
                        ToolResult(
                            "The shared workspace is busy while a non-isolated fork "
                            "holds the write lease.",
                            True,
                        ),
                    )
                )
                continue
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
                if self.query_source == QuerySource.SUBAGENT:
                    completed.append(
                        (
                            tool_call,
                            ToolResult(
                                "Permission denied: subagents cannot request user approval.",
                                True,
                            ),
                        )
                    )
                    continue
                choice = yield PermissionRequestEvent(
                    tool_call.call_id,
                    tool_call.name,
                    decision.content,
                    decision.message,
                )
                if choice == "allow_always":
                    self.permission_checker.remember_allow(tool_call, tool=tool)
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
        skill_calls = sorted(
            (call for call in allowed if call.name == "LoadSkill"),
            key=lambda call: str(call.arguments.get("name", "")),
        )
        for tool_call in skill_calls:
            skill = (
                self.skill_manager.get(str(tool_call.arguments.get("name", "")))
                if self.skill_manager is not None
                else None
            )
            result = (
                (yield from self._run_fork_skill(tool_call))
                if skill is not None and skill.mode == "fork"
                else self.tools.execute(tool_call)
            )
            completed.append((tool_call, result))
            self._record_tool_result(tool_call, result)
            self._sync_active_skills()
        agent_calls = [call for call in allowed if call.name == "Agent"]
        for tool_call in agent_calls:
            result = yield from self._run_subagent(tool_call)
            completed.append((tool_call, result))
            self._record_tool_result(tool_call, result)
        for tool_call, result in self.tools.execute_many(
            [call for call in allowed if call.name not in {"LoadSkill", "Agent"}]
        ):
            completed.append((tool_call, result))
            self._record_tool_result(tool_call, result)
        self._sync_active_skills()
        if self.tools.dirty:
            self.context.set_tool_schemas(self.tools.schemas())
            self.tools.mark_clean()
        return completed

    def _tool_loop_guard_result(self, tool_call: ToolCall) -> ToolResult | None:
        failure_key = self._tool_failure_key(tool_call)
        if (
            failure_key is None
            or self._tool_failures.get(failure_key, 0) < MAX_IDENTICAL_TOOL_FAILURES
        ):
            return None
        return ToolResult(
            "Loop guard: this identical side-effecting tool call already failed "
            "twice. Inspect the current state or change the arguments before retrying.",
            True,
        )

    def _tool_failure_key(self, tool_call: ToolCall) -> tuple[str, str] | None:
        get_tool = getattr(self.tools, "get", None)
        tool = get_tool(tool_call.name) if callable(get_tool) else None
        if tool is None or tool.is_read_only:
            return None
        try:
            arguments = json.dumps(
                tool_call.arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError):
            return None
        return tool_call.name, arguments

    def _record_tool_result(self, tool_call: ToolCall, result: ToolResult) -> None:
        failure_key = self._tool_failure_key(tool_call)
        if failure_key is None:
            return
        if result.is_error:
            self._tool_failures[failure_key] = (
                self._tool_failures.get(failure_key, 0) + 1
            )
        else:
            self._tool_failures.clear()

    def list_skills(self) -> Generator[AgentEvent, None, None]:
        yield from self._refresh_skills(force=True)
        yield LoopCompleteEvent("completed", 0)

    def _refresh_skills(self, force: bool = False) -> Generator[AgentEvent, None, None]:
        if self.skill_manager is None:
            self.context.set_skill_catalog("")
            return
        skills, warning = self.skill_manager.refresh()
        self.context.set_skill_catalog(self.skill_manager.catalog_block())
        snapshot = tuple(
            {
                "name": skill.name,
                "description": skill.description,
                "mode": skill.mode,
                "scope": skill.scope,
            }
            for skill in skills
        )
        if force or snapshot != self._skill_list_snapshot:
            self._skill_list_snapshot = snapshot
            yield SkillListEvent(snapshot)
        if warning:
            yield ErrorEvent(warning, "skill")

    def _refresh_definitions(self) -> Generator[AgentEvent, None, None]:
        if self.definition_manager is None:
            return
        definitions, warning = self.definition_manager.refresh()
        self.tools.register(
            create_agent_tool(
                {definition.type: definition.when_to_use for definition in definitions},
                lambda **_: "Agent calls are handled by the Agent runtime.",
            )
        )
        self.context.set_tool_schemas(self.tools.schemas())
        if warning:
            yield ErrorEvent(warning, "subagent_definition")
        if self.subagent_manager is not None:
            worktree_warning = self.subagent_manager.startup_warning()
            if worktree_warning:
                yield ErrorEvent(worktree_warning, "worktree")

    def _run_subagent(
        self, tool_call: ToolCall
    ) -> Generator[AgentEvent, AgentResponse, ToolResult]:
        if self.query_source == QuerySource.SUBAGENT:
            return ToolResult("Subagents cannot invoke the Agent tool.", True)
        if self.subagent_manager is None:
            return ToolResult("Subagents are unavailable in this Agent.", True)
        tool = self.tools.get("Agent")
        try:
            schema = tool.schema() if tool is not None else {}
            values = (
                schema.get("parameters", {})
                .get("properties", {})
                .get("subagent_type", {})
                .get("enum", ())
            )
            definition_types = tuple(
                value for value in values if isinstance(value, str)
            )
            arguments = validate_agent_arguments(
                dict(tool_call.arguments), definition_types
            )
        except Exception as exc:
            return ToolResult(str(exc), True)
        return (
            yield from self.subagent_manager.run(
                tool_call.call_id,
                arguments,
                session_key=self._session_key(),
                context=self.context,
                permission_mode=self.permission_checker.permission_mode,
            )
        )

    def _drain_subagents(self) -> Generator[AgentEvent, None, None]:
        if self.subagent_manager is None:
            return
        events, messages = self.subagent_manager.drain(self._session_key())
        for event in events:
            yield event
        for message in messages:
            synthetic_call = ToolCall(uuid4().hex, "Agent", {})
            _, compacted = self.context.compact_tool_results(
                [(synthetic_call, ToolResult(message))]
            )[0]
            self._add_user(compacted.content, visible=False)

    def detach_subagent(self) -> bool:
        return (
            self.subagent_manager.detach_foreground()
            if self.subagent_manager is not None
            else False
        )

    def _session_key(self) -> str:
        if self.session_manager is not None:
            session_id = self.session_manager.current_session_id
            if session_id is not None:
                return session_id
        return self._runtime_conversation_key

    def _run_fork_skill(
        self, tool_call: ToolCall
    ) -> Generator[AgentEvent, AgentResponse, ToolResult]:
        if self.skill_manager is None or self._fork_agent_factory is None:
            return ToolResult("Fork Skills are unavailable in this Agent.", True)
        try:
            arguments = validate_load_skill_arguments(dict(tool_call.arguments))
        except Exception as exc:
            return ToolResult(str(exc), True)
        skill, block, error = self.skill_manager.load_fork(
            arguments["name"], arguments["task"]
        )
        if error is not None:
            return error
        assert skill is not None and block is not None

        messages = self.context.messages()
        user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].kind == "message" and messages[index].role == "user"
            ),
            None,
        )
        if user_index is None:
            return ToolResult("Fork Skill has no current user message.", True)

        child: Agent | None = None
        child_stream: Generator[AgentEvent, AgentResponse, None] | None = None
        child_errors: list[str] = []
        reason: str | None = None
        history = deepcopy(messages[:user_index])
        try:
            child = self._fork_agent_factory()
            child.context.system_prompt = _fork_system_prompt(
                self.context.system_prompt, child.context.system_prompt
            )
            child.context.long_term_memory = self.context.long_term_memory
            child.context.reasoning = self.context.reasoning
            child.context.restore(
                history,
                self.context.abstraction,
                self.context.reminder,
            )
            child.context.set_mode(self.context.mode)
            child.context.set_active_skills(block)
            child.context.set_tool_schemas(child.tools.schemas())
            if child._skill_root_callback is not None and skill.root is not None:
                child._skill_root_callback((skill.root,))
            if (
                child.permission_checker.permission_mode
                != self.permission_checker.permission_mode
            ):
                child.set_permission_mode(self.permission_checker.permission_mode)

            child_stream = child.stream(messages[user_index].content)
            try:
                event = next(child_stream)
                while True:
                    if isinstance(event, PermissionRequestEvent):
                        choice = yield PermissionRequestEvent(
                            f"{tool_call.call_id}/{event.call_id}",
                            event.name,
                            event.content,
                            event.message,
                        )
                        event = child_stream.send(choice)
                        continue
                    if isinstance(event, ToolCallEvent):
                        yield ToolCallEvent(
                            ToolCall(
                                f"{tool_call.call_id}/{event.tool_call.call_id}",
                                event.tool_call.name,
                                event.tool_call.arguments,
                            )
                        )
                    elif isinstance(event, ToolResultEvent):
                        yield ToolResultEvent(
                            f"{tool_call.call_id}/{event.call_id}",
                            event.name,
                            event.content,
                            event.is_error,
                        )
                    elif isinstance(event, UsageEvent):
                        yield event
                    elif isinstance(event, ErrorEvent):
                        child_errors.append(event.message)
                    elif isinstance(event, LoopCompleteEvent):
                        reason = event.reason
                    event = next(child_stream)
            except StopIteration:
                pass

            if reason == "cancelled":
                raise KeyboardInterrupt
            final = next(
                (
                    message.content
                    for message in reversed(child.context.messages()[len(history) :])
                    if message.kind == "message"
                    and message.role == "assistant"
                    and message.status == "completed"
                    and message.content.strip()
                ),
                "",
            )
            if reason == "completed" and final:
                return ToolResult(final)
            detail = child_errors[-1] if child_errors else reason or "no final response"
            return ToolResult(f"Fork Skill '{skill.name}' failed: {detail}.", True)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            return ToolResult(f"Fork Skill '{skill.name}' failed: {exc}", True)
        finally:
            if child_stream is not None:
                try:
                    child_stream.close()
                except Exception:
                    pass
            if child is not None:
                try:
                    child.close()
                except Exception:
                    pass

    def _sync_active_skills(self) -> None:
        if self.skill_manager is None:
            return
        self.context.set_active_skills(self.skill_manager.active_block())
        if self._skill_root_callback is not None:
            self._skill_root_callback(tuple(self.skill_manager.active_roots.values()))

    def _clear_active_skills(self) -> None:
        if self.skill_manager is not None:
            self.skill_manager.clear_active()
        self.context.set_active_skills("")
        if self._skill_root_callback is not None:
            self._skill_root_callback(())

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

    def _startup_compaction(self) -> Generator[AgentEvent, None, None]:
        if self._startup_checked:
            return
        self._startup_checked = True
        self.context.set_tool_schemas(self.tools.schemas())
        if self.context.should_compact():
            status = yield from self._compact(automatic=True)
            if status != "completed" or self.context.should_compact():
                yield ErrorEvent(
                    "Restored session still exceeds the context threshold. "
                    "Use /new to start an empty session.",
                    "context_length",
                )

    def _require_sessions(self) -> Any:
        if self.session_manager is None:
            raise RuntimeError("Session persistence is disabled.")
        return self.session_manager

    def _ensure_session_switch_allowed(self) -> None:
        if self.context.mode == "plan":
            raise RuntimeError("Session switching is unavailable in Plan Mode.")

    def _add_user(self, content: str, *, visible: bool = True) -> None:
        if self.session_manager is None:
            self.context.add_user(content)
        else:
            self.session_manager.commit_message("user", content, visible=visible)

    def _finish_assistant(
        self,
        index: int,
        status: Literal["completed", "error"],
        token_usage: int = 0,
    ) -> None:
        if self.session_manager is None:
            if status == "completed":
                self.context.finish_assistant_stream(index, token_usage)
            else:
                self.context.fail_assistant_stream(index)
        else:
            self.session_manager.commit_assistant_stream(index, status, token_usage)

    def _add_tool_call(self, tool_call: ToolCall) -> None:
        if self.session_manager is None:
            self.context.add_tool_call(tool_call)
        else:
            self.session_manager.commit_tool_call(tool_call)

    def _add_tool_result(self, tool_call: ToolCall, result: ToolResult) -> None:
        if self.session_manager is None:
            self.context.add_tool_result(tool_call.call_id, result.to_model_output())
        else:
            self.session_manager.commit_tool_result(tool_call, result)

    def _apply_compaction(self, summary: str, cutoff: int, token_usage: int) -> None:
        if self.session_manager is None:
            self.context.apply_compaction(summary, cutoff)
        else:
            self.session_manager.commit_compaction(summary, cutoff, token_usage)

    def _start_memory_worker(self, start: int | None) -> None:
        if self.memory_manager is None or self.session_manager is None or start is None:
            return
        path = self.session_manager.current_path
        session_id = self.session_manager.current_session_id
        if path is None or session_id is None:
            return
        end = len(self.session_manager.snapshot().records)
        try:
            self.memory_manager.spawn_worker(path, session_id, start, end)
        except Exception as exc:
            try:
                self.memory_manager.write_state(str(exc))
            except Exception:
                pass

    def close(self) -> None:
        try:
            if self.subagent_manager is not None:
                self.subagent_manager.close()
        finally:
            try:
                if self.mcp_manager is not None:
                    self.mcp_manager.close()
            finally:
                try:
                    if self.lsp_manager is not None and self._owns_lsp_manager:
                        self.lsp_manager.close()
                finally:
                    try:
                        close = getattr(self.client, "close", None)
                        if callable(close):
                            close()
                    finally:
                        try:
                            self.context.close()
                        finally:
                            self.permission_checker.close()

    def _initialize_mcp(
        self,
    ) -> Generator[AgentEvent, AgentResponse, None]:
        if self.mcp_manager is None or self._mcp_initialized:
            return
        request = self.mcp_manager.permission_request()
        choice: PermissionChoice | None = None
        if request is not None:
            choice = yield PermissionRequestEvent(
                "mcp_project_config",
                "MCP",
                request.content,
                request.message,
            )
        warnings = self.mcp_manager.initialize(choice)
        self._mcp_initialized = True
        self.context.set_mcp_catalog(self.mcp_manager.catalog_block())
        if self._session_snapshot is not None:
            warnings.extend(
                self.mcp_manager.restore_session(
                    tuple(record.as_dict() for record in self._session_snapshot.records)
                )
            )
        self.context.set_tool_schemas(self.tools.schemas())
        for warning in warnings:
            yield ErrorEvent(warning, "mcp")

    def _restore_mcp_session(self, snapshot: Any) -> Generator[AgentEvent, None, None]:
        if self.mcp_manager is None or not self._mcp_initialized:
            return
        warnings = self.mcp_manager.restore_session(
            tuple(record.as_dict() for record in snapshot.records)
        )
        self.context.set_tool_schemas(self.tools.schemas())
        for warning in warnings:
            yield ErrorEvent(warning, "mcp")

    def _initialize_lsp(self) -> Generator[AgentEvent, None, None]:
        if self.lsp_manager is None or self._lsp_initialized:
            return
        self._lsp_initialized = True
        for warning in self.lsp_manager.initialize():
            yield ErrorEvent(warning, "lsp")


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


def _fork_system_prompt(parent: str, child: str) -> str:
    start = "User and project instructions:\n"
    end = "\n\nMode instructions:"
    parent_start = parent.find(start)
    child_start = child.find(start)
    parent_end = parent.find(end, parent_start)
    child_end = child.find(end, child_start)
    if min(parent_start, child_start, parent_end, child_end) < 0:
        return parent
    return child[:child_start] + parent[parent_start:parent_end] + child[child_end:]


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


def _persistence_error(exc: Exception) -> ErrorEvent:
    return ErrorEvent(str(exc), "session_persistence")


def _session_state(action: str, snapshot: Any) -> SessionStateEvent:
    return SessionStateEvent(
        action,
        snapshot.session_id,
        tuple(record.as_dict() for record in snapshot.records),
        snapshot.token_usage,
        snapshot.cleaned,
        snapshot.invalid,
        snapshot.restored,
    )
