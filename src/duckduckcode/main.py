from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import Config
from .core.agent import Agent, _fork_system_prompt
from .core.context import ContextManager, Message
from .core.event import (
    ConversationEvent,
    ErrorEvent,
    LoopCompleteEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from .core.prompts import build_system_prompt
from .core.skill import SkillManager
from .core.subagent import DEFINITION_TOOLS, DefinitionManager, SubagentManager
from .interfaces.backend import _event_to_json, run_backend
from .interfaces.tui import run_tui
from .memory import MemoryManager, SessionManager, load_instructions
from .permissions import (
    PathSandbox,
    PermissionChecker,
    RulePolicy,
    check_bash_blacklist,
)
from .providers import create_client
from .tools import (
    create_bash_tool,
    create_edit_file_tool,
    create_glob_tool,
    create_grep_tool,
    create_read_file_tool,
    create_write_file_tool,
)
from .tools.os_sandbox import OSSandbox
from .tools.tool import (
    QuerySource,
    ToolManager,
    create_agent_tool,
    create_exit_plan_mode_tool,
    create_load_skill_tool,
)

_PERMISSION_TOOL_NAMES = {
    "ReadFile",
    "WriteFile",
    "EditFile",
    "Glob",
    "Grep",
    "Bash",
    "LoadSkill",
    "Agent",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a DuckDuckCode chat session.")
    parser.add_argument("--backend", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--subagent-worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("prompt", nargs="?", help=argparse.SUPPRESS)
    args = parser.parse_args()

    agent: Agent | None = None
    try:
        workspace = Path.cwd().resolve()
        config = Config.from_env()
        if args.subagent_worker:
            run_subagent_worker(config)
            return
        if args.backend:
            agent = build_agent(config, workspace)
            run_backend(agent)
            return
        if args.prompt:
            agent = build_agent(config, workspace)
            for event in agent.stream(args.prompt):
                if isinstance(event, ConversationEvent):
                    print(event.delta, end="", flush=True)
                elif isinstance(event, ErrorEvent):
                    if event.code == "memory":
                        continue
                    raise RuntimeError(event.message)
            print()
            return
        run_tui(
            config.agent.model,
            str(workspace),
            permission_mode=RulePolicy.read_permission_mode(workspace),
        )
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        parser.exit(1, f"duckduckcode: error: {exc}\n")
    finally:
        if agent is not None:
            agent.close()


def build_agent(
    config: Config,
    workspace: Path,
    *,
    max_iterations: int | None = None,
    context_window_tokens: int | None = None,
    compaction_trigger_tokens: int | None = None,
    compaction_target_tokens: int | None = None,
    include_user_instructions: bool = True,
    enable_sessions: bool = True,
    enable_memory: bool = True,
    enable_skills: bool = True,
    enable_exit_plan_mode: bool = True,
    enable_subagents: bool = True,
    allowed_tools: set[str] | None = None,
    query_source: QuerySource = QuerySource.USER,
    force_os_sandbox: bool = False,
    model_role: str = "agent",
    model_override: str | None = None,
) -> Agent:
    path_sandbox = PathSandbox(workspace)
    try:
        settings = getattr(config, model_role)
        model = model_override or settings.model
        policy: RulePolicy | None = None
        os_sandbox = OSSandbox(
            workspace,
            path_sandbox.temporary_directory,
            lambda: force_os_sandbox
            or (policy is not None and policy.permission_mode != "full_access"),
        )
        tools = ToolManager(source=query_source)
        skill_manager = (
            SkillManager(workspace, builtin_commands=_builtin_slash_commands())
            if enable_skills
            else None
        )

        def enabled(name: str) -> bool:
            return allowed_tools is None or name in allowed_tools

        if enabled("ReadFile"):
            tools.register(create_read_file_tool(workspace))
        if enabled("WriteFile"):
            tools.register(create_write_file_tool(workspace))
        if enabled("EditFile"):
            tools.register(create_edit_file_tool(workspace))
        if enabled("Glob"):
            tools.register(
                create_glob_tool(workspace, path_sandbox.current_allowed_directories)
            )
        if enabled("Grep"):
            tools.register(
                create_grep_tool(workspace, path_sandbox.current_allowed_directories)
            )
        if enabled("Bash"):
            tools.register(create_bash_tool(workspace, os_sandbox))
        if enable_exit_plan_mode and enabled("ExitPlanMode"):
            tools.register(create_exit_plan_mode_tool())
        if skill_manager is not None and enabled("LoadSkill"):
            tools.register(create_load_skill_tool(skill_manager.load))
        definition_manager = None
        subagent_manager = None
        if enable_subagents and enabled("Agent"):
            definition_manager = DefinitionManager(
                workspace,
                known_tools={schema["name"] for schema in tools.schemas()},
            )
            definitions, _ = definition_manager.refresh(mark_reported=False)
            tools.register(
                create_agent_tool(
                    {
                        definition.type: definition.when_to_use
                        for definition in definitions
                    },
                    lambda **_: "Agent calls are handled by the Agent runtime.",
                )
            )
            subagent_manager = SubagentManager(
                workspace,
                definitions=definition_manager,
                parent_model=config.subagent.model,
            )
        policy = RulePolicy.load(
            workspace,
            path_sandbox.temporary_directory,
            _PERMISSION_TOOL_NAMES,
        )
        memory_manager = MemoryManager(workspace) if enable_memory else None
        memory_snapshot = (
            memory_manager.refresh(check_state=False)[0] if memory_manager else ""
        )
        context = ContextManager(
            system_prompt=build_system_prompt(
                workspace,
                model=model,
                temporary_directory=path_sandbox.temporary_directory,
                tool_result_directory=path_sandbox.tool_result_directory,
                instructions=load_instructions(
                    workspace,
                    include_user=include_user_instructions,
                ),
            ),
            reasoning=config.reasoning,
            long_term_memory=memory_snapshot,
            tool_schemas=tools.schemas(),
            tool_result_directory=path_sandbox.tool_result_directory,
            context_window_tokens=context_window_tokens or config.context_window_tokens,
            compaction_trigger_tokens=compaction_trigger_tokens,
            compaction_target_tokens=compaction_target_tokens,
        )
        static_tokens = context.estimated_tokens()
        if static_tokens >= context.auto_compact_tokens:
            raise RuntimeError(
                "Static system prompt and tool schemas are estimated at "
                f"{static_tokens} tokens, reaching the compaction trigger of "
                f"{context.auto_compact_tokens} tokens. Shorten DDCODE instructions "
                "or MEMORY content, or raise the threshold."
            )
        session_manager = (
            SessionManager(workspace, context) if enable_sessions else None
        )
        return Agent(
            create_client(config, settings, model=model_override),
            context,
            tools,
            max_iterations=max_iterations or 50,
            permission_checker=PermissionChecker(
                [check_bash_blacklist, path_sandbox],
                policy,
            ),
            plan_file=workspace / ".duckduckcode" / "plan.md",
            session_manager=session_manager,
            memory_manager=memory_manager,
            skill_manager=skill_manager,
            skill_root_callback=path_sandbox.set_skill_directories,
            fork_agent_factory=(
                (
                    lambda: build_agent(
                        config,
                        workspace,
                        max_iterations=max_iterations,
                        context_window_tokens=context_window_tokens,
                        compaction_trigger_tokens=compaction_trigger_tokens,
                        compaction_target_tokens=compaction_target_tokens,
                        include_user_instructions=include_user_instructions,
                        enable_sessions=False,
                        enable_memory=False,
                        enable_skills=False,
                        enable_exit_plan_mode=False,
                        enable_subagents=False,
                        query_source=QuerySource.SUBAGENT,
                        model_role="subagent",
                    )
                )
                if skill_manager is not None
                else None
            ),
            definition_manager=definition_manager,
            subagent_manager=subagent_manager,
            query_source=query_source,
        )
    except Exception:
        path_sandbox.close()
        raise


def run_subagent_worker(config: Config) -> None:
    line = sys.stdin.readline()
    agent: Agent | None = None
    try:
        request = json.loads(line)
        workspace = Path(request["workspace"]).resolve()
        definition = request.get("definition")
        allowed = (
            DEFINITION_TOOLS - set(definition.get("disallowed_tools", ()))
            if isinstance(definition, dict)
            else {"ReadFile", "Glob", "Grep", "WriteFile", "EditFile", "Bash"}
        )
        agent = build_agent(
            config,
            workspace,
            max_iterations=(
                int(definition["max_turns"]) if isinstance(definition, dict) else 50
            ),
            include_user_instructions=False,
            enable_sessions=False,
            enable_memory=False,
            enable_skills=False,
            enable_exit_plan_mode=False,
            enable_subagents=False,
            allowed_tools=allowed,
            query_source=QuerySource.SUBAGENT,
            force_os_sandbox=bool(request.get("isolation", False)),
            model_role="subagent",
            model_override=(str(request["model"]) if request.get("model") else None),
        )
        agent.set_permission_mode(request["permission_mode"])
        boilerplate = str(request.get("boilerplate", ""))
        if request.get("mode") == "definition":
            agent.context.system_prompt += (
                f"\n\nSubagent definition:\n{definition['body']}\n\n{boilerplate}"
            )
            prompt = str(request["prompt"])
        else:
            agent.context.system_prompt = _fork_system_prompt(
                str(request.get("system_prompt", agent.context.system_prompt)),
                agent.context.system_prompt,
            )
            agent.context.system_prompt += f"\n\n{boilerplate}"
            agent.context.long_term_memory = str(request.get("long_term_memory", ""))
            agent.context.restore(
                _completed_messages(request.get("messages", [])),
                str(request.get("abstraction", "")),
            )
            prompt = "Assigned fork task: " + str(request["prompt"])
        errors = []
        reason = None
        for event in agent.stream(prompt):
            if isinstance(event, (ToolCallEvent, ToolResultEvent, UsageEvent)):
                print(json.dumps(_event_to_json(event), ensure_ascii=False), flush=True)
            elif isinstance(event, ErrorEvent):
                errors.append(event.message)
            elif isinstance(event, LoopCompleteEvent):
                reason = event.reason
        final = next(
            (
                message.content
                for message in reversed(agent.context.messages())
                if message.role == "assistant"
                and message.kind == "message"
                and message.status == "completed"
                and message.content.strip()
            ),
            "",
        )
        if reason == "completed" and final:
            result = {"type": "worker_result", "status": "completed", "result": final}
        else:
            detail = errors[-1] if errors else reason or "no final response"
            result = {"type": "worker_result", "status": "failed", "result": detail}
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(
            json.dumps(
                {"type": "worker_result", "status": "failed", "result": str(exc)},
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        if agent is not None:
            agent.close()


def _completed_messages(values: object) -> list[Message]:
    if not isinstance(values, list):
        return []
    messages = []
    pending: dict[str, int] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        try:
            message = Message(**value)
        except TypeError:
            continue
        if message.status == "streaming":
            continue
        if message.kind == "tool_call" and message.tool_call_id is not None:
            pending[message.tool_call_id] = len(messages)
        elif message.kind == "tool_result" and message.tool_call_id is not None:
            pending.pop(message.tool_call_id, None)
        messages.append(message)
    if pending:
        del messages[min(pending.values()) :]
    return messages


def _builtin_slash_commands() -> set[str]:
    return {
        "/compact",
        "/delete-session",
        "/help",
        "/new",
        "/permissions",
        "/plan",
        "/sessions",
        "/skills",
        "/status",
    }


if __name__ == "__main__":
    main()
