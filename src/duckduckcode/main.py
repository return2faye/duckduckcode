from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .core.agent import Agent
from .core.context import ContextManager
from .core.prompts import build_system_prompt
from .interfaces.backend import run_backend
from .interfaces.tui import run_tui
from .memory import load_instructions
from .permissions import (
    PathSandbox,
    PermissionChecker,
    RulePolicy,
    check_bash_blacklist,
)
from .providers.openai.client import OpenAIClient
from .tools import (
    create_bash_tool,
    create_edit_file_tool,
    create_glob_tool,
    create_grep_tool,
    create_read_file_tool,
    create_write_file_tool,
)
from .tools.os_sandbox import OSSandbox
from .tools.tool import ToolManager, create_exit_plan_mode_tool


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a DuckDuckCode chat session.")
    parser.add_argument("--backend", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    agent: Agent | None = None
    try:
        workspace = Path.cwd().resolve()
        config = Config.from_env()
        if args.backend:
            agent = build_agent(config, workspace)
            run_backend(agent)
            return
        run_tui(
            config.openai_model,
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
) -> Agent:
    path_sandbox = PathSandbox(workspace)
    try:
        policy: RulePolicy | None = None
        os_sandbox = OSSandbox(
            workspace,
            path_sandbox.temporary_directory,
            lambda: policy is not None and policy.permission_mode != "full_access",
        )
        tools = ToolManager()
        tools.register(create_read_file_tool(workspace))
        tools.register(create_write_file_tool(workspace))
        tools.register(create_edit_file_tool(workspace))
        tools.register(create_glob_tool(workspace, path_sandbox.allowed_directories))
        tools.register(create_grep_tool(workspace, path_sandbox.allowed_directories))
        tools.register(create_bash_tool(workspace, os_sandbox))
        tools.register(create_exit_plan_mode_tool())
        policy = RulePolicy.load(
            workspace,
            path_sandbox.temporary_directory,
            {
                schema["name"]
                for schema in tools.schemas()
                if schema["name"] != "ExitPlanMode"
            },
        )
        context = ContextManager(
            system_prompt=build_system_prompt(
                workspace,
                model=config.openai_model,
                temporary_directory=path_sandbox.temporary_directory,
                tool_result_directory=path_sandbox.tool_result_directory,
                instructions=load_instructions(
                    workspace,
                    include_user=include_user_instructions,
                ),
            ),
            reasoning=config.reasoning,
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
                "or raise the threshold."
            )
        return Agent(
            OpenAIClient(
                api_key=config.openai_api_key,
                model=config.openai_model,
                langsmith_tracing=config.langsmith_tracing,
                langsmith_api_key=config.langsmith_api_key,
                langsmith_project=config.langsmith_project,
            ),
            context,
            tools,
            max_iterations=max_iterations or 50,
            permission_checker=PermissionChecker(
                [check_bash_blacklist, path_sandbox],
                policy,
            ),
            plan_file=workspace / ".duckduckcode" / "plan.md",
        )
    except Exception:
        path_sandbox.close()
        raise


if __name__ == "__main__":
    main()
