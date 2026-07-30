from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .core.agent import Agent
from .core.context import ContextManager
from .core.prompts import build_system_prompt
from .interfaces.backend import run_backend
from .interfaces.tui import run_tui
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
from .tools.tool import ToolManager


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
        run_tui(config.openai_model, str(workspace))
    except RuntimeError as exc:
        parser.exit(1, f"duckduckcode: error: {exc}\n")
    finally:
        if agent is not None:
            agent.close()


def build_agent(config: Config, workspace: Path) -> Agent:
    path_sandbox = PathSandbox(workspace)
    try:
        tools = ToolManager()
        tools.register(create_read_file_tool(workspace))
        tools.register(create_write_file_tool(workspace))
        tools.register(create_edit_file_tool(workspace))
        tools.register(create_glob_tool(workspace, path_sandbox.allowed_directories))
        tools.register(create_grep_tool(workspace, path_sandbox.allowed_directories))
        tools.register(create_bash_tool(workspace))
        policy = RulePolicy.load(
            workspace,
            path_sandbox.temporary_directory,
            {schema["name"] for schema in tools.schemas()},
        )
        return Agent(
            OpenAIClient(api_key=config.openai_api_key, model=config.openai_model),
            ContextManager(
                system_prompt=build_system_prompt(
                    workspace,
                    model=config.openai_model,
                    temporary_directory=path_sandbox.temporary_directory,
                ),
                reasoning=config.reasoning,
            ),
            tools,
            permission_checker=PermissionChecker(
                [check_bash_blacklist, path_sandbox],
                policy,
            ),
        )
    except Exception:
        path_sandbox.close()
        raise


if __name__ == "__main__":
    main()
