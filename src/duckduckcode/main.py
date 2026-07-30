from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TextIO

from .config import Config
from .core.agent import Agent
from .core.context import ContextManager
from .core.event import ConversationEvent, ErrorEvent
from .core.prompts import build_system_prompt
from .interfaces.backend import run_backend
from .interfaces.tui import run_tui
from .permissions import PermissionChecker, check_bash_blacklist
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


def write_stream_response(
    agent: Agent, prompt: str, output_stream: TextIO = sys.stdout
) -> None:
    for event in agent.stream(prompt):
        if isinstance(event, ConversationEvent):
            output_stream.write(event.delta)
            output_stream.flush()
        elif isinstance(event, ErrorEvent):
            output_stream.write(f"\nerror: {event.message}")
            break
    output_stream.write("\n")


def run_repl(
    agent: Agent,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    output_stream.write(
        "duckduckcode: 你好，我是 DuckDuckCode。输入 exit 或 quit 结束。\n"
    )

    while True:
        output_stream.write("you: ")
        output_stream.flush()

        user_message = input_stream.readline()
        if not user_message:
            break

        user_message = user_message.strip()
        if user_message.lower() in {"exit", "quit"}:
            break
        if not user_message:
            continue

        output_stream.write("duckduckcode: ")
        write_stream_response(agent, user_message, output_stream)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a DuckDuckCode chat session.")
    parser.add_argument("--backend", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repl", action="store_true", help="Use the simple line REPL.")
    parser.add_argument("prompt", nargs="*", help="Optional first prompt.")
    args = parser.parse_args()

    try:
        workspace = Path.cwd().resolve()
        config = Config.from_env()
        agent = build_agent(config, workspace)
        if args.backend:
            run_backend(agent)
            return

        first_prompt = " ".join(args.prompt).strip()
        if args.repl:
            run_repl(agent)
            return
        if not first_prompt or first_prompt == "tui":
            run_tui(config.openai_model, str(workspace))
            return
        if first_prompt:
            write_stream_response(agent, first_prompt)
            return
    except RuntimeError as exc:
        parser.exit(1, f"duckduckcode: error: {exc}\n")


def build_agent(config: Config, workspace: Path) -> Agent:
    tools = ToolManager()
    tools.register(create_read_file_tool(workspace))
    tools.register(create_write_file_tool(workspace))
    tools.register(create_edit_file_tool(workspace))
    tools.register(create_glob_tool(workspace))
    tools.register(create_grep_tool(workspace))
    tools.register(create_bash_tool(workspace))
    return Agent(
        OpenAIClient(api_key=config.openai_api_key, model=config.openai_model),
        ContextManager(
            system_prompt=build_system_prompt(workspace, model=config.openai_model),
            reasoning=config.reasoning,
        ),
        tools,
        permission_checker=PermissionChecker([check_bash_blacklist]),
    )


if __name__ == "__main__":
    main()
