from __future__ import annotations

import argparse
import sys
from typing import TextIO

from .agent import Agent
from .config import Config
from .context import ContextManager
from .openai_client import OpenAIClient


def run_repl(
    agent: Agent,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    output_stream.write("duckduckcode: 你好，我是 DuckDuckCode。输入 exit 或 quit 结束。\n")

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

        output_stream.write(f"duckduckcode: {agent.ask(user_message)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a DuckDuckCode chat session.")
    parser.add_argument("prompt", nargs="*", help="Optional first prompt.")
    args = parser.parse_args()

    try:
        config = Config.from_env()
        agent = Agent(
            OpenAIClient(api_key=config.openai_api_key, model=config.openai_model),
            ContextManager(reasoning=config.reasoning),
        )
        first_prompt = " ".join(args.prompt).strip()
        if first_prompt:
            print(agent.ask(first_prompt))
            return
        run_repl(agent)
    except RuntimeError as exc:
        parser.exit(1, f"duckduckcode: error: {exc}\n")
